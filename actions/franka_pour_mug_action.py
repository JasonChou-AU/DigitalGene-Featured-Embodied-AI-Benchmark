import os
import random

import numpy as np
import torch
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import ArticulationKinematicsSolver, interface_config_loader
from omni.isaac.motion_generation.lula import LulaKinematicsSolver


class FrankaPourMugAction:
    def __init__(
        self,
        world,
        franka,
        deterministic_seed=22,
        execution_speed_scale=0.5,
    ):
        self._world = world
        self._franka = franka
        self._deterministic_seed = int(deterministic_seed)
        self._execution_speed_scale = float(execution_speed_scale)

        self._tensor_args = TensorDeviceType()
        self._motion_gen = None
        self._usd_help = None
        self._lula_kinematics = None
        self._articulation_kinematics = None
        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._joint_cmd_start_positions = None
        self._joint_cmd_target_positions = None
        self._joint_cmd_progress = 0
        self._joint_cmd_steps = 60

        self._gripper_open_pos = self._franka.gripper.joint_opened_positions
        self._gripper_closed_pos = self._franka.gripper.joint_closed_positions
        self._gripper_locked = False

        self._post_exec_action = None
        self._wait_timer = 0

        self._saved_grasp_orient = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._saved_approach_dir = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._last_grasp_pre_pos = None
        self._gripper_length = 0.1
        self._grasp_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self._grasp_candidate_obj_path = None
        self._grasp_obj_pre_pos = None
        self._last_error = None
        self._wrist_roll_joint_idx = self._resolve_joint_index("panda_joint7", default_idx=6)

        self._configure_determinism()
        self._init_curobo()
        self._init_lula_kinematics()

    @property
    def saved_grasp_orientation(self):
        return self._saved_grasp_orient

    @property
    def saved_approach_dir(self):
        return self._saved_approach_dir

    @property
    def last_error(self):
        return self._last_error

    def is_busy(self):
        return (
            self._cmd_plan is not None
            or self._joint_cmd_target_positions is not None
            or self._post_exec_action is not None
        )

    def open_gripper(self):
        self._gripper_locked = False
        self._franka.gripper.apply_action(ArticulationAction(joint_positions=self._gripper_open_pos))

    def close_gripper(self):
        self._franka.gripper.apply_action(ArticulationAction(joint_positions=self._gripper_closed_pos))

    def release(self):
        self.open_gripper()
        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._joint_cmd_start_positions = None
        self._joint_cmd_target_positions = None
        self._joint_cmd_progress = 0
        self._post_exec_action = None
        self._wait_timer = 0
        self._grasp_candidate_obj_path = None
        self._grasp_obj_pre_pos = None
        self._last_error = None

    def move(self, target_pose, use_curobo=False):
        self._last_error = None
        pose = self._to_pose(target_pose)
        if use_curobo:
            return self._plan_and_set_execution(pose)
        return self._set_ik_joint_execution(
            target_pose=pose,
            steps=140,
            label="move without CuRobo",
        )

    def pour(self, angle_deg=90.0, steps=120):
        self._last_error = None
        return self._set_in_place_wrist_rotation_execution(
            joint_delta_rad=-np.deg2rad(float(angle_deg)),
            steps=steps,
            label="pour wrist roll",
        )

    def grasp(self, object_id_or_path, grasp_id, object_prim_path_map=None):
        self._last_error = None

        obj_path = self._resolve_object_prim_path(object_id_or_path, object_prim_path_map)
        grasp_pos, grasp_orient, approach_dir = self._get_grasp_world_pose(obj_path, grasp_id)
        if grasp_pos is None:
            self._last_error = f"grasp pose not found: {obj_path}/grasps/grasp_{grasp_id}"
            print(f"[ERROR] {self._last_error}")
            return False

        pre_pos = grasp_pos - approach_dir * self._gripper_length - self._grasp_offset
        self._last_grasp_pre_pos = np.array(pre_pos, dtype=np.float32)
        self._saved_grasp_orient = grasp_orient.astype(np.float32)
        self._saved_approach_dir = approach_dir.astype(np.float32)

        target_pose = Pose(
            position=self._tensor_args.to_device(pre_pos.astype(np.float32)),
            quaternion=self._tensor_args.to_device(grasp_orient.astype(np.float32)),
        )

        if not self._plan_and_set_execution(target_pose):
            self._last_error = f"grasp pre-pose planning failed: {self._last_error}"
            return False

        self._grasp_candidate_obj_path = obj_path
        self._grasp_obj_pre_pos = self._get_prim_world_position(obj_path)
        self._post_exec_action = ("close_and_lift", obj_path)
        self._wait_timer = 0
        return True

    def _set_post_grasp_lift(self, lift_delta_z=0.2):
        target_pos = self._get_hand_world_position()
        target_pos[2] += float(lift_delta_z)

        target_pose = Pose(
            position=self._tensor_args.to_device(target_pos.astype(np.float32)),
            quaternion=self._tensor_args.to_device(self.get_hand_world_pose()[1].astype(np.float32)),
        )
        return self._set_ik_joint_execution(
            target_pose=target_pose,
            steps=70,
            position_tolerance=0.02,
            orientation_tolerance=0.25,
            label="post-grasp lift without CuRobo",
        )

    def get_hand_world_pose(self):
        stage = self._world.stage
        hand_prim = stage.GetPrimAtPath("/World/Franka/panda_hand")
        if not hand_prim.IsValid():
            raise RuntimeError("hand prim not found: /World/Franka/panda_hand")

        xf = UsdGeom.Xformable(hand_prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        pos = np.array([t[0], t[1], t[2]], dtype=np.float32)

        x_axis = np.array([mat[0][0], mat[0][1], mat[0][2]], dtype=np.float32)
        y_axis = np.array([mat[1][0], mat[1][1], mat[1][2]], dtype=np.float32)
        z_axis = np.array([mat[2][0], mat[2][1], mat[2][2]], dtype=np.float32)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        q = R.from_matrix(rot_mat).as_quat()
        quat_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        return pos, quat_wxyz, rot_mat

    def _get_hand_world_position(self):
        hand_pos, _, _ = self.get_hand_world_pose()
        return hand_pos

    def _get_prim_world_position(self, prim_path):
        stage = self._world.stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"prim not found: {prim_path}")

        xf = UsdGeom.Xformable(prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)

    def get_gripper_downward_angle_deg(self):
        _, _, hand_rot = self.get_hand_world_pose()
        approach = hand_rot[:, 2].astype(np.float64)
        approach = approach / (np.linalg.norm(approach) + 1e-12)
        target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cos_v = float(np.clip(np.dot(approach, target), -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_v)))

    def step(self):
        if self._joint_cmd_target_positions is not None:
            alpha = min(float(self._joint_cmd_progress + 1) / float(self._joint_cmd_steps), 1.0)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target_gripper_pos = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos

            positions = (
                (1.0 - smooth_alpha) * self._joint_cmd_start_positions
                + smooth_alpha * self._joint_cmd_target_positions
            )
            velocities = np.zeros_like(positions)
            positions[7:] = target_gripper_pos

            self._franka.apply_action(
                ArticulationAction(
                    joint_positions=positions,
                    joint_velocities=velocities,
                )
            )
            self._joint_cmd_progress += 1
            if self._joint_cmd_progress < self._joint_cmd_steps:
                return False

            self._joint_cmd_start_positions = None
            self._joint_cmd_target_positions = None
            self._joint_cmd_progress = 0
            return False

        if self._cmd_plan is not None:
            last_idx = len(self._cmd_plan.position) - 1
            if self._cmd_progress > last_idx:
                self._cmd_plan = None
                self._cmd_progress = 0.0

            else:
                idx0 = int(np.floor(self._cmd_progress))
                idx1 = min(idx0 + 1, last_idx)
                alpha = float(self._cmd_progress - idx0)

                cmd0 = self._cmd_plan[idx0]
                cmd1 = self._cmd_plan[idx1]
                raw_positions0 = cmd0.position.cpu().numpy().flatten()
                raw_positions1 = cmd1.position.cpu().numpy().flatten()
                raw_velocities0 = cmd0.velocity.cpu().numpy().flatten()
                raw_velocities1 = cmd1.velocity.cpu().numpy().flatten()

                raw_positions = (1.0 - alpha) * raw_positions0 + alpha * raw_positions1
                raw_velocities = ((1.0 - alpha) * raw_velocities0 + alpha * raw_velocities1) * self._execution_speed_scale
                target_gripper_pos = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos
                num_dof = self._franka.num_dof

                if len(raw_positions) == num_dof:
                    full_positions = raw_positions.copy()
                    full_positions[7:] = target_gripper_pos
                    full_velocities = raw_velocities.copy()
                    full_velocities[7:] = 0.0
                else:
                    full_positions = np.concatenate([raw_positions[:7], target_gripper_pos])
                    full_velocities = np.concatenate([raw_velocities[:7], [0.0, 0.0]])

                art_action = ArticulationAction(
                    joint_positions=full_positions,
                    joint_velocities=full_velocities,
                )
                self._franka.apply_action(art_action)
                self._cmd_progress += self._execution_speed_scale
                return False

        if self._post_exec_action is not None:
            action_name, obj_path = self._post_exec_action

            if action_name == "close_and_lift":
                self.close_gripper()
                self._wait_timer += 1

                if self._wait_timer > 30:
                    self._gripper_locked = True
                    self._wait_timer = 0
                    self._post_exec_action = None
                    print("[INFO] Gripper closed; skipping post-grasp lift and moving directly to MOVE_PLAN.")
                    return True
                return False

        return True

    def _init_curobo(self):
        robot_cfg_path = get_robot_configs_path()
        robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]

        world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self._tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=4,
            num_graph_seeds=2,
            interpolation_dt=0.05,
            collision_cache={"obb": 4, "mesh": 4},
        )
        self._motion_gen = MotionGen(motion_gen_config)

        if os.environ.get("CUROBO_ENABLE_WARMUP", "0") == "1":
            self._motion_gen.warmup(enable_graph=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self._usd_help = UsdHelper()
        self._usd_help.load_stage(self._world.stage)

    def _init_lula_kinematics(self):
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._lula_kinematics = LulaKinematicsSolver(**kinematics_config)
        base_pos, base_rot = self._get_franka_base_world_pose()
        base_quat_xyzw = R.from_matrix(base_rot).as_quat()
        base_quat_wxyz = np.array(
            [base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]],
            dtype=np.float32,
        )
        self._lula_kinematics.set_robot_base_pose(base_pos, base_quat_wxyz)
        self._articulation_kinematics = ArticulationKinematicsSolver(
            self._franka,
            self._lula_kinematics,
            "panda_hand",
        )

    def _configure_determinism(self):
        seed = self._deterministic_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _set_plan_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_pose(self, target_pose):
        if isinstance(target_pose, Pose):
            return target_pose
        if isinstance(target_pose, dict):
            pos = np.array(target_pose["position"], dtype=np.float32)
            quat = np.array(target_pose["orientation"], dtype=np.float32)
            return Pose(
                position=self._tensor_args.to_device(pos),
                quaternion=self._tensor_args.to_device(quat),
            )
        raise TypeError("target_pose must be curobo Pose or {'position': [x,y,z], 'orientation': [w,x,y,z]}")

    def _set_ik_joint_execution(
        self,
        target_pose,
        steps=60,
        position_tolerance=0.01,
        orientation_tolerance=0.2,
        label="IK joint execution",
    ):
        target_pos = target_pose.position.detach().cpu().numpy().reshape(-1)[:3].astype(np.float32)
        target_quat_wxyz = target_pose.quaternion.detach().cpu().numpy().reshape(-1)[:4].astype(np.float32)

        ik_action, success = self._articulation_kinematics.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_quat_wxyz,
            position_tolerance=float(position_tolerance),
            orientation_tolerance=float(orientation_tolerance),
        )
        if not success:
            self._last_error = f"{label} failed"
            print(f"[ERROR] {self._last_error}")
            return False

        sim_js = self._franka.get_joints_state()
        start_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        target_positions = self._build_target_positions_from_ik_action(ik_action, start_positions=start_positions)

        self._set_joint_target_execution(start_positions, target_positions, steps=steps)
        print(f"[INFO] {label}: IK success, steps={self._joint_cmd_steps}")
        return True

    def _set_in_place_wrist_rotation_execution(self, joint_delta_rad, steps=120, label="wrist roll"):
        if self._wrist_roll_joint_idx is None:
            self._last_error = "panda_joint7 not found for in-place wrist rotation"
            print(f"[ERROR] {self._last_error}")
            return False

        sim_js = self._franka.get_joints_state()
        start_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        target_positions = start_positions.copy()
        target_positions[self._wrist_roll_joint_idx] += float(joint_delta_rad)
        target_positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos

        self._set_joint_target_execution(start_positions, target_positions, steps=steps)
        self._last_error = None
        print(
            "[INFO] In-place wrist roll, "
            f"label={label}, "
            f"joint={self._franka.dof_names[self._wrist_roll_joint_idx]}, "
            f"delta_deg={np.rad2deg(float(joint_delta_rad)):.2f}, "
            f"steps={self._joint_cmd_steps}"
        )
        return True

    def _set_joint_target_execution(self, start_positions, target_positions, steps):
        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._joint_cmd_start_positions = np.asarray(start_positions, dtype=np.float64).copy()
        self._joint_cmd_target_positions = np.asarray(target_positions, dtype=np.float64).copy()
        self._joint_cmd_progress = 0
        self._joint_cmd_steps = max(int(steps), 1)
        self._last_error = None

    def _build_target_positions_from_ik_action(self, ik_action, start_positions=None):
        if start_positions is None:
            sim_js = self._franka.get_joints_state()
            target_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        else:
            target_positions = np.asarray(start_positions, dtype=np.float64).copy()

        ik_positions = np.asarray(ik_action.joint_positions, dtype=np.float64).reshape(-1)
        if ik_action.joint_indices is None:
            target_positions[: len(ik_positions)] = ik_positions
        else:
            target_positions[np.asarray(ik_action.joint_indices, dtype=np.int64)] = ik_positions
        target_positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos
        return target_positions

    def _resolve_joint_index(self, joint_name, default_idx=None):
        try:
            return self._franka.dof_names.index(joint_name)
        except ValueError:
            return default_idx

    def _resolve_object_prim_path(self, object_id_or_path, object_prim_path_map=None):
        if isinstance(object_id_or_path, str):
            return object_id_or_path
        if isinstance(object_id_or_path, int):
            if object_prim_path_map is not None and object_id_or_path in object_prim_path_map:
                return object_prim_path_map[object_id_or_path]
            return f"/World/object{object_id_or_path}"
        raise TypeError("object_id_or_path must be object prim path string or object_id int")

    def _get_grasp_world_pose(self, obj_path, grasp_id):
        stage = self._world.stage
        target_path = f"{obj_path}/grasps/grasp_{grasp_id}"
        grasp_prim = stage.GetPrimAtPath(target_path)
        if not grasp_prim.IsValid():
            return None, None, None

        xf = UsdGeom.Xformable(grasp_prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        pos = np.array([t[0], t[1], t[2]], dtype=np.float32)

        x_axis = np.array([mat[0][0], mat[0][1], mat[0][2]], dtype=np.float32)
        y_axis = np.array([mat[1][0], mat[1][1], mat[1][2]], dtype=np.float32)
        z_axis = np.array([mat[2][0], mat[2][1], mat[2][2]], dtype=np.float32)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)

        rot_mat_clean = np.stack([x_axis, y_axis, z_axis], axis=1)
        approach_dir_world = z_axis / np.linalg.norm(z_axis)
        q = R.from_matrix(rot_mat_clean).as_quat()
        orientation = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        return pos, orientation, approach_dir_world

    def _get_franka_base_world_pose(self):
        stage = self._world.stage
        base_prim = stage.GetPrimAtPath("/World/Franka")
        if not base_prim.IsValid():
            raise RuntimeError("franka base prim not found: /World/Franka")

        xf = UsdGeom.Xformable(base_prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        pos = np.array([t[0], t[1], t[2]], dtype=np.float32)

        x_axis = np.array([mat[0][0], mat[0][1], mat[0][2]], dtype=np.float32)
        y_axis = np.array([mat[1][0], mat[1][1], mat[1][2]], dtype=np.float32)
        z_axis = np.array([mat[2][0], mat[2][1], mat[2][2]], dtype=np.float32)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        return pos, rot_mat

    def _world_pose_to_franka_base_pose(self, target_pose: Pose):
        base_pos_w, base_rot_w = self._get_franka_base_world_pose()
        base_rot_inv = base_rot_w.T

        target_pos_w = target_pose.position.detach().cpu().numpy().reshape(-1)[:3].astype(np.float32)
        target_quat_wxyz = target_pose.quaternion.detach().cpu().numpy().reshape(-1)[:4].astype(np.float32)

        target_pos_r = base_rot_inv @ (target_pos_w - base_pos_w)

        target_rot_w = R.from_quat(
            [target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]]
        ).as_matrix()
        target_rot_r = base_rot_inv @ target_rot_w
        q_xyzw = R.from_matrix(target_rot_r).as_quat()
        target_quat_r_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        return Pose(
            position=self._tensor_args.to_device(target_pos_r.astype(np.float32)),
            quaternion=self._tensor_args.to_device(target_quat_r_wxyz.astype(np.float32)),
        )

    def _update_world_collision(self):
        ignore_list = ["/World/Franka", "/curobo", "/visual"]
        world_cfg = self._usd_help.get_obstacles_from_stage(
            reference_prim_path="/World/Franka",
            ignore_substring=ignore_list,
        )
        self._motion_gen.update_world(world_cfg.get_collision_check_world())

    def _plan_and_set_execution(self, target_pose, disable_collision=False):
        target_pose = self._world_pose_to_franka_base_pose(target_pose)
        cu_js = self._get_cu_joint_state()

        if disable_collision:
            self._motion_gen.update_world(WorldConfig().get_collision_check_world())
            print("[INFO] CuRobo planning with collision disabled.")
        else:
            self._update_world_collision()

        plan_profiles = [
            {"enable_graph": False, "max_attempts": 3, "time_dilation_factor": 0.15},
            {"enable_graph": False, "max_attempts": 6, "time_dilation_factor": 0.25},
            {"enable_graph": True, "max_attempts": 8, "time_dilation_factor": 0.25},
            {"enable_graph": True, "max_attempts": 12, "time_dilation_factor": 0.35},
            {"enable_graph": True, "max_attempts": 20, "time_dilation_factor": 0.5},
        ]

        result = None
        for idx, cfg in enumerate(plan_profiles):
            self._set_plan_seed(self._deterministic_seed + idx)
            plan_config = MotionGenPlanConfig(
                enable_graph=cfg["enable_graph"],
                max_attempts=cfg["max_attempts"],
                time_dilation_factor=cfg["time_dilation_factor"],
            )
            result = self._motion_gen.plan_single(cu_js.unsqueeze(0), target_pose, plan_config)
            if result.success.item():
                full_plan = result.get_interpolated_plan()
                self._joint_cmd_start_positions = None
                self._joint_cmd_target_positions = None
                self._joint_cmd_progress = 0
                self._cmd_plan = self._motion_gen.get_full_js(full_plan)
                self._cmd_progress = 0.0
                self._last_error = None
                print(
                    "[INFO] CuRobo planning success, "
                    f"profile={idx}, graph={cfg['enable_graph']}, attempts={cfg['max_attempts']}, "
                    f"td={cfg['time_dilation_factor']}, waypoints={len(self._cmd_plan.position)}"
                )
                return True

        self._last_error = f"CuRobo planning failed: {result.status}"
        print(f"[ERROR] {self._last_error}")
        return False

    def _get_cu_joint_state(self):
        sim_js = self._franka.get_joints_state()
        return JointState(
            position=self._tensor_args.to_device(sim_js.positions),
            velocity=self._tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self._tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self._tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self._franka.dof_names,
        ).get_ordered_joint_state(self._motion_gen.kinematics.joint_names)
