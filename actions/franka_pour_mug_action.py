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

        self._cmd_plan = None
        self._cmd_progress = 0.0

        self._post_exec_action = None
        self._wait_timer = 0

        self._gripper_open_pos = self._franka.gripper.joint_opened_positions
        self._gripper_closed_pos = self._franka.gripper.joint_closed_positions
        self._gripper_locked = False

        self._last_error = None
        self._saved_grasp_orient = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._saved_approach_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._gripper_length = 0.1

        self._configure_determinism()
        self._init_curobo()

        self._grasp_offset = np.array([0.0, 0.02, 0.0], dtype=np.float32)

    @property
    def last_error(self):
        return self._last_error

    @property
    def saved_grasp_orientation(self):
        return self._saved_grasp_orient

    @property
    def saved_approach_dir(self):
        return self._saved_approach_dir

    def is_busy(self):
        return self._cmd_plan is not None or self._post_exec_action is not None

    def open_gripper(self):
        self._gripper_locked = False
        self._franka.gripper.apply_action(ArticulationAction(joint_positions=self._gripper_open_pos))

    def close_gripper(self):
        self._franka.gripper.apply_action(ArticulationAction(joint_positions=self._gripper_closed_pos))

    def release(self):
        self.open_gripper()
        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._post_exec_action = None
        self._wait_timer = 0
        self._last_error = None

    def move(self, target_pose, use_collision=True):
        self._last_error = None
        return self._plan_and_set_execution(self._to_pose(target_pose), use_collision=bool(use_collision))

    def grasp(self, object_id_or_path, grasp_id, object_prim_path_map=None, pregrasp_offset=None):
        self._last_error = None

        obj_path = self._resolve_object_prim_path(object_id_or_path, object_prim_path_map)
        grasp_pos, grasp_orient, approach_dir = self._get_grasp_world_pose(obj_path, grasp_id)
        if grasp_pos is None:
            self._last_error = f"grasp pose not found: {obj_path}/grasps/grasp_{grasp_id}"
            print(f"[ERROR] {self._last_error}")
            return False

        offset = self._gripper_length if pregrasp_offset is None else float(pregrasp_offset)
        pre_pos = grasp_pos - approach_dir * float(offset) - self._grasp_offset
        pre_pose = {
            "position": pre_pos.astype(np.float32),
            "orientation": grasp_orient.astype(np.float32),
        }

        if not self.move(pre_pose, use_collision=True):
            self._last_error = f"grasp pre-pose planning failed: {self._last_error}"
            return False

        self._saved_grasp_orient = grasp_orient.astype(np.float32)
        self._saved_approach_dir = approach_dir.astype(np.float32)
        # User requirement: pregrasp pose itself is the best grasp pose.
        self._post_exec_action = ("close_only", None)
        self._wait_timer = 0
        return True

    def plan_lift(self, lift_delta_z=0.5, disable_collision=True):
        self._last_error = None
        hand_pos, _, _ = self.get_hand_world_pose()

        target_pos = hand_pos.copy()
        target_pos[2] += float(lift_delta_z)
        # Keep lift orientation consistent with grasp to avoid unnecessary wrist rotation.
        target_quat_wxyz = self._saved_grasp_orient.astype(np.float32)
        target_pose = {
            "position": target_pos.astype(np.float32),
            "orientation": target_quat_wxyz,
        }
        lift_plan_profiles = [
            {"enable_graph": False, "max_attempts": 8, "time_dilation_factor": 0.2},
            {"enable_graph": False, "max_attempts": 12, "time_dilation_factor": 0.3},
            {"enable_graph": False, "max_attempts": 20, "time_dilation_factor": 0.4},
        ]
        ok = self._plan_and_set_execution(
            self._to_pose(target_pose),
            use_collision=not bool(disable_collision),
            plan_profiles=lift_plan_profiles,
        )
        if not ok:
            self._last_error = f"lift planning failed: {self._last_error}"
            return False
        return True

    def plan_pour_orientation_down(self, keep_position=True, disable_collision=True):
        self._last_error = None
        hand_pos, _, hand_rot = self.get_hand_world_pose()

        current_approach = hand_rot[:, 2].astype(np.float64)
        target_approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        align_rot = self._rotation_align_a_to_b(current_approach, target_approach)
        target_rot = align_rot @ hand_rot.astype(np.float64)

        q_xyzw = R.from_matrix(target_rot).as_quat()
        target_quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        target_pos = hand_pos.copy() if keep_position else hand_pos.copy()
        target_pose = {
            "position": target_pos.astype(np.float32),
            "orientation": target_quat_wxyz,
        }
        ok = self._plan_and_set_execution(self._to_pose(target_pose), use_collision=not bool(disable_collision))
        if not ok:
            self._last_error = f"pour orientation planning failed: {self._last_error}"
            return False
        return True

    def get_gripper_downward_angle_deg(self):
        _, _, hand_rot = self.get_hand_world_pose()
        approach = hand_rot[:, 2].astype(np.float64)
        approach = approach / (np.linalg.norm(approach) + 1e-12)
        target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cos_v = float(np.clip(np.dot(approach, target), -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_v)))

    def step(self):
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
            action_name, action_data = self._post_exec_action
            if action_name == "close_only":
                self.close_gripper()
                self._wait_timer += 1
                if self._wait_timer > 30:
                    self._gripper_locked = True
                    self._post_exec_action = None
                    self._wait_timer = 0
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

    def _plan_and_set_execution(self, target_pose, use_collision=True, plan_profiles=None):
        target_pose = self._world_pose_to_franka_base_pose(target_pose)
        cu_js = self._get_cu_joint_state()

        if use_collision:
            self._update_world_collision()
        else:
            self._motion_gen.update_world(WorldConfig().get_collision_check_world())

        if plan_profiles is None:
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
                self._cmd_plan = self._motion_gen.get_full_js(full_plan)
                self._cmd_progress = 0.0
                self._last_error = None
                print(
                    "[INFO] CuRobo planning success, "
                    f"profile={idx}, graph={cfg['enable_graph']}, attempts={cfg['max_attempts']}, "
                    f"td={cfg['time_dilation_factor']}, use_collision={use_collision}, "
                    f"waypoints={len(self._cmd_plan.position)}"
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

    @staticmethod
    def _rotation_align_a_to_b(vec_a, vec_b):
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)

        dot_ab = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dot_ab > 1.0 - 1e-6:
            return np.eye(3, dtype=np.float64)

        if dot_ab < -1.0 + 1e-6:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            return R.from_rotvec(axis * np.pi).as_matrix()

        axis = np.cross(a, b)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return np.eye(3, dtype=np.float64)
        axis = axis / axis_norm
        angle = float(np.arctan2(axis_norm, dot_ab))
        return R.from_rotvec(axis * angle).as_matrix()
