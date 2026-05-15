import os
import random

import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics
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


class FrankaTwistBottleCapAction:
    def __init__(
        self,
        world,
        franka,
        deterministic_seed=42,
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
        self._joint_cmd_start_positions = None
        self._joint_cmd_target_positions = None
        self._joint_cmd_progress = 0
        self._joint_cmd_steps = 45

        self._gripper_open_pos = self._franka.gripper.joint_opened_positions
        self._gripper_closed_pos = self._franka.gripper.joint_closed_positions
        self._gripper_locked = False

        self._post_exec_action = None
        self._wait_timer = 0

        self._saved_grasp_orient = np.array([0, 1, 0, 0], dtype=np.float32)
        self._saved_approach_dir = np.array([0, 0, 0], dtype=np.float32)
        self._last_error = None

        self._twist_initialized = False
        self._twist_axis_world = None
        self._twist_prev_vec = None
        self._twist_accum_deg = 0.0
        self._twist_home_orientation = np.array([0.0, 0.70717, 0.70717, 0.0], dtype=np.float32)
        self._twist_turned_orientation = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self._wrist_roll_joint_idx = self._resolve_joint_index("panda_joint7", default_idx=6)

        self._configure_determinism()
        self._init_curobo()

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
        self._last_error = None
        self._twist_initialized = False
        self._twist_axis_world = None
        self._twist_prev_vec = None
        self._twist_accum_deg = 0.0

    def move(self, target_pose):
        self._last_error = None
        return self._plan_and_set_execution(self._to_pose(target_pose))

    def move_twist_step(self, target_pose):
        self._last_error = None
        if isinstance(target_pose, dict) and target_pose.get("kind") in ("twist", "reset"):
            return self._set_in_place_wrist_rotation_execution(target_pose)
        twist_profiles = [
            {"enable_graph": False, "max_attempts": 2, "time_dilation_factor": 0.12},
            {"enable_graph": False, "max_attempts": 4, "time_dilation_factor": 0.18},
            {"enable_graph": True, "max_attempts": 8, "time_dilation_factor": 0.25},
        ]
        return self._plan_and_set_execution(
            self._to_pose(target_pose),
            use_collision=False,
            plan_profiles=twist_profiles,
        )

    def queue_gripper_action(self, action_name):
        if action_name not in ("open_gripper", "close_gripper"):
            raise ValueError(f"unsupported gripper action: {action_name}")
        self._last_error = None
        self._post_exec_action = (action_name, None)
        self._wait_timer = 0
        return True

    def plan_lid_grasp(self, bottle_root_prim_path, grasp_id=5, pregrasp_offset=0.10):
        self._last_error = None

        grasp_pos, grasp_orient, approach_dir = self._get_grasp_world_pose(bottle_root_prim_path, grasp_id)
        if grasp_pos is None:
            self._last_error = f"grasp pose not found: {bottle_root_prim_path}/grasps/grasp_{grasp_id}"
            print(f"[ERROR] {self._last_error}")
            return False

        pre_pos = grasp_pos - approach_dir * float(pregrasp_offset)
        pre_pose = Pose(
            position=self._tensor_args.to_device(pre_pos.astype(np.float32)),
            quaternion=self._tensor_args.to_device(grasp_orient.astype(np.float32)),
        )
        if not self._plan_and_set_execution(pre_pose):
            self._last_error = "lid grasp pre-pose planning failed"
            return False

        self._saved_grasp_orient = grasp_orient.astype(np.float32)
        self._saved_approach_dir = approach_dir.astype(np.float32)
        self._post_exec_action = ("close_gripper", None)
        self._wait_timer = 0
        return True

    def build_twist_waypoints(
        self,
        lid_joint_prim_path,
        total_twist_deg=360.0,
        segments=16,
    ):
        stats = self.get_lid_twist_stats(lid_joint_prim_path)
        if stats is None:
            return None, None

        hand_pos, hand_quat_wxyz, _ = self._get_hand_world_pose()
        hand_rot = R.from_quat([hand_quat_wxyz[1], hand_quat_wxyz[2], hand_quat_wxyz[3], hand_quat_wxyz[0]])
        axis_world = stats["axis_world"]
        anchor_world = stats["anchor_world"]

        rel = hand_pos - anchor_world
        axial = axis_world * float(np.dot(rel, axis_world))
        radial = rel - axial

        n_seg = max(int(segments), 1)
        waypoints = []
        for i in range(1, n_seg + 1):
            frac = float(i) / float(n_seg)
            angle_deg = float(total_twist_deg) * frac
            rot_delta = R.from_rotvec(axis_world * np.deg2rad(angle_deg))

            if np.linalg.norm(radial) > 1e-5:
                pos_i = anchor_world + rot_delta.apply(radial) + axial
            else:
                pos_i = hand_pos.copy()

            rot_i = rot_delta * hand_rot
            q_xyzw = rot_i.as_quat()
            quat_i_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

            waypoints.append(
                {
                    "position": pos_i.astype(np.float32),
                    "orientation": quat_i_wxyz,
                }
            )

        plan_info = {
            "segments": n_seg,
            "total_twist_deg": float(total_twist_deg),
            "axis_world": axis_world.astype(np.float32),
            "anchor_world": anchor_world.astype(np.float32),
            "radial_norm": float(np.linalg.norm(radial)),
        }
        return waypoints, plan_info

    def build_ratchet_twist_phases(
        self,
        total_twist_deg=360.0,
        step_twist_deg=90.0,
    ):
        hand_pos, _, _ = self._get_hand_world_pose()
        total_twist_deg = float(total_twist_deg)
        step_twist_deg = float(step_twist_deg)
        if step_twist_deg <= 0.0:
            self._last_error = f"invalid twist step: {step_twist_deg}"
            print(f"[ERROR] {self._last_error}")
            return None, None

        stroke_count = int(np.ceil(abs(total_twist_deg) / step_twist_deg))
        stroke_count = max(stroke_count, 1)
        phases = []
        accum_deg = 0.0
        for stroke_idx in range(stroke_count):
            accum_deg = min(abs(total_twist_deg), accum_deg + step_twist_deg)
            phases.append(
                {
                    "kind": "twist",
                    "position": hand_pos.astype(np.float32),
                    "orientation": self._twist_turned_orientation.copy(),
                    "joint_delta_rad": -np.deg2rad(step_twist_deg),
                    "stroke": stroke_idx + 1,
                    "accum_target_deg": accum_deg,
                }
            )
            phases.append({"kind": "open_gripper", "stroke": stroke_idx + 1, "accum_target_deg": accum_deg})
            if stroke_idx < stroke_count - 1:
                phases.append(
                    {
                        "kind": "reset",
                        "position": hand_pos.astype(np.float32),
                        "orientation": self._twist_home_orientation.copy(),
                        "joint_delta_rad": np.deg2rad(step_twist_deg),
                        "stroke": stroke_idx + 1,
                        "accum_target_deg": accum_deg,
                    }
                )
                phases.append({"kind": "close_gripper", "stroke": stroke_idx + 1, "accum_target_deg": accum_deg})

        phases.append({"kind": "close_gripper", "stroke": stroke_count, "accum_target_deg": accum_deg})
        plan_info = {
            "strokes": stroke_count,
            "total_twist_deg": abs(total_twist_deg),
            "step_twist_deg": step_twist_deg,
            "home_orientation": self._twist_home_orientation.copy(),
            "turned_orientation": self._twist_turned_orientation.copy(),
        }
        self._last_error = None
        return phases, plan_info

    def plan_post_twist_lift(self, lift_delta_z=0.08, lid_joint_prim_path=None):
        hand_pos, hand_quat_wxyz, _ = self._get_hand_world_pose()
        lift_dir = -self._saved_approach_dir.astype(np.float32)
        lift_norm = float(np.linalg.norm(lift_dir))
        if lift_norm < 1e-6:
            lift_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            lift_dir = lift_dir / lift_norm
        target_pos = hand_pos + lift_dir * float(lift_delta_z)
        target_pose = Pose(
            position=self._tensor_args.to_device(target_pos.astype(np.float32)),
            quaternion=self._tensor_args.to_device(hand_quat_wxyz.astype(np.float32)),
        )
        return self._plan_and_set_execution(target_pose, use_collision=False)

    def reset_twist_tracking(self, lid_link_prim_path, lid_joint_prim_path):
        stats = self.get_lid_twist_stats(lid_joint_prim_path)
        if stats is None:
            return False
        self._twist_axis_world = stats["axis_world"].astype(np.float64)
        vec = self._extract_lid_reference_vector(lid_link_prim_path, self._twist_axis_world)
        if vec is None:
            self._last_error = "cannot initialize twist tracking vector"
            print(f"[ERROR] {self._last_error}")
            return False
        self._twist_prev_vec = vec
        self._twist_accum_deg = 0.0
        self._twist_initialized = True
        return True

    def get_accumulated_twist_deg(self, lid_link_prim_path, lid_joint_prim_path):
        if not self._twist_initialized:
            if not self.reset_twist_tracking(lid_link_prim_path, lid_joint_prim_path):
                return None

        vec_cur = self._extract_lid_reference_vector(lid_link_prim_path, self._twist_axis_world)
        if vec_cur is None:
            self._last_error = "cannot update twist tracking vector"
            print(f"[WARN] {self._last_error}")
            return None

        sin_term = float(np.dot(self._twist_axis_world, np.cross(self._twist_prev_vec, vec_cur)))
        cos_term = float(np.dot(self._twist_prev_vec, vec_cur))
        delta_deg = float(np.rad2deg(np.arctan2(sin_term, cos_term)))
        self._twist_accum_deg += delta_deg
        self._twist_prev_vec = vec_cur
        return float(self._twist_accum_deg)

    def get_lid_twist_stats(self, lid_joint_prim_path):
        stage = self._world.stage
        joint = UsdPhysics.RevoluteJoint.Get(stage, lid_joint_prim_path)
        if not joint:
            self._last_error = f"invalid revolute joint path: {lid_joint_prim_path}"
            print(f"[ERROR] {self._last_error}")
            return None

        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) == 0 or len(body1_targets) == 0:
            self._last_error = f"joint body targets missing: {lid_joint_prim_path}"
            print(f"[ERROR] {self._last_error}")
            return None

        body0_path = str(body0_targets[0])
        body1_path = str(body1_targets[0])
        axis_world, anchor_world = self._compute_joint_axis_anchor_world(joint, body0_path)
        if axis_world is None:
            self._last_error = f"joint axis world failed: {lid_joint_prim_path}"
            print(f"[ERROR] {self._last_error}")
            return None

        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        lower = float(lower if lower is not None else -180.0)
        upper = float(upper if upper is not None else 180.0)

        return {
            "body0_path": body0_path,
            "body1_path": body1_path,
            "axis_world": axis_world.astype(np.float32),
            "anchor_world": anchor_world.astype(np.float32),
            "lower_limit_deg": lower,
            "upper_limit_deg": upper,
        }

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
            action_name, _ = self._post_exec_action
            if action_name == "close_gripper":
                self.close_gripper()
                self._wait_timer += 1
                if self._wait_timer > 30:
                    self._gripper_locked = True
                    self._post_exec_action = None
                    self._wait_timer = 0
                return False
            if action_name == "open_gripper":
                self.open_gripper()
                self._wait_timer += 1
                if self._wait_timer > 30:
                    self._post_exec_action = None
                    self._wait_timer = 0
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

    def _set_in_place_wrist_rotation_execution(self, phase):
        if self._wrist_roll_joint_idx is None:
            self._last_error = "panda_joint7 not found for in-place wrist rotation"
            print(f"[ERROR] {self._last_error}")
            return False

        sim_js = self._franka.get_joints_state()
        start_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        target_positions = start_positions.copy()
        target_positions[self._wrist_roll_joint_idx] += float(phase["joint_delta_rad"])

        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._joint_cmd_start_positions = start_positions
        self._joint_cmd_target_positions = target_positions
        self._joint_cmd_progress = 0
        self._last_error = None
        print(
            "[INFO] In-place wrist roll, "
            f"kind={phase.get('kind')}, "
            f"joint={self._franka.dof_names[self._wrist_roll_joint_idx]}, "
            f"delta_deg={np.rad2deg(float(phase['joint_delta_rad'])):.2f}, "
            f"steps={self._joint_cmd_steps}"
        )
        return True

    def _resolve_joint_index(self, joint_name, default_idx=None):
        try:
            return self._franka.dof_names.index(joint_name)
        except ValueError:
            return default_idx

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

    def _get_hand_world_pose(self):
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

    def _get_prim_world_position(self, prim_path):
        prim = self._world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"prim not found: {prim_path}")
        xf = UsdGeom.Xformable(prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)

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

    def _compute_joint_axis_anchor_world(self, revolute_joint, body0_path):
        axis_token = str(revolute_joint.GetAxisAttr().Get())
        axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis_token.upper() == "Y":
            axis_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif axis_token.upper() == "Z":
            axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        local_pos0 = np.array(revolute_joint.GetLocalPos0Attr().Get(), dtype=np.float64)
        local_rot0 = revolute_joint.GetLocalRot0Attr().Get()
        local_rot0_m = self._quat_wxyz_to_matrix(local_rot0)

        t0 = self._get_prim_world_matrix(body0_path)
        anchor_world = self._transform_point(t0, local_pos0)
        axis_world = t0[:3, :3] @ (local_rot0_m @ axis_local)
        axis_norm = np.linalg.norm(axis_world)
        if axis_norm < 1e-8:
            return None, None
        axis_world = axis_world / axis_norm
        return axis_world, anchor_world

    def _extract_lid_reference_vector(self, lid_link_prim_path, axis_world):
        t_lid = self._get_prim_world_matrix(lid_link_prim_path)
        x_axis = t_lid[:3, 0]
        y_axis = t_lid[:3, 1]

        def _project(v):
            v_proj = v - axis_world * float(np.dot(v, axis_world))
            n = np.linalg.norm(v_proj)
            if n < 1e-8:
                return None
            return v_proj / n

        v = _project(x_axis)
        if v is None:
            v = _project(y_axis)
        return v

    @staticmethod
    def _quat_wxyz_to_matrix(q):
        if q is None:
            return np.eye(3, dtype=np.float64)
        try:
            w = float(q.GetReal())
            imag = q.GetImaginary()
            x, y, z = float(imag[0]), float(imag[1]), float(imag[2])
            return R.from_quat([x, y, z, w]).as_matrix()
        except Exception:
            arr = np.asarray(q, dtype=np.float64).reshape(-1)
            if arr.shape[0] != 4:
                return np.eye(3, dtype=np.float64)
            w, x, y, z = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            return R.from_quat([x, y, z, w]).as_matrix()

    def _get_prim_world_matrix(self, prim_path):
        prim = self._world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"prim not found: {prim_path}")
        xf = UsdGeom.Xformable(prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array([[mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)

    @staticmethod
    def _transform_point(t, p_local):
        p4 = np.array([p_local[0], p_local[1], p_local[2], 1.0], dtype=np.float64)
        pw = t @ p4
        return pw[:3]
