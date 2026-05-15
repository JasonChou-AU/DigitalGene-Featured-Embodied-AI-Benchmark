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


class FrankaPressShampooAction:
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

        self._gripper_open_pos = self._franka.gripper.joint_opened_positions
        self._gripper_closed_pos = self._franka.gripper.joint_closed_positions
        self._gripper_locked = False

        self._press_active = False
        self._press_progress = 0
        self._press_steps = 80
        self._press_distance_m = 0.10
        self._press_step_distance_m = 0.0
        self._press_start_pos = None
        self._press_last_pos = None
        self._press_target_distance_m = 0.0
        self._press_lateral_drift_m = 0.0
        self._max_joint_delta_per_step = 0.025
        self._dls_lambda = 0.08

        self._saved_grasp_orient = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._saved_approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._pregrasp_pos = None
        self._grasp_pos = None
        self._last_error = None

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

    @property
    def pregrasp_position(self):
        return self._pregrasp_pos

    @property
    def grasp_position(self):
        return self._grasp_pos

    def is_busy(self):
        return self._cmd_plan is not None or self._press_active

    def open_gripper(self):
        self._gripper_locked = False
        self._franka.gripper.apply_action(ArticulationAction(joint_positions=self._gripper_open_pos))

    def close_gripper(self):
        self._gripper_locked = True
        self._franka.gripper.apply_action(ArticulationAction(joint_positions=self._gripper_closed_pos))

    def release(self):
        self.open_gripper()
        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._press_active = False
        self._press_progress = 0
        self._press_start_pos = None
        self._press_last_pos = None
        self._press_target_distance_m = 0.0
        self._press_lateral_drift_m = 0.0
        self._last_error = None

    def plan_pregrasp(self, shampoo_root_prim_path, grasp_id=0, pregrasp_offset=0.10):
        self._last_error = None

        grasp_pos, grasp_orient, approach_dir = self._get_grasp_world_pose(shampoo_root_prim_path, grasp_id)
        if grasp_pos is None:
            self._last_error = f"grasp pose not found: {shampoo_root_prim_path}/grasps/grasp_{grasp_id}"
            print(f"[ERROR] {self._last_error}")
            return False

        pregrasp_pos = grasp_pos - approach_dir * float(pregrasp_offset)
        target_pose = Pose(
            position=self._tensor_args.to_device(pregrasp_pos.astype(np.float32)),
            quaternion=self._tensor_args.to_device(grasp_orient.astype(np.float32)),
        )

        if not self._plan_and_set_execution(target_pose):
            self._last_error = "shampoo pregrasp planning failed"
            return False

        self._saved_grasp_orient = grasp_orient.astype(np.float32)
        self._saved_approach_dir = approach_dir.astype(np.float32)
        self._pregrasp_pos = pregrasp_pos.astype(np.float32)
        self._grasp_pos = grasp_pos.astype(np.float32)

        print(
            "[PRESS_PREGRASP] "
            f"root={shampoo_root_prim_path}, grasp={grasp_id}, "
            f"pregrasp={self._pregrasp_pos.tolist()}, "
            f"approach={self._saved_approach_dir.tolist()}, "
            f"offset_m={float(pregrasp_offset):.4f}"
        )
        return True

    def start_press(self, press_distance_m=0.10, steps=80, close_gripper=False):
        self._last_error = None
        approach_norm = float(np.linalg.norm(self._saved_approach_dir))
        if approach_norm < 1e-6:
            self._last_error = "invalid approach direction for shampoo press"
            print(f"[ERROR] {self._last_error}")
            return False

        self._saved_approach_dir = (self._saved_approach_dir / approach_norm).astype(np.float32)
        self._press_distance_m = float(press_distance_m)
        self._press_steps = max(int(steps), 1)
        self._press_step_distance_m = self._press_distance_m / float(self._press_steps)
        self._press_progress = 0
        self._press_start_pos = self._get_hand_world_position()
        self._press_last_pos = self._press_start_pos.copy()
        self._press_target_distance_m = 0.0
        self._press_lateral_drift_m = 0.0
        self._press_active = True

        if close_gripper:
            self.close_gripper()
        else:
            self.open_gripper()

        print(
            "[PRESS_START] "
            f"distance_m={self._press_distance_m:.4f}, steps={self._press_steps}, "
            f"approach={self._saved_approach_dir.tolist()}, close_gripper={bool(close_gripper)}"
        )
        return True

    def get_press_metrics(self):
        if self._press_start_pos is None:
            return {
                "target_distance_m": 0.0,
                "actual_distance_m": 0.0,
                "lateral_drift_m": 0.0,
                "steps_completed": int(self._press_progress),
            }

        end_pos = self._press_last_pos if self._press_last_pos is not None else self._get_hand_world_position()
        delta = np.asarray(end_pos, dtype=np.float32) - np.asarray(self._press_start_pos, dtype=np.float32)
        approach = self._saved_approach_dir.astype(np.float32)
        approach = approach / (np.linalg.norm(approach) + 1e-12)
        actual_distance = float(np.dot(delta, approach))
        lateral_vec = delta - approach * actual_distance
        lateral_drift = float(np.linalg.norm(lateral_vec))
        return {
            "target_distance_m": float(self._press_distance_m),
            "actual_distance_m": actual_distance,
            "lateral_drift_m": lateral_drift,
            "steps_completed": int(self._press_progress),
        }

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

                self._franka.apply_action(
                    ArticulationAction(
                        joint_positions=full_positions,
                        joint_velocities=full_velocities,
                    )
                )
                self._cmd_progress += self._execution_speed_scale
                return False

        if self._press_active:
            if self._press_progress >= self._press_steps:
                self._press_active = False
                self._press_last_pos = self._get_hand_world_position()
                metrics = self.get_press_metrics()
                print(
                    "[PRESS_DONE] "
                    f"actual_m={metrics['actual_distance_m']:.4f}, "
                    f"target_m={metrics['target_distance_m']:.4f}, "
                    f"lateral_m={metrics['lateral_drift_m']:.4f}, "
                    f"steps={metrics['steps_completed']}"
                )
                return False

            if not self._press_servo_step():
                self._press_active = False
                return True
            return False

        return True

    def _press_servo_step(self):
        jacobian = self._get_hand_position_jacobian()
        if jacobian is None:
            self._last_error = "cannot query Franka hand jacobian for press servo"
            print(f"[ERROR] {self._last_error}")
            return False

        sim_js = self._franka.get_joints_state()
        cur_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        target_positions = cur_positions.copy()

        dx_world = self._saved_approach_dir.astype(np.float64) * float(self._press_step_distance_m)
        _, base_rot_w = self._get_franka_base_world_pose()
        dx = base_rot_w.T.astype(np.float64) @ dx_world
        arm_jac = jacobian[:, :7].astype(np.float64)
        damping = float(self._dls_lambda)
        jj_t = arm_jac @ arm_jac.T
        dq_arm = arm_jac.T @ np.linalg.solve(jj_t + (damping * damping) * np.eye(3), dx)
        dq_arm = np.clip(dq_arm, -self._max_joint_delta_per_step, self._max_joint_delta_per_step)

        target_positions[:7] = cur_positions[:7] + dq_arm
        target_positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos

        self._franka.apply_action(
            ArticulationAction(
                joint_positions=target_positions,
                joint_velocities=np.zeros_like(target_positions),
            )
        )

        self._press_progress += 1
        self._press_last_pos = self._get_hand_world_position()
        metrics = self.get_press_metrics()
        self._press_target_distance_m = metrics["actual_distance_m"]
        self._press_lateral_drift_m = metrics["lateral_drift_m"]
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
        return pos, orientation, approach_dir_world.astype(np.float32)

    def _get_hand_world_position(self):
        stage = self._world.stage
        hand_prim = stage.GetPrimAtPath("/World/Franka/panda_hand")
        if not hand_prim.IsValid():
            raise RuntimeError("hand prim not found: /World/Franka/panda_hand")

        xf = UsdGeom.Xformable(hand_prim)
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

    def _world_pose_to_franka_base_pose(self, target_pose):
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

    def _plan_and_set_execution(self, target_pose):
        target_pose = self._world_pose_to_franka_base_pose(target_pose)
        cu_js = self._get_cu_joint_state()
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
                self._cmd_plan = self._motion_gen.get_full_js(full_plan)
                self._cmd_progress = 0.0
                self._last_error = None
                print(
                    "[INFO] CuRobo pregrasp planning success, "
                    f"profile={idx}, graph={cfg['enable_graph']}, attempts={cfg['max_attempts']}, "
                    f"td={cfg['time_dilation_factor']}, waypoints={len(self._cmd_plan.position)}"
                )
                return True

        self._last_error = f"CuRobo pregrasp planning failed: {result.status}"
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

    def _get_hand_position_jacobian(self):
        if not hasattr(self._franka, "get_jacobians"):
            return None

        jacobians = np.asarray(self._franka.get_jacobians())
        if jacobians.ndim == 4:
            jacobians = jacobians[0]
        if jacobians.ndim != 3 or jacobians.shape[1] < 3:
            return None

        body_index = self._resolve_hand_body_index()
        candidate_indices = []
        if body_index is not None:
            candidate_indices.extend([body_index, body_index - 1])
        candidate_indices.extend([jacobians.shape[0] - 1, max(jacobians.shape[0] - 2, 0)])

        for idx in candidate_indices:
            if 0 <= idx < jacobians.shape[0]:
                jac = np.asarray(jacobians[idx], dtype=np.float64)
                if jac.shape[0] >= 3 and jac.shape[1] >= 7 and np.linalg.norm(jac[:3, :7]) > 1e-9:
                    return jac[:3, :]
        return None

    def _resolve_hand_body_index(self):
        if hasattr(self._franka, "get_body_index"):
            for name in ("panda_hand", "panda_link8", "panda_rightfinger", "panda_leftfinger"):
                try:
                    idx = self._franka.get_body_index(name)
                    if idx is not None and int(idx) >= 0:
                        return int(idx)
                except Exception:
                    pass

        body_names = getattr(self._franka, "body_names", None)
        if body_names is not None:
            for name in ("panda_hand", "panda_link8", "panda_rightfinger", "panda_leftfinger"):
                if name in body_names:
                    return int(body_names.index(name))
        return None
