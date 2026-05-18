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
        self._lula_kinematics = None
        self._articulation_kinematics = None

        self._cmd_plan = None
        self._cmd_progress = 0.0
        self._target_position = np.array([3.7, 2.6, 1.5], dtype=np.float32)
        self._target_orientation = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        self._gripper_open_pos = self._franka.gripper.joint_opened_positions
        self._gripper_closed_pos = self._franka.gripper.joint_closed_positions
        self._gripper_locked = False

        self._press_active = False
        self._press_phase = None
        self._press_progress = 0
        self._raise_progress = 0
        self._press_steps = 80
        self._raise_steps = 80
        self._press_hold_seconds = 1.0
        self._press_hold_elapsed = 0.0
        self._press_distance_m = 0.05
        self._press_requested_distance_m = 0.05
        self._press_step_distance_m = 0.0
        self._raise_step_distance_m = 0.0
        self._press_start_pos = None
        self._press_bottom_pos = None
        self._press_last_pos = None
        self._press_start_quat = None
        self._press_orientation_rot = None
        self._press_start_joint_positions = None
        self._press_bottom_joint_positions = None
        self._press_target_distance_m = 0.0
        self._press_lateral_drift_m = 0.0
        self._max_joint_delta_per_step = 0.025
        self._dls_lambda = 0.08

        self._saved_grasp_orient = self._target_orientation.copy()
        self._saved_approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._pregrasp_pos = None
        self._grasp_pos = None
        self._last_error = None

        self._configure_determinism()
        self._init_curobo()
        self._init_lula_kinematics()

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
        self._press_phase = None
        self._press_progress = 0
        self._raise_progress = 0
        self._press_hold_elapsed = 0.0
        self._press_start_pos = None
        self._press_bottom_pos = None
        self._press_last_pos = None
        self._press_start_quat = None
        self._press_orientation_rot = None
        self._press_start_joint_positions = None
        self._press_bottom_joint_positions = None
        self._press_target_distance_m = 0.0
        self._press_lateral_drift_m = 0.0
        self._press_requested_distance_m = self._press_distance_m
        self._last_error = None

    def plan_target(self, target_position=None, target_orientation=None):
        self._last_error = None

        if target_position is None:
            target_position = self._target_position
        if target_orientation is None:
            target_orientation = self._target_orientation

        target_position = np.asarray(target_position, dtype=np.float32)
        target_orientation = np.asarray(target_orientation, dtype=np.float32)
        target_pose = Pose(
            position=self._tensor_args.to_device(target_position),
            quaternion=self._tensor_args.to_device(target_orientation),
        )

        if not self._plan_and_set_execution(target_pose):
            self._last_error = "shampoo target planning failed"
            return False

        self._target_position = target_position.copy()
        self._target_orientation = target_orientation.copy()
        self._saved_grasp_orient = target_orientation.copy()
        self._saved_approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._pregrasp_pos = target_position.copy()
        self._grasp_pos = target_position.copy()

        print(
            "[PRESS_TARGET] "
            f"target_position={self._target_position.tolist()}, "
            f"target_orientation={self._target_orientation.tolist()}, "
            f"approach={self._saved_approach_dir.tolist()}"
        )
        return True

    def plan_pregrasp(self, shampoo_root_prim_path=None, grasp_id=0, pregrasp_offset=0.10):
        return self.plan_target()

    def start_press(
        self,
        press_distance_m=0.05,
        steps=80,
        close_gripper=True,
        hold_seconds=1.0,
        shampoo_root_prim_path=None,
    ):
        self._last_error = None
        approach_norm = float(np.linalg.norm(self._saved_approach_dir))
        if approach_norm < 1e-6:
            self._last_error = "invalid approach direction for shampoo press"
            print(f"[ERROR] {self._last_error}")
            return False

        self._saved_approach_dir = (self._saved_approach_dir / approach_norm).astype(np.float32)
        self._press_requested_distance_m = float(press_distance_m)
        self._press_distance_m = self._press_requested_distance_m
        self._press_steps = max(int(steps), 1)
        self._raise_steps = max(int(steps), 1)
        self._press_hold_seconds = max(float(hold_seconds), 0.0)
        self._press_hold_elapsed = 0.0
        self._press_step_distance_m = self._press_distance_m / float(self._press_steps)
        self._raise_step_distance_m = self._press_distance_m / float(self._raise_steps)
        self._press_phase = "press"
        self._press_progress = 0
        self._raise_progress = 0
        self._press_start_pos, self._press_start_quat, self._press_orientation_rot = self.get_hand_world_pose()
        self._press_bottom_pos = None
        self._press_last_pos = self._press_start_pos.copy()
        self._press_start_joint_positions = None
        self._press_bottom_joint_positions = None
        self._press_target_distance_m = 0.0
        self._press_lateral_drift_m = 0.0

        if not self._prepare_press_joint_targets():
            return False

        self._press_active = True

        if close_gripper:
            self.close_gripper()
        else:
            self.open_gripper()

        print(
            "[PRESS_START] "
            f"requested_m={self._press_requested_distance_m:.4f}, "
            f"effective_m={self._press_distance_m:.4f}, "
            f"steps={self._press_steps}, "
            f"hold_s={self._press_hold_seconds:.3f}, "
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

        if self._press_bottom_pos is not None:
            end_pos = self._press_bottom_pos
        else:
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

    def step(self, step_size=None):
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
            self.close_gripper()

            if self._press_phase == "press":
                if self._press_progress >= self._press_steps:
                    self._press_bottom_pos = self._get_hand_world_position()
                    metrics = self.get_press_metrics()
                    print(
                        "[PRESS_DOWN_DONE] "
                        f"actual_m={metrics['actual_distance_m']:.4f}, "
                        f"target_m={metrics['target_distance_m']:.4f}, "
                        f"lateral_m={metrics['lateral_drift_m']:.4f}, "
                        f"steps={metrics['steps_completed']}, "
                        f"hold_s={self._press_hold_seconds:.3f}"
                    )
                    self._press_phase = "hold"
                    self._press_hold_elapsed = 0.0
                    return False

                if not self._execute_press_joint_step(pressing_down=True):
                    self._press_active = False
                    return True
                self._press_progress += 1
                self._press_last_pos = self._get_hand_world_position()
                metrics = self.get_press_metrics()
                self._press_target_distance_m = metrics["actual_distance_m"]
                self._press_lateral_drift_m = metrics["lateral_drift_m"]
                return False

            if self._press_phase == "hold":
                self._apply_press_joint_positions(self._press_bottom_joint_positions)
                self._press_last_pos = self._get_hand_world_position()
                self._press_bottom_pos = self._press_last_pos.copy()
                dt = float(step_size) if step_size is not None else 1.0 / 60.0
                self._press_hold_elapsed += max(dt, 0.0)
                if self._press_hold_elapsed >= self._press_hold_seconds:
                    self._press_phase = "raise"
                    self._raise_progress = 0
                    print(
                        "[PRESS_HOLD_DONE] "
                        f"elapsed_s={self._press_hold_elapsed:.3f}, raise_steps={self._raise_steps}"
                    )
                return False

            if self._press_phase == "raise":
                if self._raise_progress >= self._raise_steps:
                    self._press_active = False
                    self._press_phase = None
                    self._press_last_pos = self._get_hand_world_position()
                    metrics = self.get_press_metrics()
                    print(
                        "[PRESS_DONE] "
                        f"actual_m={metrics['actual_distance_m']:.4f}, "
                        f"target_m={metrics['target_distance_m']:.4f}, "
                        f"lateral_m={metrics['lateral_drift_m']:.4f}, "
                        f"press_steps={metrics['steps_completed']}, "
                        f"raise_steps={self._raise_progress}"
                    )
                    return False

                if not self._execute_press_joint_step(pressing_down=False):
                    self._press_active = False
                    return True
                self._raise_progress += 1
                self._press_last_pos = self._get_hand_world_position()
                return False

            self._last_error = f"unknown shampoo press phase: {self._press_phase}"
            print(f"[ERROR] {self._last_error}")
            self._press_active = False
            return True

        return True

    def _prepare_press_joint_targets(self):
        if self._press_start_pos is None or self._press_start_quat is None:
            self._last_error = "press start pose is not initialized"
            print(f"[ERROR] {self._last_error}")
            return False

        direction = self._saved_approach_dir.astype(np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        bottom_pos = self._press_start_pos.astype(np.float64) + direction * float(self._press_distance_m)
        target_quat = self._press_start_quat.astype(np.float32)

        ik_action, success = self._articulation_kinematics.compute_inverse_kinematics(
            target_position=bottom_pos.astype(np.float32),
            target_orientation=target_quat,
            position_tolerance=0.006,
            orientation_tolerance=0.12,
        )
        if not success:
            self._last_error = (
                "press bottom IK failed: "
                f"start_position={self._press_start_pos.tolist()}, "
                f"bottom_target={bottom_pos.astype(np.float32).tolist()}, "
                f"target_orientation={target_quat.tolist()}"
            )
            print(f"[ERROR] {self._last_error}")
            return False

        sim_js = self._franka.get_joints_state()
        self._press_start_joint_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        self._press_bottom_joint_positions = self._build_target_positions_from_ik_action(
            ik_action,
            self._press_start_joint_positions,
        )
        print(
            "[PRESS_IK] "
            f"start_position={self._press_start_pos.tolist()}, "
            f"bottom_target={bottom_pos.astype(np.float32).tolist()}, "
            f"distance_m={self._press_distance_m:.4f}, steps={self._press_steps}"
        )
        return True

    def _execute_press_joint_step(self, pressing_down):
        if self._press_start_joint_positions is None or self._press_bottom_joint_positions is None:
            self._last_error = "press joint targets are not initialized"
            print(f"[ERROR] {self._last_error}")
            return False

        if pressing_down:
            progress = self._press_progress
            steps = self._press_steps
            start_positions = self._press_start_joint_positions
            target_positions = self._press_bottom_joint_positions
        else:
            progress = self._raise_progress
            steps = self._raise_steps
            start_positions = self._press_bottom_joint_positions
            target_positions = self._press_start_joint_positions

        alpha = min(float(progress + 1) / float(max(steps, 1)), 1.0)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        positions = (1.0 - smooth_alpha) * start_positions + smooth_alpha * target_positions
        self._apply_press_joint_positions(positions)
        return True

    def _apply_press_joint_positions(self, positions):
        if positions is None:
            return

        positions = np.asarray(positions, dtype=np.float64).copy()
        positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos
        self._franka.apply_action(
            ArticulationAction(
                joint_positions=positions,
                joint_velocities=np.zeros_like(positions),
            )
        )

    def _build_target_positions_from_ik_action(self, ik_action, start_positions):
        target_positions = np.asarray(start_positions, dtype=np.float64).copy()
        ik_positions = np.asarray(ik_action.joint_positions, dtype=np.float64).reshape(-1)
        if ik_action.joint_indices is None:
            target_positions[: len(ik_positions)] = ik_positions
        else:
            target_positions[np.asarray(ik_action.joint_indices, dtype=np.int64)] = ik_positions
        target_positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos
        return target_positions

    def _cartesian_servo_step(self, direction_world, step_distance_m):
        jacobian = self._get_hand_jacobian()
        if jacobian is None:
            return self._cartesian_ik_step(direction_world, step_distance_m)

        sim_js = self._franka.get_joints_state()
        cur_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        target_positions = cur_positions.copy()

        direction_world = np.asarray(direction_world, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction_world))
        if direction_norm < 1e-9:
            self._last_error = "invalid cartesian servo direction"
            print(f"[ERROR] {self._last_error}")
            return False

        dx_world = direction_world / direction_norm * float(step_distance_m)
        _, base_rot_w = self._get_franka_base_world_pose()
        dx = base_rot_w.T.astype(np.float64) @ dx_world

        if jacobian.shape[0] >= 6:
            _, _, current_rot = self.get_hand_world_pose()
            if self._press_orientation_rot is None:
                self._press_orientation_rot = current_rot.copy()
            rot_err_world = (
                R.from_matrix(self._press_orientation_rot.astype(np.float64))
                * R.from_matrix(current_rot.astype(np.float64)).inv()
            ).as_rotvec()
            rot_err = base_rot_w.T.astype(np.float64) @ rot_err_world
            desired = np.concatenate([dx, 0.35 * rot_err])
            arm_jac = jacobian[:6, :7].astype(np.float64)
        else:
            desired = dx
            arm_jac = jacobian[:3, :7].astype(np.float64)

        damping = float(self._dls_lambda)
        jj_t = arm_jac @ arm_jac.T
        dq_arm = arm_jac.T @ np.linalg.solve(jj_t + (damping * damping) * np.eye(arm_jac.shape[0]), desired)
        dq_arm = np.clip(dq_arm, -self._max_joint_delta_per_step, self._max_joint_delta_per_step)

        target_positions[:7] = cur_positions[:7] + dq_arm
        target_positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos

        self._franka.apply_action(
            ArticulationAction(
                joint_positions=target_positions,
                joint_velocities=np.zeros_like(target_positions),
            )
        )
        return True

    def _cartesian_ik_step(self, direction_world, step_distance_m):
        direction_world = np.asarray(direction_world, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction_world))
        if direction_norm < 1e-9:
            self._last_error = "invalid cartesian IK direction"
            print(f"[ERROR] {self._last_error}")
            return False

        current_pos, _, _ = self.get_hand_world_pose()
        target_pos = current_pos.astype(np.float64) + direction_world / direction_norm * float(step_distance_m)
        target_quat = self._saved_grasp_orient.astype(np.float32)

        ik_action, success = self._articulation_kinematics.compute_inverse_kinematics(
            target_position=target_pos.astype(np.float32),
            target_orientation=target_quat,
            position_tolerance=0.004,
            orientation_tolerance=0.08,
        )
        if not success:
            self._last_error = (
                "press IK step failed: "
                f"target_position={target_pos.astype(np.float32).tolist()}, "
                f"target_orientation={target_quat.tolist()}"
            )
            print(f"[ERROR] {self._last_error}")
            return False

        sim_js = self._franka.get_joints_state()
        target_positions = np.asarray(sim_js.positions, dtype=np.float64).copy()
        ik_positions = np.asarray(ik_action.joint_positions, dtype=np.float64).reshape(-1)
        if ik_action.joint_indices is None:
            target_positions[: len(ik_positions)] = ik_positions
        else:
            target_positions[np.asarray(ik_action.joint_indices, dtype=np.int64)] = ik_positions
        target_positions[7:] = self._gripper_closed_pos if self._gripper_locked else self._gripper_open_pos

        self._franka.apply_action(
            ArticulationAction(
                joint_positions=target_positions,
                joint_velocities=np.zeros_like(target_positions),
            )
        )
        return True

    def _press_servo_step(self):
        if not self._cartesian_servo_step(self._saved_approach_dir, self._press_step_distance_m):
            return False
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
                    "[INFO] CuRobo target planning success, "
                    f"profile={idx}, graph={cfg['enable_graph']}, attempts={cfg['max_attempts']}, "
                    f"td={cfg['time_dilation_factor']}, waypoints={len(self._cmd_plan.position)}"
                )
                return True

        self._last_error = f"CuRobo target planning failed: {result.status}"
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

    def _get_hand_jacobian(self):
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
                    return jac
        return None

    def _get_hand_position_jacobian(self):
        jac = self._get_hand_jacobian()
        if jac is None:
            return None
        return jac[:3, :]

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
