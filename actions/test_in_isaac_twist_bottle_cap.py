import os

import numpy as np
from omni.isaac.examples.base_sample import BaseSample

from actions.franka_twist_bottle_cap_action import FrankaTwistBottleCapAction
from actions.scene_object_loader import setup_scene_from_config


def evaluate_twist_and_separation(
    twist_deg,
    lid_pos,
    body_pos,
    twist_success_deg=300.0,
    separation_success_m=0.04,
):
    lid_pos = np.asarray(lid_pos, dtype=np.float32)
    body_pos = np.asarray(body_pos, dtype=np.float32)
    sep_dist = float(np.linalg.norm(lid_pos - body_pos))

    twist_ok = float(abs(twist_deg)) >= float(twist_success_deg)
    separation_ok = sep_dist >= float(separation_success_m)
    success = bool(twist_ok and separation_ok)
    return {
        "success": success,
        "twist_ok": bool(twist_ok),
        "separation_ok": bool(separation_ok),
        "twist_deg": float(twist_deg),
        "separation_m": float(sep_dist),
        "twist_threshold_deg": float(twist_success_deg),
        "separation_threshold_m": float(separation_success_m),
    }


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._state = "WAIT"
        self._settle_steps = 120
        self._settle_counter = 0

        self._world = None
        self._twist_action = None

        self._scene_config_path = os.path.join(os.path.dirname(__file__), "scene_objects.json")
        self._object_prim_path_map = {}

        # User fixed spec
        self._target_object_id = 3
        self._target_grasp_id = 5

        # Twist config
        self._target_twist_deg = 360.0
        self._twist_segments = 18
        self._twist_success_deg = 300.0
        self._separation_success_m = 0.04
        self._post_lift_delta_z = 0.08

        self._bottle_root_path = None
        self._lid_link_path = None
        self._body_link_path = None
        self._lid_joint_path = None

        self._planned_waypoints = []
        self._plan_info = None
        self._twist_wp_idx = 0
        self._latest_twist_deg = 0.0

        self._physics_cb_name = "sim_step"
        self._physics_cb_registered = False
        self._failure_reason = None
        self._failure_reported = False

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        setup_result = setup_scene_from_config(world, self._scene_config_path)
        self._franka = setup_result["franka"]
        self._object_prim_path_map = setup_result["object_prim_path_map"]

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("franka")
        self._twist_action = FrankaTwistBottleCapAction(world=self._world, franka=self._franka)
        self._twist_action.open_gripper()

        self._resolve_bottle_paths()
        self._reset_state_vars()
        self._register_physics_callback()
        await self._world.play_async()

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._resolve_bottle_paths()
        self._reset_state_vars()
        if self._twist_action is not None:
            self._twist_action.release()
            self._twist_action.open_gripper()
        self._register_physics_callback()
        await self._world.play_async()

    def _reset_state_vars(self):
        self._state = "WAIT"
        self._settle_counter = 0
        self._planned_waypoints = []
        self._plan_info = None
        self._twist_wp_idx = 0
        self._latest_twist_deg = 0.0
        self._failure_reason = None
        self._failure_reported = False

    def _resolve_bottle_paths(self):
        self._bottle_root_path = self._object_prim_path_map.get(self._target_object_id, f"/World/object{self._target_object_id}")
        self._lid_link_path = f"{self._bottle_root_path}/links/lid_link"
        self._body_link_path = f"{self._bottle_root_path}/links/body_link"
        self._lid_joint_path = f"{self._bottle_root_path}/joints/lid_twist_joint"

    def _remove_physics_callback(self):
        if self._world is None or not self._physics_cb_registered:
            return
        try:
            self._world.remove_physics_callback(self._physics_cb_name)
        except Exception:
            pass
        self._physics_cb_registered = False

    def _register_physics_callback(self):
        if self._physics_cb_registered:
            self._remove_physics_callback()
        self._world.add_physics_callback(self._physics_cb_name, self.physics_step)
        self._physics_cb_registered = True

    def _report_eval(self):
        lid_pos = self._twist_action._get_prim_world_position(self._lid_link_path)
        body_pos = self._twist_action._get_prim_world_position(self._body_link_path)
        eval_res = evaluate_twist_and_separation(
            twist_deg=self._latest_twist_deg,
            lid_pos=lid_pos,
            body_pos=body_pos,
            twist_success_deg=self._twist_success_deg,
            separation_success_m=self._separation_success_m,
        )
        print(
            "[EVAL] "
            f"success={eval_res['success']}, "
            f"twist_ok={eval_res['twist_ok']}, "
            f"separation_ok={eval_res['separation_ok']}, "
            f"twist_deg={eval_res['twist_deg']:.3f}, "
            f"twist_thr={eval_res['twist_threshold_deg']:.3f}, "
            f"sep_m={eval_res['separation_m']:.4f}, "
            f"sep_thr={eval_res['separation_threshold_m']:.4f}, "
            f"lid_pos={[float(lid_pos[0]), float(lid_pos[1]), float(lid_pos[2])]}, "
            f"body_pos={[float(body_pos[0]), float(body_pos[1]), float(body_pos[2])]}"
        )

    def physics_step(self, step_size):
        if self._state == "WAIT":
            self._settle_counter += 1
            if self._settle_counter < self._settle_steps:
                return
            self._state = "GRASP_PLAN"

        elif self._state == "GRASP_PLAN":
            ok = self._twist_action.plan_lid_grasp(
                bottle_root_prim_path=self._bottle_root_path,
                grasp_id=self._target_grasp_id,
                pregrasp_offset=0.10,
            )
            if not ok:
                self._failure_reason = f"grasp planning failed: {self._twist_action.last_error}"
                self._state = "FAILED"
            else:
                print(
                    "[STATE] GRASP_PLAN ok: "
                    f"object_id={self._target_object_id}, grasp={self._target_grasp_id} (top-down)"
                )
                self._state = "GRASP_EXEC"

        elif self._state == "GRASP_EXEC":
            if self._twist_action.step():
                if self._twist_action.last_error is not None:
                    self._failure_reason = f"grasp execution failed: {self._twist_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "TWIST_PLAN"

        elif self._state == "TWIST_PLAN":
            waypoints, plan_info = self._twist_action.build_twist_waypoints(
                lid_joint_prim_path=self._lid_joint_path,
                total_twist_deg=self._target_twist_deg,
                segments=self._twist_segments,
            )
            if waypoints is None or len(waypoints) == 0:
                self._failure_reason = f"twist waypoint planning failed: {self._twist_action.last_error}"
                self._state = "FAILED"
            else:
                self._planned_waypoints = waypoints
                self._plan_info = plan_info
                self._twist_wp_idx = 0
                if not self._twist_action.reset_twist_tracking(self._lid_link_path, self._lid_joint_path):
                    self._failure_reason = f"twist tracking init failed: {self._twist_action.last_error}"
                    self._state = "FAILED"
                    return
                print(
                    "[PLAN] "
                    f"target_twist_deg={plan_info['total_twist_deg']:.2f}, "
                    f"segments={plan_info['segments']}, "
                    f"radial_norm={plan_info['radial_norm']:.4f}, "
                    f"axis_world={[float(plan_info['axis_world'][0]), float(plan_info['axis_world'][1]), float(plan_info['axis_world'][2])]}"
                )
                self._state = "TWIST_STEP_PLAN"

        elif self._state == "TWIST_STEP_PLAN":
            if self._twist_wp_idx >= len(self._planned_waypoints):
                self._state = "LIFT_PLAN"
                return

            target_pose = self._planned_waypoints[self._twist_wp_idx]
            if self._twist_action.move_twist_step(target_pose):
                self._state = "TWIST_STEP_EXEC"
            else:
                self._failure_reason = (
                    f"twist step planning failed at idx={self._twist_wp_idx}: {self._twist_action.last_error}"
                )
                self._state = "FAILED"

        elif self._state == "TWIST_STEP_EXEC":
            if self._twist_action.step():
                if self._twist_action.last_error is not None:
                    self._failure_reason = (
                        f"twist step execution failed at idx={self._twist_wp_idx}: {self._twist_action.last_error}"
                    )
                    self._state = "FAILED"
                else:
                    twist_deg = self._twist_action.get_accumulated_twist_deg(self._lid_link_path, self._lid_joint_path)
                    if twist_deg is not None:
                        self._latest_twist_deg = float(twist_deg)
                    print(
                        "[TWIST] "
                        f"step={self._twist_wp_idx + 1}/{len(self._planned_waypoints)}, "
                        f"accum_deg={self._latest_twist_deg:.2f}"
                    )
                    self._twist_wp_idx += 1
                    self._state = "TWIST_STEP_PLAN"

        elif self._state == "LIFT_PLAN":
            if self._twist_action.plan_post_twist_lift(lift_delta_z=self._post_lift_delta_z):
                self._state = "LIFT_EXEC"
            else:
                self._failure_reason = f"post twist lift planning failed: {self._twist_action.last_error}"
                self._state = "FAILED"

        elif self._state == "LIFT_EXEC":
            if self._twist_action.step():
                if self._twist_action.last_error is not None:
                    self._failure_reason = f"post twist lift execution failed: {self._twist_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "RELEASE"

        elif self._state == "RELEASE":
            self._twist_action.release()
            self._state = "EVAL"

        elif self._state == "EVAL":
            self._report_eval()
            self._state = "DONE"
            self._remove_physics_callback()

        elif self._state == "DONE":
            self._remove_physics_callback()

        elif self._state == "FAILED":
            if not self._failure_reported:
                print(f"[ERROR] Twist bottle cap task failed deterministically: {self._failure_reason}")
                self._failure_reported = True
                self._remove_physics_callback()
            self._world.pause()
