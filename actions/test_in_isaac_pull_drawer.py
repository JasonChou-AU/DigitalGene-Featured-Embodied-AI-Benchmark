import os

import numpy as np
from omni.isaac.examples.base_sample import BaseSample

from actions.franka_pull_drawer_action import FrankaPullDrawerAction
from actions.scene_object_loader import setup_scene_from_config


def compute_alpha_eval(pulled_distance, max_pull_distance, success_alpha_threshold=0.35):
    max_pull = max(float(max_pull_distance), 1e-6)
    alpha = float(np.clip(float(pulled_distance) / max_pull, 0.0, 1.0))
    success = alpha >= float(success_alpha_threshold)
    return {
        "alpha": alpha,
        "success": success,
        "threshold": float(success_alpha_threshold),
    }


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._state = "WAIT"
        self._settle_steps = 120
        self._settle_counter = 0

        self._world = None
        self._pull_action = None

        self._scene_config_path = os.path.join(os.path.dirname(__file__), "scene_objects_pull_drawer.json")
        self._franka_cfg = None
        self._object_prim_path_map = {}

        # User-locked task spec
        self._target_object_id = 3
        self._target_drawer_index = 0
        self._target_grasp_id = 2
        self._target_alpha = 0.5
        self._success_alpha_threshold = 0.35
        self._pull_segments = 8
        self._pull_max_step_distance = 0.012

        self._drawer_link_path = None
        self._drawer_joint_path = None
        self._planned_waypoints = []
        self._plan_info = None
        self._pull_wp_idx = 0

        self._physics_cb_name = "sim_step"
        self._physics_cb_registered = False
        self._failure_reason = None
        self._failure_reported = False

    def _resolve_drawer_paths(self):
        object_prim_path = self._object_prim_path_map.get(self._target_object_id, f"/World/object{self._target_object_id}")
        self._drawer_link_path = f"{object_prim_path}/links/drawer_{self._target_drawer_index}"
        self._drawer_joint_path = f"{object_prim_path}/joints/drawer_joint_{self._target_drawer_index}"

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        setup_result = setup_scene_from_config(world, self._scene_config_path)
        self._franka = setup_result["franka"]
        self._franka_cfg = setup_result["franka_cfg"]
        self._object_prim_path_map = setup_result["object_prim_path_map"]

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("franka")

        self._pull_action = FrankaPullDrawerAction(world=self._world, franka=self._franka)
        self._pull_action.open_gripper()
        self._resolve_drawer_paths()

        self._state = "WAIT"
        self._settle_counter = 0
        self._planned_waypoints = []
        self._plan_info = None
        self._pull_wp_idx = 0
        self._failure_reason = None
        self._failure_reported = False

        self._register_physics_callback()
        await self._world.play_async()

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._resolve_drawer_paths()
        self._state = "WAIT"
        self._settle_counter = 0
        self._planned_waypoints = []
        self._plan_info = None
        self._pull_wp_idx = 0
        self._failure_reason = None
        self._failure_reported = False
        if self._pull_action is not None:
            self._pull_action.release()
            self._pull_action.open_gripper()
        self._register_physics_callback()
        await self._world.play_async()

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
        stats = self._pull_action.get_drawer_pull_stats(self._drawer_joint_path)
        if stats is None:
            print(f"[WARN] EVAL skipped: {self._pull_action.last_error}")
            return

        eval_res = compute_alpha_eval(
            pulled_distance=stats["pulled_distance"],
            max_pull_distance=stats["max_pull_distance"],
            success_alpha_threshold=self._success_alpha_threshold,
        )

        print(
            "[EVAL] "
            f"alpha={eval_res['alpha']:.4f}, "
            f"success={eval_res['success']}, "
            f"threshold={eval_res['threshold']:.4f}, "
            f"pulled_distance={stats['pulled_distance']:.4f}, "
            f"max_pull_distance={stats['max_pull_distance']:.4f}, "
            f"raw_displacement={stats['raw_displacement']:.4f}, "
            f"joint_limits=[{stats['lower_limit']:.4f}, {stats['upper_limit']:.4f}]"
        )

    def physics_step(self, step_size):
        if self._state == "WAIT":
            self._settle_counter += 1
            if self._settle_counter < self._settle_steps:
                return
            self._state = "GRASP_PLAN"

        elif self._state == "GRASP_PLAN":
            ok = self._pull_action.plan_drawer_grasp(
                drawer_link_prim_path=self._drawer_link_path,
                grasp_id=self._target_grasp_id,
            )
            if not ok:
                self._failure_reason = f"grasp planning failed: {self._pull_action.last_error}"
                self._state = "FAILED"
            else:
                print(
                    "[STATE] GRASP_PLAN ok: "
                    f"object_id={self._target_object_id}, drawer={self._target_drawer_index}, grasp={self._target_grasp_id}"
                )
                self._state = "GRASP_EXEC"

        elif self._state == "GRASP_EXEC":
            if self._pull_action.step():
                if self._pull_action.last_error is not None:
                    self._failure_reason = f"grasp execution failed: {self._pull_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "PULL_PLAN"

        elif self._state == "PULL_PLAN":
            waypoints, plan_info = self._pull_action.build_pull_waypoints(
                drawer_joint_prim_path=self._drawer_joint_path,
                desired_alpha=self._target_alpha,
                segments=self._pull_segments,
                safety_margin=0.01,
                max_step_distance=self._pull_max_step_distance,
            )
            if waypoints is None or len(waypoints) == 0:
                self._failure_reason = f"pull waypoint planning failed: {self._pull_action.last_error}"
                self._state = "FAILED"
            else:
                self._planned_waypoints = waypoints
                self._plan_info = plan_info
                self._pull_wp_idx = 0
                print(
                    "[PLAN] "
                    f"current_alpha={plan_info['current_alpha']:.4f}, "
                    f"target_alpha={plan_info['target_alpha']:.4f}, "
                    f"delta_pull={plan_info['delta_pull']:.4f}, "
                    f"segments={len(self._planned_waypoints)}, "
                    f"max_step={plan_info['max_step_distance']:.4f}"
                )
                self._state = "PULL_STEP_PLAN"

        elif self._state == "PULL_STEP_PLAN":
            if self._pull_wp_idx >= len(self._planned_waypoints):
                self._state = "RELEASE"
                return

            target_pose = self._planned_waypoints[self._pull_wp_idx]
            if self._pull_action.move_pull_step(target_pose):
                self._state = "PULL_STEP_EXEC"
            else:
                self._failure_reason = (
                    f"pull step planning failed at idx={self._pull_wp_idx}: {self._pull_action.last_error}"
                )
                self._state = "FAILED"

        elif self._state == "PULL_STEP_EXEC":
            if self._pull_action.step():
                if self._pull_action.last_error is not None:
                    self._failure_reason = (
                        f"pull step execution failed at idx={self._pull_wp_idx}: {self._pull_action.last_error}"
                    )
                    self._state = "FAILED"
                else:
                    self._pull_wp_idx += 1
                    self._state = "PULL_STEP_PLAN"

        elif self._state == "RELEASE":
            self._pull_action.release()
            self._state = "EVAL"

        elif self._state == "EVAL":
            self._report_eval()
            self._state = "DONE"
            self._remove_physics_callback()

        elif self._state == "DONE":
            self._remove_physics_callback()

        elif self._state == "FAILED":
            if not self._failure_reported:
                print(f"[ERROR] Pull drawer task failed deterministically: {self._failure_reason}")
                self._failure_reported = True
                self._remove_physics_callback()
            self._world.pause()
