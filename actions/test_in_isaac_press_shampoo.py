import json
import os
import tempfile
import numpy as np
from omni.isaac.examples.base_sample import BaseSample

from .franka_press_shampoo_action import FrankaPressShampooAction
from .scene_object_loader import setup_scene_from_config


def evaluate_press_motion(
    target_distance_m,
    actual_distance_m,
    lateral_drift_m,
    distance_tolerance_m=0.025,
    max_lateral_drift_m=0.030,
):
    distance_error_m = abs(float(actual_distance_m) - float(target_distance_m))
    distance_ok = distance_error_m <= float(distance_tolerance_m)
    lateral_ok = float(lateral_drift_m) <= float(max_lateral_drift_m)
    success = bool(distance_ok and lateral_ok)
    return {
        "success": success,
        "distance_ok": bool(distance_ok),
        "lateral_ok": bool(lateral_ok),
        "target_distance_m": float(target_distance_m),
        "actual_distance_m": float(actual_distance_m),
        "distance_error_m": float(distance_error_m),
        "distance_tolerance_m": float(distance_tolerance_m),
        "lateral_drift_m": float(lateral_drift_m),
        "max_lateral_drift_m": float(max_lateral_drift_m),
    }


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._state = "WAIT"
        self._settle_steps = 120
        self._settle_counter = 0
        self._hold_steps = 45
        self._hold_counter = 0

        self._world = None
        self._press_action = None
        self._scene_config_path = os.path.join(os.path.dirname(__file__), "scene_objects_press_shampoo.json")
        self._object_prim_path_map = {}

        self._target_object_id = 3
        self._target_grasp_id = 0
        self._pregrasp_offset_m = 0.10
        self._press_distance_m = 0.10
        self._press_steps = 80
        self._distance_tolerance_m = 0.025
        self._max_lateral_drift_m = 0.030

        self._shampoo_root_path = None
        self._physics_cb_name = "sim_step"
        self._physics_cb_registered = False
        self._failure_reason = None
        self._failure_reported = False

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        self._scene_config_path = self._resolve_scene_config_path()
        setup_result = setup_scene_from_config(world, self._scene_config_path)
        self._franka = setup_result["franka"]
        self._object_prim_path_map = setup_result["object_prim_path_map"]

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("franka")
        self._press_action = FrankaPressShampooAction(world=self._world, franka=self._franka)
        self._press_action.open_gripper()

        self._resolve_shampoo_paths()
        self._reset_state_vars()
        self._register_physics_callback()
        await self._world.play_async()

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._resolve_shampoo_paths()
        self._reset_state_vars()
        if self._press_action is not None:
            self._press_action.release()
            self._press_action.open_gripper()
        self._register_physics_callback()
        await self._world.play_async()

    def _resolve_scene_config_path(self):
        shampoo_usd = os.environ.get("SHAMPOO_USD_PATH")
        if shampoo_usd is None:
            with open(self._scene_config_path, "r", encoding="utf-8") as f:
                shampoo_usd = json.load(f)["OBJECTS"][2]["path"]

        if not os.path.exists(shampoo_usd):
            raise FileNotFoundError(
                "Shampoo USD has not been generated yet. "
                f"Expected {shampoo_usd}. Set SHAMPOO_USD_PATH to the exported shampoo USD once available."
            )
        if os.environ.get("SHAMPOO_USD_PATH") is None:
            return self._scene_config_path

        with open(self._scene_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["OBJECTS"][2]["path"] = shampoo_usd

        fd, path = tempfile.mkstemp(prefix="scene_objects_press_shampoo_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return path

    def _reset_state_vars(self):
        self._state = "WAIT"
        self._settle_counter = 0
        self._hold_counter = 0
        self._failure_reason = None
        self._failure_reported = False

    def _resolve_shampoo_paths(self):
        self._shampoo_root_path = self._object_prim_path_map.get(
            self._target_object_id,
            f"/World/object{self._target_object_id}",
        )

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
        metrics = self._press_action.get_press_metrics()
        eval_res = evaluate_press_motion(
            target_distance_m=self._press_distance_m,
            actual_distance_m=metrics["actual_distance_m"],
            lateral_drift_m=metrics["lateral_drift_m"],
            distance_tolerance_m=self._distance_tolerance_m,
            max_lateral_drift_m=self._max_lateral_drift_m,
        )
        print(
            "[EVAL] "
            f"success={eval_res['success']}, "
            f"distance_ok={eval_res['distance_ok']}, "
            f"lateral_ok={eval_res['lateral_ok']}, "
            f"target_m={eval_res['target_distance_m']:.4f}, "
            f"actual_m={eval_res['actual_distance_m']:.4f}, "
            f"error_m={eval_res['distance_error_m']:.4f}, "
            f"tol_m={eval_res['distance_tolerance_m']:.4f}, "
            f"lateral_m={eval_res['lateral_drift_m']:.4f}, "
            f"lateral_thr_m={eval_res['max_lateral_drift_m']:.4f}, "
            f"steps={metrics['steps_completed']}"
        )

    def physics_step(self, step_size):
        if self._state == "WAIT":
            self._settle_counter += 1
            if self._settle_counter < self._settle_steps:
                return
            self._state = "PREGRASP_PLAN"

        elif self._state == "PREGRASP_PLAN":
            ok = self._press_action.plan_pregrasp(
                shampoo_root_prim_path=self._shampoo_root_path,
                grasp_id=self._target_grasp_id,
                pregrasp_offset=self._pregrasp_offset_m,
            )
            if not ok:
                self._failure_reason = f"pregrasp planning failed: {self._press_action.last_error}"
                self._state = "FAILED"
            else:
                print(
                    "[STATE] PREGRASP_PLAN ok: "
                    f"object_id={self._target_object_id}, grasp={self._target_grasp_id}"
                )
                self._state = "PREGRASP_EXEC"

        elif self._state == "PREGRASP_EXEC":
            if self._press_action.step():
                if self._press_action.last_error is not None:
                    self._failure_reason = f"pregrasp execution failed: {self._press_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "HOLD"

        elif self._state == "HOLD":
            self._hold_counter += 1
            if self._hold_counter == 1:
                print(f"[STATE] HOLD at pregrasp for {self._hold_steps} steps")
            if self._hold_counter >= self._hold_steps:
                self._state = "PRESS_START"

        elif self._state == "PRESS_START":
            ok = self._press_action.start_press(
                press_distance_m=self._press_distance_m,
                steps=self._press_steps,
                close_gripper=False,
            )
            if not ok:
                self._failure_reason = f"press setup failed: {self._press_action.last_error}"
                self._state = "FAILED"
            else:
                self._state = "PRESS_EXEC"

        elif self._state == "PRESS_EXEC":
            if self._press_action.step():
                if self._press_action.last_error is not None:
                    self._failure_reason = f"press execution failed: {self._press_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "EVAL"

        elif self._state == "EVAL":
            self._report_eval()
            self._state = "DONE"
            self._remove_physics_callback()

        elif self._state == "DONE":
            self._remove_physics_callback()

        elif self._state == "FAILED":
            if not self._failure_reported:
                print(f"[ERROR] Press shampoo task failed deterministically: {self._failure_reason}")
                self._failure_reported = True
                self._remove_physics_callback()
            self._world.pause()
