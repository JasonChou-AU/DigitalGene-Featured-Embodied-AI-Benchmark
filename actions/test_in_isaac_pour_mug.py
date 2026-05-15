import os

import numpy as np
from omni.isaac.examples.base_sample import BaseSample

from .franka_pour_mug_action import FrankaPourMugAction
from .scene_object_loader import setup_scene_from_config


def evaluate_pour_downward(downward_angle_deg, success_threshold_deg=20.0):
    success = float(downward_angle_deg) <= float(success_threshold_deg)
    return {
        "success": bool(success),
        "downward_angle_deg": float(downward_angle_deg),
        "success_threshold_deg": float(success_threshold_deg),
    }


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._state = "WAIT"
        self._settle_steps = 120
        self._settle_counter = 0

        self._world = None
        self._franka = None
        self._franka_cfg = None
        self._pour_action = None

        self._scene_config_path = os.path.join(os.path.dirname(__file__), "scene_objects_pour_mug.json")
        self._object_prim_path_map = {}

        self._target_object_id = 3
        self._target_grasp_id = 0
        self._target_position = np.array([3.5, 2.70, 1.3], dtype=np.float32)
        self._pour_angle_deg = 90.0
        self._pour_steps = 160
        self._pre_pour_wait_sec = 1.0
        self._pre_pour_wait_timer = 0.0
        self._downward_success_threshold_deg = 20.0

        self._physics_cb_name = "sim_step"
        self._physics_cb_registered = False
        self._failure_reason = None
        self._failure_reported = False

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
        if self._franka is None:
            setup_result = setup_scene_from_config(self._world, self._scene_config_path)
            self._franka = setup_result["franka"]
            self._franka_cfg = setup_result["franka_cfg"]
            self._object_prim_path_map = setup_result["object_prim_path_map"]
            await self._world.reset_async()
            self._franka = self._world.scene.get_object("franka") or self._franka

        self._pour_action = FrankaPourMugAction(world=self._world, franka=self._franka)
        self._pour_action.open_gripper()

        self._reset_state_vars()
        self._register_physics_callback()
        await self._world.play_async()

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._reset_state_vars()
        if self._pour_action is not None:
            self._pour_action.release()
            self._pour_action.open_gripper()
        self._register_physics_callback()
        await self._world.play_async()

    def _reset_state_vars(self):
        self._state = "WAIT"
        self._settle_counter = 0
        self._pre_pour_wait_timer = 0.0
        self._failure_reason = None
        self._failure_reported = False

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

    def _get_target_pose(self, target_position):
        return {
            "position": target_position.astype(np.float32),
            "orientation": self._pour_action.saved_grasp_orientation.astype(np.float32),
        }

    def _report_eval(self):
        downward_angle_deg = self._pour_action.get_gripper_downward_angle_deg()
        eval_res = evaluate_pour_downward(
            downward_angle_deg=downward_angle_deg,
            success_threshold_deg=self._downward_success_threshold_deg,
        )
        print(
            "[EVAL] "
            f"success={eval_res['success']}, "
            f"downward_angle_deg={eval_res['downward_angle_deg']:.3f}, "
            f"threshold_deg={eval_res['success_threshold_deg']:.3f}, "
            f"target_position={self._target_position.tolist()}, "
            f"pour_angle_deg={self._pour_angle_deg:.3f}"
        )

    def physics_step(self, step_size):
        if self._state == "WAIT":
            self._settle_counter += 1
            if self._settle_counter < self._settle_steps:
                return
            self._state = "GRASP_PLAN"

        elif self._state == "GRASP_PLAN":
            ok = self._pour_action.grasp(
                object_id_or_path=self._target_object_id,
                grasp_id=self._target_grasp_id,
                object_prim_path_map=self._object_prim_path_map,
            )
            if not ok:
                self._failure_reason = f"grasp planning failed: {self._pour_action.last_error}"
                self._state = "FAILED"
            else:
                print(
                    "[STATE] GRASP_PLAN ok: "
                    f"object_id={self._target_object_id}, grasp={self._target_grasp_id}"
                )
                self._state = "GRASP_EXEC"

        elif self._state == "GRASP_EXEC":
            if self._pour_action.step():
                if self._pour_action.last_error is not None:
                    self._failure_reason = f"grasp execution failed: {self._pour_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "MOVE_PLAN"

        elif self._state == "MOVE_PLAN":
            target_pose = self._get_target_pose(self._target_position)
            if self._pour_action.move(target_pose=target_pose, use_curobo=True):
                print(
                    "[STATE] MOVE_PLAN ok: "
                    f"target_position={self._target_position.tolist()}, "
                    f"saved_grasp_orientation={target_pose['orientation'].tolist()}"
                )
                self._state = "MOVE_EXEC"
            else:
                self._failure_reason = f"move with CuRobo failed: {self._pour_action.last_error}"
                self._state = "FAILED"

        elif self._state == "MOVE_EXEC":
            if self._pour_action.step():
                if self._pour_action.last_error is not None:
                    self._failure_reason = f"move execution failed: {self._pour_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._pre_pour_wait_timer = 0.0
                    print(f"[STATE] MOVE_EXEC done: waiting {self._pre_pour_wait_sec:.3f}s before pour.")
                    self._state = "POUR_WAIT"

        elif self._state == "POUR_WAIT":
            self._pre_pour_wait_timer += float(step_size)
            if self._pre_pour_wait_timer >= self._pre_pour_wait_sec:
                self._state = "POUR_PLAN"

        elif self._state == "POUR_PLAN":
            if self._pour_action.pour(angle_deg=self._pour_angle_deg, steps=self._pour_steps):
                print(
                    "[STATE] POUR_PLAN ok: "
                    f"angle_deg={self._pour_angle_deg:.3f}, steps={self._pour_steps}"
                )
                self._state = "POUR_EXEC"
            else:
                self._failure_reason = f"pour wrist rotation failed: {self._pour_action.last_error}"
                self._state = "FAILED"

        elif self._state == "POUR_EXEC":
            if self._pour_action.step():
                if self._pour_action.last_error is not None:
                    self._failure_reason = f"pour execution failed: {self._pour_action.last_error}"
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
                print(f"[ERROR] Pour mug task failed deterministically: {self._failure_reason}")
                self._failure_reported = True
                self._remove_physics_callback()
            self._world.pause()

