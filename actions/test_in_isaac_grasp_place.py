from omni.isaac.examples.base_sample import BaseSample
import numpy as np
import os
from pxr import UsdGeom, Gf


from actions.franka_grasp_and_place_action import FrankaBaseAction



from actions.scene_object_loader import setup_scene_from_config



def is_succeess(bottle_position, target_position, threshold=0.1):
    bottle_xy = np.asarray(bottle_position, dtype=np.float32)[:2]
    target_xy = np.asarray(target_position, dtype=np.float32)[:2]
    dist_l2 = float(np.linalg.norm(bottle_xy - target_xy))
    return dist_l2 < float(threshold)


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._state = "WAIT"
        self._settle_steps = 120
        self._settle_counter = 0

        self._world = None
        self._base_action = None

        self._drop_pos = np.array([3.3, 2.6, 1.2], dtype=np.float32)
        self._target_place_position = np.array([3.3, 2.6, 0.73], dtype=np.float32)

        self._target_marker_path = "/World/visual/target_projection_marker"
        self._scene_config_path = os.path.join(os.path.dirname(__file__), "scene_objects.json")
        self._franka_cfg = None
        self._object_prim_path_map = {}

        self._grasp_tasks = [{"object_id": 3, "grasp_pose_id": 5}]
        self._task_idx = 0

        self._physics_cb_name = "sim_step"
        self._physics_cb_registered = False
        self._failure_reason = None
        self._failure_reported = False
        self._gripper_length = 0.09

    def _get_object_world_position(self, object_id):
        prim_path = self._object_prim_path_map.get(int(object_id), f"/World/object{int(object_id)}")
        prim = self._world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"object prim not found: {prim_path}")

        from pxr import Usd, UsdGeom

        xf = UsdGeom.Xformable(prim)
        mat = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        setup_result = setup_scene_from_config(world, self._scene_config_path)
        self._franka = setup_result["franka"]
        self._franka_cfg = setup_result["franka_cfg"]
        self._object_prim_path_map = setup_result["object_prim_path_map"]
        self._create_target_projection_marker(size=0.05)

    def _create_target_projection_marker(self, size=0.05):
        stage = self.get_world().stage
        UsdGeom.Xform.Define(stage, "/World/visual")
        marker = UsdGeom.Cube.Define(stage, self._target_marker_path)
        marker.CreateSizeAttr(float(size))
        marker.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

        proj_pos = self._target_place_position
        xf = UsdGeom.Xformable(marker.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(proj_pos[0]), float(proj_pos[1]), float(proj_pos[2])))
        xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("franka")

        self._base_action = FrankaBaseAction(world=self._world, franka=self._franka)
        self._base_action.open_gripper()

        self._state = "WAIT"
        self._settle_counter = 0
        self._task_idx = 0
        self._failure_reason = None
        self._failure_reported = False
        self._register_physics_callback()
        await self._world.play_async()

    async def setup_pre_reset(self):
        self._remove_physics_callback()

    async def setup_post_reset(self):
        self._state = "WAIT"
        self._settle_counter = 0
        self._task_idx = 0
        self._failure_reason = None
        self._failure_reported = False
        if self._base_action is not None:
            self._base_action.release()
            self._base_action.open_gripper()
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

    def physics_step(self, step_size):
        if self._state == "WAIT":
            self._settle_counter += 1
            if self._settle_counter < self._settle_steps:
                return
            self._state = "GRASP_PLAN"

        elif self._state == "GRASP_PLAN":
            if self._task_idx >= len(self._grasp_tasks):
                self._state = "DONE"
                return

            task = self._grasp_tasks[self._task_idx]
            object_id = int(task["object_id"])
            grasp_pose_id = int(task["grasp_pose_id"])
            if self._base_action.grasp(
                object_id_or_path=object_id,
                grasp_id=grasp_pose_id,
                object_prim_path_map=self._object_prim_path_map,
            ):
                self._state = "GRASP_EXEC"
            else:
                self._failure_reason = (
                    f"grasp plan failed: object_id={object_id}, grasp_pose_id={grasp_pose_id}, "
                    f"detail={self._base_action.last_error}"
                )
                self._state = "FAILED"

        elif self._state == "GRASP_EXEC":
            if self._base_action.step():
                if self._base_action.last_error is not None:
                    self._failure_reason = f"physical grasp failed: {self._base_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "MOVE_PLAN"

        elif self._state == "MOVE_PLAN":
            target_pose = {
                "position": self._drop_pos - self._base_action.saved_approach_dir * self._gripper_length,
                "orientation": self._base_action.saved_grasp_orientation,
            }
            if self._base_action.move(target_pose=target_pose):
                self._state = "MOVE_EXEC"
            else:
                self._failure_reason = (
                    "move plan failed: drop pose unreachable or in collision, "
                    f"detail={self._base_action.last_error}"
                )
                self._state = "FAILED"

        elif self._state == "MOVE_EXEC":
            if self._base_action.step():
                if self._base_action.last_error is not None:
                    self._failure_reason = f"move execution failed: {self._base_action.last_error}"
                    self._state = "FAILED"
                else:
                    self._state = "RELEASE"

        elif self._state == "RELEASE":
            self._base_action.release()
            self._task_idx += 1
            if self._task_idx < len(self._grasp_tasks):
                self._settle_counter = 0
                self._state = "WAIT"
            else:
                try:
                    bottle_pos = self._get_object_world_position(object_id=1)
                    success = is_succeess(bottle_pos, self._target_place_position, threshold=0.1)
                    bottle_xy = np.asarray(bottle_pos, dtype=np.float32)[:2]
                    target_xy = np.asarray(self._target_place_position, dtype=np.float32)[:2]
                    dist_l2 = float(np.linalg.norm(bottle_xy - target_xy))
                    print(
                        "[EVAL] "
                        f"is_suceess={success}, "
                        f"bottle_xy={[float(bottle_xy[0]), float(bottle_xy[1])]}, "
                        f"target_xy={[float(target_xy[0]), float(target_xy[1])]}, "
                        f"l2_xy={dist_l2:.4f}"
                    )
                except Exception as exc:
                    print(f"[WARN] EVAL skipped: {exc}")
                self._state = "DONE"
                self._remove_physics_callback()

        elif self._state == "FAILED":
            if not self._failure_reported:
                print(f"[ERROR] Task failed deterministically: {self._failure_reason}")
                self._failure_reported = True
                self._remove_physics_callback()
            self._world.pause()

# {
#     "FRANKA": {
#     "prim_path": "/World/Franka",
#     "name": "franka",
#     "POSITION": [3.5, 2.0, 0.74],
#     "WXYZ": [0.70710678, 0.0, 0.0, 0.70710678],
#     "SCALE": 1.0
#     },
#   "OBJECTS": [
#     {
#       "path": "/home/zjx/project/ArtVIP/ArtVIP/Scenes/dining_room/diningroom/model_diningroom.usd",
#       "ADJUSTED_POS": [0, 0, 0],
#       "ADJUSTED_WXYZ": [1, 0, 0, 0],
#       "STATIC_COLLIDER": true
#     },
#     {
#       "path": "/home/zjx/project/ArtVIP/ArtVIP/Articulated_objects/small_furniture/table/table_7/model_table_7.usd",
#       "ADJUSTED_POS": [3.7, 2.5, 0.0],
#       "ADJUSTED_WXYZ": [0.70710678, 0.0, 0.0, -0.70710678],
#       "STATIC_COLLIDER": true
#     },
#          {
#       "path": "/home/zjx/project/benchmark/real_object_data/Bottle/usda_output/lemon_tea_simple_collision.usda",
#        "ADJUSTED_POS": [3.7, 2.6, 1.1],
#        "ADJUSTED_WXYZ": [0.70710678, 0.70710678, 0.0, 0.0],
#        "SCALE": 0.2
#      }


#   ]
# }