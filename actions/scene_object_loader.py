import json
import os

import carb
import numpy as np
import omni.usd
from omni.isaac.franka import Franka
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def load_scene_objects_config(scene_config_path):
    if not os.path.exists(scene_config_path):
        raise FileNotFoundError(f"场景配置文件不存在: {scene_config_path}")

    with open(scene_config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    object_list = config_data
    if isinstance(config_data, dict):
        object_list = config_data.get("OBJECTS", [])

    required_fields = ["path", "ADJUSTED_POS", "ADJUSTED_WXYZ"]
    if not isinstance(object_list, list) or len(object_list) == 0:
        raise ValueError("scene_objects.json 中 OBJECTS 必须是非空数组。")

    for idx, item in enumerate(object_list):
        for field in required_fields:
            if field not in item:
                raise ValueError(f"scene_objects.json 第 {idx} 个物体缺少字段: {field}")
        if len(item["ADJUSTED_POS"]) != 3 or len(item["ADJUSTED_WXYZ"]) != 4:
            raise ValueError(f"scene_objects.json 第 {idx} 个物体的 ADJUSTED_POS 必须长度为 3，ADJUSTED_WXYZ 必须长度为 4。")
        if not os.path.exists(item["path"]):
            raise FileNotFoundError(f"scene_objects.json 第 {idx} 个物体路径不存在: {item['path']}")
        if "STATIC_COLLIDER" in item and not isinstance(item["STATIC_COLLIDER"], bool):
            raise ValueError(f"scene_objects.json 第 {idx} 个物体的 STATIC_COLLIDER 必须是 bool。")

    return object_list


def load_scene_full_config(scene_config_path):
    if not os.path.exists(scene_config_path):
        raise FileNotFoundError(f"场景配置文件不存在: {scene_config_path}")
    with open(scene_config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    if isinstance(config_data, dict):
        return config_data
    return {"OBJECTS": config_data}


def parse_franka_config(config_data):
    default_cfg = {
        "prim_path": "/World/Franka",
        "name": "franka",
        "POSITION": [0.0, 0.0, 0.0],
        "WXYZ": [1.0, 0.0, 0.0, 0.0],
        "SCALE": 1.0,
    }
    franka_cfg = dict(default_cfg)
    if isinstance(config_data, dict):
        user_cfg = config_data.get("FRANKA", {})
        if user_cfg:
            if "POSITION" in user_cfg and len(user_cfg["POSITION"]) != 3:
                raise ValueError("FRANKA.POSITION must be length 3.")
            if "WXYZ" in user_cfg and len(user_cfg["WXYZ"]) != 4:
                raise ValueError("FRANKA.WXYZ must be length 4.")
            if "SCALE" in user_cfg and float(user_cfg["SCALE"]) <= 0:
                raise ValueError("FRANKA.SCALE must be > 0.")
            franka_cfg.update(user_cfg)
    return franka_cfg


def add_franka_from_config(world, franka_cfg):
    franka = world.scene.add(
        Franka(
            prim_path=franka_cfg["prim_path"],
            name=franka_cfg["name"],
            position=np.array(franka_cfg["POSITION"], dtype=np.float32),
            orientation=np.array(franka_cfg["WXYZ"], dtype=np.float32),
        )
    )
    stage = world.stage
    prim = stage.GetPrimAtPath(franka_cfg["prim_path"])
    if prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        scale_attr = prim.GetAttribute("xformOp:scale")
        scale = float(franka_cfg["SCALE"])
        scale_v = Gf.Vec3f(scale, scale, scale)
        if scale_attr.IsValid():
            scale_attr.Set(scale_v)
        else:
            xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(scale_v)
    return franka


def import_scene_objects(object_configs):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前 USD stage 不可用，无法导入场景物体。")

    object_prim_paths = []
    object_prim_path_map = {}
    grasp_object_prim_path = "/World/object1"

    for idx, cfg in enumerate(object_configs):
        orientation = cfg["ADJUSTED_WXYZ"]

        obj_id = idx + 1
        prim_path = f"/World/object{obj_id}"

        obj_xform = UsdGeom.Xform.Define(stage, prim_path)
        obj_prim = obj_xform.GetPrim()
        obj_prim.GetReferences().AddReference(cfg["path"])
        xformable = UsdGeom.Xformable(obj_prim)

        # 关键：只设置平移/朝向，避免默认写入 scale=1 覆盖引用资产内的缩放。
        translate_attr = obj_prim.GetAttribute("xformOp:translate")
        translate_val = Gf.Vec3d(float(cfg["ADJUSTED_POS"][0]), float(cfg["ADJUSTED_POS"][1]), float(cfg["ADJUSTED_POS"][2]))
        if translate_attr.IsValid():
            translate_attr.Set(translate_val)
        else:
            xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(translate_val)

        orient_attr = obj_prim.GetAttribute("xformOp:orient")
        orient_val = Gf.Quatf(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        if orient_attr.IsValid():
            orient_attr.Set(orient_val)
        else:
            xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(orient_val)

        # 可选：如果 scene json 里配置 SCALE，则在“资产原始 scale”基础上再乘一层。
        if "SCALE" in cfg:
            scale_attr = obj_prim.GetAttribute("xformOp:scale")
            base_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            if scale_attr.IsValid():
                raw_scale = scale_attr.Get()
                if raw_scale is not None:
                    base_scale = np.array(raw_scale, dtype=np.float32)
            scene_scale = np.array([cfg["SCALE"], cfg["SCALE"], cfg["SCALE"]], dtype=np.float32)
            final_scale = base_scale * scene_scale
            scale_val = Gf.Vec3f(float(final_scale[0]), float(final_scale[1]), float(final_scale[2]))
            if scale_attr.IsValid():
                scale_attr.Set(scale_val)
            else:
                xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(scale_val)

        if cfg.get("STATIC_COLLIDER", False):
            _force_static_colliders(stage, prim_path)

        object_prim_paths.append(prim_path)
        object_prim_path_map[obj_id] = prim_path
        if idx == 0:
            grasp_object_prim_path = prim_path

    return {
        "object_prim_paths": object_prim_paths,
        "object_prim_path_map": object_prim_path_map,
        "grasp_object_prim_path": grasp_object_prim_path,
    }


def load_and_import_scene_objects(scene_config_path):
    object_configs = load_scene_objects_config(scene_config_path)
    return import_scene_objects(object_configs)


def _try_enable_gpu_dynamics():
    settings = carb.settings.get_settings()
    candidate_keys = [
        "/physics/useGpuDynamics",
        "/persistent/physics/useGpuDynamics",
    ]
    enabled = False
    for key in candidate_keys:
        try:
            settings.set_bool(key, True)
            enabled = True
        except Exception:
            pass
    if enabled:
        print("[INFO] Requested PhysX GPU dynamics for SDF-based assets.")
    else:
        print("[WARN] Could not set GPU dynamics via settings. SDF rigid actor errors may remain.")


def _force_static_colliders(stage, root_prim_path: str):
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateRigidBodyEnabledAttr().Set(False)
            rb.CreateKinematicEnabledAttr().Set(False)

        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr().Set(True)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


def setup_scene_from_config(world, scene_config_path, add_franka=True, enable_gpu_dynamics=True):
    config_data = load_scene_full_config(scene_config_path)
    if enable_gpu_dynamics:
        _try_enable_gpu_dynamics()

    franka_cfg = parse_franka_config(config_data)
    franka = add_franka_from_config(world, franka_cfg) if add_franka else None
    import_result = import_scene_objects(config_data.get("OBJECTS", []))
    return {
        "franka": franka,
        "franka_cfg": franka_cfg,
        **import_result,
    }
