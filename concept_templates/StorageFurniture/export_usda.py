import os
import pickle
import inspect
import copy
import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from scipy.spatial.transform import Rotation as Rot

from concept_template import *
from knowledge_definitions import get_grasp_spec


def to_vec3f(v):
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def to_vec3d(v):
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def to_quatf_xyzw(q_xyzw):
    return Gf.Quatf(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def set_initial_pose(prim_xform, position, euler_deg):
    prim_xform.ClearXformOpOrder()
    prim_xform.AddTranslateOp().Set(to_vec3d(position))
    r = Rot.from_euler("xyz", euler_deg, degrees=True)
    prim_xform.AddOrientOp().Set(to_quatf_xyzw(r.as_quat()))


def _create_mesh(stage, mesh_path, verts, faces, collision=False, approximation="convexHull"):
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([to_vec3f(v) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

    if collision:
        prim = mesh.GetPrim()
        UsdGeom.Imageable(mesh).CreatePurposeAttr().Set("physics")
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approximation)

    return mesh


def _create_link(
    stage,
    link_path,
    verts,
    faces,
    mass,
    kinematic=False,
    collision_approx="convexHull",
    translate=(0.0, 0.0, 0.0),
    orient_xyzw=(0.0, 0.0, 0.0, 1.0),
    linear_damping=None,
    angular_damping=None,
):
    link = UsdGeom.Xform.Define(stage, link_path)
    link.ClearXformOpOrder()
    link.AddTranslateOp().Set(to_vec3d(translate))
    link.AddOrientOp().Set(to_quatf_xyzw(orient_xyzw))
    link.AddScaleOp().Set(to_vec3f((1.0, 1.0, 1.0)))
    prim = link.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass))
    if linear_damping is not None:
        prim.CreateAttribute("physics:linearDamping", Sdf.ValueTypeNames.Float).Set(float(linear_damping))
    if angular_damping is not None:
        prim.CreateAttribute("physics:angularDamping", Sdf.ValueTypeNames.Float).Set(float(angular_damping))
    if kinematic:
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    _create_mesh(stage, f"{link_path}/visual", verts, faces, collision=False)
    _create_mesh(stage, f"{link_path}/collision", verts, faces, collision=True, approximation=collision_approx)
    return link


def _split_mesh_by_face_blocks(overall_mesh, faces_per_part, num_parts):
    vertices = np.asarray(overall_mesh.vertices, dtype=np.float32)
    faces = np.asarray(overall_mesh.faces, dtype=np.int32)
    parts = []
    cursor = 0
    for _ in range(num_parts):
        start = cursor
        end = min(cursor + faces_per_part, len(faces))
        part_faces_global = faces[start:end]
        if len(part_faces_global) == 0:
            parts.append((np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)))
            cursor = end
            continue
        unique_vids = np.unique(part_faces_global.reshape(-1))
        part_verts = vertices[unique_vids]
        mapping = -np.ones((len(vertices),), dtype=np.int32)
        mapping[unique_vids] = np.arange(len(unique_vids), dtype=np.int32)
        part_faces = mapping[part_faces_global]
        parts.append((part_verts, part_faces))
        cursor = end
    return parts


def _door_joint_origin_world(part, door_idx):
    door_rot = float(part.door_rotation[door_idx])
    door_offset = np.array(part.door_offset[door_idx], dtype=float)
    door_size = np.array(part.door_size[door_idx], dtype=float)
    handle_offset = np.array(part.handle_offset[door_idx], dtype=float)
    handle_size = np.array(part.handle_size[door_idx], dtype=float)

    is_right_hinge = (max(handle_size) > 1e-3 and handle_offset[0] > 0) or (door_offset[0] < 0)
    if door_offset[2] < 0:
        is_right_hinge = not is_right_hinge

    door_depth_sign = np.sign(float(door_offset[2]))
    if abs(door_depth_sign) < 1e-8:
        door_depth_sign = 1.0
    hinge_z_offset = -door_depth_sign * float(door_size[2]) / 2.0

    hinge_local_in_door = np.array(
        [
            -door_size[0] / 2.0 if is_right_hinge else door_size[0] / 2.0,
            0.0,
            hinge_z_offset,
        ],
        dtype=float,
    )

    r_door = Rot.from_euler("xyz", [0.0, door_rot, 0.0], degrees=False).as_matrix()
    hinge_in_part = door_offset + r_door @ hinge_local_in_door

    r_part = Rot.from_euler("xyz", part.rotation, degrees=False).as_matrix()
    p_part = np.array(part.position, dtype=float)
    return p_part + r_part @ hinge_in_part


def _door_joint_limits_deg(part, door_idx):
    door_rot = abs(float(part.door_rotation[door_idx]))
    handle_size = np.array(part.handle_size[door_idx], dtype=float)
    handle_offset = np.array(part.handle_offset[door_idx], dtype=float)
    door_offset = np.array(part.door_offset[door_idx], dtype=float)

    is_right_hinge = (max(handle_size) > 1e-3 and handle_offset[0] > 0) or (door_offset[0] < 0)
    if door_offset[2] < 0:
        is_right_hinge = not is_right_hinge

    if is_right_hinge:
        lower = -np.pi / 2 + door_rot
        upper = door_rot
    else:
        lower = -door_rot
        upper = np.pi / 2 - door_rot
    return np.degrees(lower), np.degrees(upper)


def _drawer_joint_limits(part, drawer_idx):
    travel_proportion = 0.85
    drawer_len = float(part.drawer_size[drawer_idx][2])
    travel_total = drawer_len * travel_proportion
    # 只允许向外拉，不允许向回退（关节位移不能小于初始位）。
    lower = 0.0
    upper = max(0.0, travel_total)
    return lower, upper


def _carve_base_openings(base_verts, base_faces, door_parts, plane_tol=0.035, margin=0.01):
    if len(door_parts) == 0 or len(base_faces) == 0:
        return base_faces

    keep_mask = np.ones((len(base_faces),), dtype=bool)
    face_centers = base_verts[base_faces].mean(axis=1)

    for door_obj in door_parts:
        r_part = Rot.from_euler("xyz", door_obj.rotation, degrees=False).as_matrix()
        p_part = np.array(door_obj.position, dtype=float)
        x_axis = r_part @ np.array([1.0, 0.0, 0.0], dtype=float)
        y_axis = r_part @ np.array([0.0, 1.0, 0.0], dtype=float)
        z_axis = r_part @ np.array([0.0, 0.0, 1.0], dtype=float)

        for door_idx in range(door_obj.number_of_door[0]):
            door_offset = np.array(door_obj.door_offset[door_idx], dtype=float)
            door_size = np.array(door_obj.door_size[door_idx], dtype=float)

            center_world = p_part + r_part @ door_offset
            depth_sign = np.sign(float(door_offset[2]))
            if abs(depth_sign) < 1e-8:
                depth_sign = 1.0
            # 开口平面位于门板靠柜体一侧
            plane_point = center_world - depth_sign * (float(door_size[2]) / 2.0) * z_axis
            normal = depth_sign * z_axis

            rel = face_centers - plane_point[None, :]
            dist = np.abs(rel @ normal)
            u = rel @ x_axis
            v = rel @ y_axis

            in_plane = dist < plane_tol
            in_rect = (np.abs(u) <= door_size[0] / 2.0 + margin) & (np.abs(v) <= door_size[1] / 2.0 + margin)
            keep_mask &= ~(in_plane & in_rect)

    return base_faces[keep_mask]


def _add_grasps(
    stage,
    link_path,
    obj,
    params_list,
    parent_translate=(0.0, 0.0, 0.0),
    parent_orient_xyzw=(0.0, 0.0, 0.0, 1.0),
    scale=1.0,
):
    grasp_root = UsdGeom.Xform.Define(stage, f"{link_path}/grasps")
    del grasp_root
    valid_idx = 0
    parent_t = np.array(parent_translate, dtype=float)
    parent_r = Rot.from_quat(np.array(parent_orient_xyzw, dtype=float)).as_matrix()
    t_parent = np.eye(4, dtype=float)
    t_parent[:3, :3] = parent_r
    t_parent[:3, 3] = parent_t
    t_parent_inv = np.linalg.inv(t_parent)

    for params in params_list:
        spec = get_grasp_spec(obj, manipulation_params=params)
        if spec is None:
            continue

        # get_grasp_spec() 返回的是物体根坐标系（此处可视作“world”）下的抓取位姿；
        # 需要转成当前 link 的局部坐标再写入 Xform。
        t_grasp_parent = t_parent_inv @ spec["world_transformation_matrix"]
        t_grasp_parent_scaled = t_grasp_parent.copy()
        t_grasp_parent_scaled[:3, 3] = t_grasp_parent_scaled[:3, 3] * float(scale)
        pos_local = t_grasp_parent_scaled[:3, 3]
        rot_local = t_grasp_parent[:3, :3]
        quat_local = Rot.from_matrix(rot_local).as_quat()  # [x, y, z, w]
        approach_local = rot_local @ np.array([0.0, 0.0, 1.0], dtype=float)
        finger_local = rot_local @ np.array([1.0, 0.0, 0.0], dtype=float)

        grasp_path = f"{link_path}/grasps/grasp_{valid_idx}"
        g_xform = UsdGeom.Xform.Define(stage, grasp_path)
        g_xform.ClearXformOpOrder()
        g_xform.AddTranslateOp().Set(to_vec3d(pos_local))
        g_xform.AddOrientOp().Set(to_quatf_xyzw(quat_local))
        g_prim = g_xform.GetPrim()
        g_prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(
            to_vec3f(approach_local)
        )
        if "world_finger_closing_direction" in spec:
            g_prim.CreateAttribute("grasp:finger_closing", Sdf.ValueTypeNames.Vector3f).Set(
                to_vec3f(finger_local)
            )
        if "grasp_width" in spec:
            g_prim.CreateAttribute("grasp:width", Sdf.ValueTypeNames.Float).Set(float(spec["grasp_width"]))
        if "manip_params_size" in spec:
            g_prim.CreateAttribute("grasp:manip_params_size", Sdf.ValueTypeNames.Int).Set(
                int(spec["manip_params_size"])
            )
        g_prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(
            t_grasp_parent_scaled.flatten().tolist()
        )
        valid_idx += 1


def _build_grasp_params_for_door(door_obj, door_idx):
    # get_grasp_spec for door expects: (trans_ratio, rot_ratio, door_idx)
    params = []
    for trans_ratio in (-0.5, 0.0, 0.5):
        for rot_ratio in (0.0,):
            params.append((trans_ratio, rot_ratio, door_idx))
    return params


def _build_grasp_params_for_drawer(drawer_obj, drawer_idx):
    # get_grasp_spec for drawer expects: (trans_ratio, rot_ratio, drawer_idx, handle_idx)
    params = []
    handle_num = int(drawer_obj.number_of_handle[drawer_idx]) if drawer_idx < len(drawer_obj.number_of_handle) else 1
    for handle_idx in range(max(handle_num, 1)):
        for trans_ratio in (-0.5, 0.0, 0.5):
            for rot_ratio in (0.0,):
                params.append((trans_ratio, rot_ratio, drawer_idx, handle_idx))
    return params


def _adapt_storagefurniture_body_params(params):
    def _scalar(v, default=0.0):
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 0:
                return default
            return v[0]
        return v

    def _vec3(v):
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
            return [v[0], v[1], v[2]]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
            return [v[0], 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    adapted = dict(params)
    whole_number = adapted.get("WHOLE_number_of_layer", [0])
    try:
        whole_count = int(whole_number[0])
    except Exception:
        whole_count = 0

    if "storagefurniture_layers_params" not in adapted:
        layer_params = []
        for i in range(1, whole_count + 2):
            num = int(_scalar(adapted.get(f"number_of_{i}_layer", 0), default=0))
            size_pair = adapted.get(f"layer_{i}_sizes", [0.0, 0.0])
            offset = float(_scalar(adapted.get(f"layer_{i}_offset", 0.0), default=0.0))
            interval = float(_scalar(adapted.get(f"interval_between_{i}_layers", 0.0), default=0.0))
            layer_params.extend([num, size_pair[0], size_pair[1], offset, interval])
        adapted["storagefurniture_layers_params"] = layer_params

    if "additional_layers_params" not in adapted:
        add_num = int(_scalar(adapted.get("number_of_additional_layers", 0), default=0))
        additional_layers_params = [add_num]
        for i in range(1, add_num + 1):
            size = _vec3(adapted.get(f"size_{i}", [0.0, 0.0, 0.0]))
            offset = _vec3(adapted.get(f"offset_{i}", [0.0, 0.0, 0.0]))
            rotation = _vec3(adapted.get(f"rotation_{i}", [0.0, 0.0, 0.0]))
            additional_layers_params.extend(
                [size[0], size[1], size[2], offset[0], offset[1], offset[2], rotation[0], rotation[1], rotation[2]]
            )
        adapted["additional_layers_params"] = additional_layers_params

    return adapted


def _adapt_template_parameters(template_name, params):
    adapted = dict(params)
    if template_name == "Storagefurniture_body":
        adapted = _adapt_storagefurniture_body_params(adapted)
    return adapted


def _instantiate_template_object(template_name, raw_params):
    cls = globals().get(template_name)
    if cls is None:
        raise ValueError(f"Unknown template class: {template_name}")

    params = _adapt_template_parameters(template_name, raw_params)
    sig = inspect.signature(cls.__init__)
    accepted = {k for k in sig.parameters.keys() if k != "self"}
    filtered = {k: v for k, v in params.items() if k in accepted}
    return cls(**filtered)


def export_storagefurniture_usda(
    data,
    save_path,
    scale=1.0,
    init_pos=(0.0, 0.0, 0.0),
    init_euler=(0.0, 0.0, 0.0),
    anchor_base=True,
    dynamic_base_mass=350.0,
    dynamic_base_linear_damping=2.5,
    dynamic_base_angular_damping=8.0,
    door_joint_drive_damping=60.0,
    door_joint_drive_stiffness=0.0,
    drawer_joint_drive_damping=180.0,
    drawer_joint_drive_stiffness=0.0,
    close_drawer_initially=True,
):
    stage = Usd.Stage.CreateNew(save_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    root_path = "/StorageFurniture"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    set_initial_pose(root, init_pos, init_euler)
    # 家具主体必须固定在世界中：仅 door / drawer 通过关节可动。
    if not anchor_base:
        print("[export_storagefurniture_usda] anchor_base=False is overridden to True to keep furniture body fixed.")
    anchor_base = True
    # 仅在非锚定模式下启用 articulation root。
    # 在 Isaac/PhysX 中，articulation + kinematic base 会报错。
    if not anchor_base:
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute("dataset:id", Sdf.ValueTypeNames.String).Set(str(data.get("id", "unknown")))

    links_path = f"{root_path}/links"
    joints_path = f"{root_path}/joints"
    UsdGeom.Xform.Define(stage, links_path)
    UsdGeom.Xform.Define(stage, joints_path)

    components = []
    for c in data["conceptualization"]:
        obj = _instantiate_template_object(c["template"], c["parameters"])
        components.append((c["template"], obj))

    base_vertices = []
    base_faces = []
    base_vert_offset = 0
    door_links = []
    drawer_links = []
    door_parts_for_opening = []
    global_door_counter = 0
    global_drawer_counter = 0

    for template_name, obj in components:
        if template_name == "Regular_door":
            door_parts_for_opening.append(obj)
            per_door_faces = 12 * 2
            door_parts = _split_mesh_by_face_blocks(obj.overall_obj_mesh, per_door_faces, obj.number_of_door[0])
            for door_idx, (verts, faces) in enumerate(door_parts):
                global_door_idx = global_door_counter
                global_door_counter += 1
                link_name = f"door_{global_door_idx}"
                link_path = f"{links_path}/{link_name}"
                if len(verts) == 0:
                    continue
                hinge_world = _door_joint_origin_world(obj, door_idx)
                local_verts = verts - hinge_world[None, :].astype(np.float32)
                local_verts = local_verts * float(scale)
                _create_link(
                    stage,
                    link_path,
                    local_verts,
                    faces,
                    mass=1.0,
                    collision_approx="convexHull",
                    translate=np.array(hinge_world, dtype=float) * float(scale),
                    orient_xyzw=(0.0, 0.0, 0.0, 1.0),
                    linear_damping=1.0,
                    angular_damping=2.5,
                )
                door_links.append((global_door_idx, door_idx, link_name, obj, hinge_world))
            continue

        if template_name in ("Regular_drawer", "Drawer_with_U_handle"):
            drawer_parts_per_obj = []
            for drawer_idx in range(obj.number_of_drawer[0]):
                # Drawer_with_U_handle 的每个把手由 3 个 cuboid 组成；
                # Regular_drawer 的每个把手由 1 个 cuboid 组成。
                if template_name == "Drawer_with_U_handle":
                    cuboid_count = 6 + 3 * int(obj.number_of_handle[drawer_idx])
                else:
                    cuboid_count = 6 + int(obj.number_of_handle[drawer_idx])
                drawer_faces = 12 * cuboid_count
                parts = _split_mesh_by_face_blocks(obj.overall_obj_mesh, drawer_faces, obj.number_of_drawer[0])
                drawer_parts_per_obj = parts
                verts, faces = drawer_parts_per_obj[drawer_idx]
                global_drawer_idx = global_drawer_counter
                global_drawer_counter += 1
                link_name = f"drawer_{global_drawer_idx}"
                link_path = f"{links_path}/{link_name}"
                if len(verts) == 0:
                    continue
                lower, upper = _drawer_joint_limits(obj, drawer_idx)
                if close_drawer_initially:
                    verts = verts.copy()
                    # 参考 simple-collision 导出逻辑：将抽屉几何回推到 q=0 的闭合位。
                    verts[:, 2] = verts[:, 2] - float(upper)
                verts = verts * float(scale)
                _create_link(
                    stage,
                    link_path,
                    verts,
                    faces,
                    mass=1.5,
                    collision_approx="convexDecomposition",
                    linear_damping=1.2,
                    angular_damping=3.0,
                )
                drawer_links.append((global_drawer_idx, drawer_idx, link_name, obj))
            continue

        verts = np.asarray(obj.vertices, dtype=np.float32)
        faces = np.asarray(obj.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            continue
        base_vertices.append(verts)
        base_faces.append(faces + base_vert_offset)
        base_vert_offset += len(verts)

    if len(base_vertices) == 0:
        raise RuntimeError("No base mesh found for StorageFurniture.")
    base_verts = np.concatenate(base_vertices, axis=0)
    base_faces = np.concatenate(base_faces, axis=0)
    base_faces = _carve_base_openings(base_verts, base_faces, door_parts_for_opening)
    base_verts = base_verts * float(scale)
    _create_link(
        stage,
        f"{links_path}/base_link",
        base_verts,
        base_faces,
        mass=2000.0 if anchor_base else dynamic_base_mass,
        kinematic=bool(anchor_base),
        collision_approx="convexDecomposition",
        linear_damping=None if anchor_base else dynamic_base_linear_damping,
        angular_damping=None if anchor_base else dynamic_base_angular_damping,
    )

    for global_door_idx, local_door_idx, link_name, door_obj, hinge_world in door_links:
        joint_path = f"{joints_path}/door_joint_{global_door_idx}"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Y")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{links_path}/base_link")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{links_path}/{link_name}")])

        joint.CreateLocalPos0Attr().Set(to_vec3f(np.array(hinge_world, dtype=float) * float(scale)))
        joint.CreateLocalPos1Attr().Set(to_vec3f((0.0, 0.0, 0.0)))
        low_deg, high_deg = _door_joint_limits_deg(door_obj, local_door_idx)
        joint.CreateLowerLimitAttr().Set(float(low_deg))
        joint.CreateUpperLimitAttr().Set(float(high_deg))
        joint.CreateCollisionEnabledAttr().Set(False)
        # 小阻尼初值：减小轻微倾斜时的自开趋势，同时不把门“锁死”。
        door_drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        door_drive.CreateTypeAttr().Set("force")
        door_drive.CreateStiffnessAttr().Set(float(door_joint_drive_stiffness))
        door_drive.CreateDampingAttr().Set(float(door_joint_drive_damping))
        door_drive.CreateMaxForceAttr().Set(1.0e8)

        _add_grasps(
            stage,
            f"{links_path}/{link_name}",
            door_obj,
            _build_grasp_params_for_door(door_obj, local_door_idx),
            parent_translate=hinge_world,
            parent_orient_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=scale,
        )

    for global_drawer_idx, local_drawer_idx, link_name, drawer_obj in drawer_links:
        joint_path = f"{joints_path}/drawer_joint_{global_drawer_idx}"
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{links_path}/base_link")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{links_path}/{link_name}")])

        origin = np.array(drawer_obj.drawer_offset[local_drawer_idx], dtype=float)
        r_part = Rot.from_euler("xyz", drawer_obj.rotation, degrees=False).as_matrix()
        p_part = np.array(drawer_obj.position, dtype=float)
        origin_world = p_part + r_part @ origin
        origin_world_scaled = origin_world * float(scale)
        joint.CreateLocalPos0Attr().Set(to_vec3f(origin_world_scaled))
        joint.CreateLocalPos1Attr().Set(to_vec3f(origin_world_scaled))
        lower, upper = _drawer_joint_limits(drawer_obj, local_drawer_idx)
        joint.CreateLowerLimitAttr().Set(float(lower * float(scale)))
        joint.CreateUpperLimitAttr().Set(float(upper * float(scale)))
        joint.CreateCollisionEnabledAttr().Set(False)
        # 小阻尼初值：抑制轻微倾斜时滑出，同时保留可拉动性。
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(drawer_joint_drive_stiffness))
        drive.CreateDampingAttr().Set(float(drawer_joint_drive_damping))
        drive.CreateMaxForceAttr().Set(1.0e8)

        grasp_obj = drawer_obj
        if close_drawer_initially:
            grasp_obj = copy.deepcopy(drawer_obj)
            grasp_obj.drawer_offset[local_drawer_idx][2] = (
                float(grasp_obj.drawer_offset[local_drawer_idx][2]) - float(upper)
            )

        _add_grasps(
            stage,
            f"{links_path}/{link_name}",
            grasp_obj,
            _build_grasp_params_for_drawer(grasp_obj, local_drawer_idx),
            parent_translate=(0.0, 0.0, 0.0),
            parent_orient_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=scale,
        )

    stage.GetRootLayer().Save()
    print(f"Exported StorageFurniture [{data.get('id', 'unknown')}] to: {save_path}")


if __name__ == "__main__":
    with open("whk_new.pkl", "rb") as f:
        data_list = pickle.load(f)

    output_dir = "storagefurniture_usda_outputs"
    os.makedirs(output_dir, exist_ok=True)

    SCALE_FACTOR = 0.0023

    # 根据需要改范围；默认导出前 10 个样本。
    for i, data in enumerate(data_list[:1]):
        save_name = f"storagefurniture_{i}.usda"
        export_storagefurniture_usda(
            data,
            os.path.join(output_dir, save_name),
            scale=SCALE_FACTOR,
            init_pos=(0.0, 0.0, 0.0),
            init_euler=(0.0, 0.0, 0.0),
            anchor_base=True,
            dynamic_base_mass=350.0,
            dynamic_base_linear_damping=2.5,
            dynamic_base_angular_damping=8.0,
        )
