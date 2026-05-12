import os
import pickle
import re
import copy

import numpy as np
from concept_template import *
from knowledge_definitions import get_grasp_spec
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as Rot


def _to_vec3f(v):
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def _to_vec3d(v):
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def _to_quatf_xyzw(q_xyzw):
    return Gf.Quatf(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _set_initial_pose(prim_xform, position, euler_deg):
    prim_xform.ClearXformOpOrder()
    prim_xform.AddTranslateOp().Set(_to_vec3d(position))
    r = Rot.from_euler("xyz", euler_deg, degrees=True)
    prim_xform.AddOrientOp().Set(_to_quatf_xyzw(r.as_quat()))


def _load_obj_vertices_faces_uv(path):
    verts = []
    uvs = []
    faces = []
    face_uvs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = []
                uv_idx = []
                for p in parts:
                    toks = p.split("/")
                    v_token = toks[0] if len(toks) > 0 else ""
                    vt_token = toks[1] if len(toks) > 1 else ""
                    if not v_token:
                        continue
                    i = int(v_token)
                    i = len(verts) + i if i < 0 else i - 1
                    idx.append(i)
                    if vt_token:
                        t = int(vt_token)
                        t = len(uvs) + t if t < 0 else t - 1
                        uv_idx.append(t)
                    else:
                        uv_idx.append(-1)
                if len(idx) < 3:
                    continue
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
                    face_uvs.append([uv_idx[0], uv_idx[k], uv_idx[k + 1]])

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError(f"empty or invalid OBJ mesh: {path}")

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        np.asarray(uvs, dtype=np.float32) if len(uvs) > 0 else None,
        np.asarray(face_uvs, dtype=np.int32),
    )


def _create_mesh(stage, mesh_path, verts, faces, scale, collision=False, approximation="convexHull", uvs=None, face_uvs=None):
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    verts = np.asarray(verts, dtype=np.float32) * float(scale)
    faces = np.asarray(faces, dtype=np.int32)

    mesh.CreatePointsAttr([_to_vec3f(v) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

    if uvs is not None and face_uvs is not None and len(uvs) > 0 and len(face_uvs) == len(faces):
        st_data = []
        for tri_uv in face_uvs:
            for uv_i in tri_uv:
                if 0 <= uv_i < len(uvs):
                    st = uvs[uv_i]
                    st_data.append(Gf.Vec2f(float(st[0]), float(st[1])))
                else:
                    st_data.append(Gf.Vec2f(0.0, 0.0))
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st_primvar.Set(st_data)

    if collision:
        prim = mesh.GetPrim()
        UsdGeom.Imageable(mesh).CreatePurposeAttr().Set("physics")
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approximation)

    return mesh


def _bind_preview_texture(stage, mesh_prim, texture_path, material_name):
    looks_scope = UsdGeom.Scope.Define(stage, "/StorageFurniture/Looks")
    mat = UsdShade.Material.Define(stage, f"{looks_scope.GetPath()}/{material_name}")
    shader = UsdShade.Shader.Define(stage, f"{looks_scope.GetPath()}/{material_name}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, f"{looks_scope.GetPath()}/{material_name}/TexCoordReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{looks_scope.GetPath()}/{material_name}/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)


def _create_link_xform(
    stage,
    link_path,
    mass,
    kinematic=False,
    translate=(0.0, 0.0, 0.0),
    orient_xyzw=(0.0, 0.0, 0.0, 1.0),
    linear_damping=None,
    angular_damping=None,
):
    link = UsdGeom.Xform.Define(stage, link_path)
    link.ClearXformOpOrder()
    link.AddTranslateOp().Set(_to_vec3d(translate))
    link.AddOrientOp().Set(_to_quatf_xyzw(orient_xyzw))
    link.AddScaleOp().Set(_to_vec3f((1.0, 1.0, 1.0)))

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


def _drawer_joint_limits(part, drawer_idx):
    travel_proportion = 0.95
    drawer_len = float(part.drawer_size[drawer_idx][2])
    travel_total = drawer_len * travel_proportion
    lower = 0.0
    upper = max(0.0, travel_total)
    return lower, upper


def _add_grasps(
    stage,
    link_path,
    obj,
    params_list,
    parent_translate=(0.0, 0.0, 0.0),
    parent_orient_xyzw=(0.0, 0.0, 0.0, 1.0),
    scale=1.0,
    grasp_name_prefix="grasp",
):
    UsdGeom.Xform.Define(stage, f"{link_path}/grasps")
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

        t_grasp_parent = t_parent_inv @ spec["world_transformation_matrix"]

        t_grasp_parent_scaled = t_grasp_parent.copy()
        t_grasp_parent_scaled[:3, 3] = t_grasp_parent_scaled[:3, 3] * float(scale)

        pos_local = t_grasp_parent_scaled[:3, 3]
        rot_local = t_grasp_parent[:3, :3]
        quat_local = Rot.from_matrix(rot_local).as_quat()
        approach_local = rot_local @ np.array([0.0, 0.0, 1.0], dtype=float)
        finger_local = rot_local @ np.array([1.0, 0.0, 0.0], dtype=float)

        grasp_path = f"{link_path}/grasps/{grasp_name_prefix}_{valid_idx}"
        g_xform = UsdGeom.Xform.Define(stage, grasp_path)
        g_xform.ClearXformOpOrder()
        g_xform.AddTranslateOp().Set(_to_vec3d(pos_local))
        g_xform.AddOrientOp().Set(_to_quatf_xyzw(quat_local))

        g_prim = g_xform.GetPrim()
        g_prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(_to_vec3f(approach_local))
        if "world_finger_closing_direction" in spec:
            g_prim.CreateAttribute("grasp:finger_closing", Sdf.ValueTypeNames.Vector3f).Set(_to_vec3f(finger_local))
        if "grasp_width" in spec:
            g_prim.CreateAttribute("grasp:width", Sdf.ValueTypeNames.Float).Set(float(spec["grasp_width"]))
        if "manip_params_size" in spec:
            g_prim.CreateAttribute("grasp:manip_params_size", Sdf.ValueTypeNames.Int).Set(int(spec["manip_params_size"]))
        g_prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(t_grasp_parent_scaled.flatten().tolist())
        valid_idx += 1


def _disable_collision_between_prims(stage, prim_paths):
    valid_paths = []
    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            valid_paths.append(p)
    for i in range(len(valid_paths)):
        pi = stage.GetPrimAtPath(valid_paths[i])
        rel = pi.GetRelationship("physics:filteredPairs")
        if not rel.IsValid():
            rel = pi.CreateRelationship("physics:filteredPairs", False)
        current = set(rel.GetTargets())
        for j in range(len(valid_paths)):
            if i == j:
                continue
            current.add(Sdf.Path(valid_paths[j]))
        rel.SetTargets(sorted(current, key=lambda x: str(x)))


def _build_grasp_params_for_drawer(drawer_obj, drawer_idx):
    params = []
    handle_num = int(drawer_obj.number_of_handle[drawer_idx]) if drawer_idx < len(drawer_obj.number_of_handle) else 1
    for handle_idx in range(max(handle_num, 1)):
        for trans_ratio in (-0.5, 0.0, 0.5):
            params.append((trans_ratio, 0.0, drawer_idx, handle_idx))
    return params


def _load_concept_data(concept_pkl_path):
    with open(concept_pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"empty pkl list: {concept_pkl_path}")
        data = data[0]
    if not isinstance(data, dict) or "conceptualization" not in data:
        raise ValueError(f"invalid conceptualization format: {concept_pkl_path}")
    return data


def _build_storagefurniture_body_args(p):
    whole_n = int(p["WHOLE_number_of_layer"][0])
    storagefurniture_layers_params = []
    for idx in range(1, whole_n + 2):
        storagefurniture_layers_params.extend(p[f"number_of_{idx}_layer"])
        storagefurniture_layers_params.extend(p[f"layer_{idx}_sizes"])
        storagefurniture_layers_params.extend(p[f"layer_{idx}_offset"])
        storagefurniture_layers_params.extend(p[f"interval_between_{idx}_layers"])

    add_n = int(p["number_of_additional_layers"][0])
    additional_layers_params = [add_n]
    for idx in range(1, add_n + 1):
        additional_layers_params.extend(p[f"size_{idx}"])
        additional_layers_params.extend(p[f"offset_{idx}"])
        additional_layers_params.extend(p[f"rotation_{idx}"])

    return dict(
        size=p["size"],
        back_size=p["back_size"],
        left_right_inner_size=p["left_right_inner_size"],
        base_size=p["base_size"],
        has_lid=p["has_lid"],
        lid_size=p["lid_size"],
        lid_offset=p["lid_offset"],
        WHOLE_number_of_layer=p["WHOLE_number_of_layer"],
        WHOLE_layer_sizes=p["WHOLE_layer_sizes"],
        WHOLE_layer_offset=p["WHOLE_layer_offset"],
        WHOLE_interval_between_layers=p["WHOLE_interval_between_layers"],
        storagefurniture_layers_params=storagefurniture_layers_params,
        additional_layers_params=additional_layers_params,
        position=p.get("position", [0, 0, 0]),
        rotation=p.get("rotation", [0, 0, 0]),
    )


def _build_drawer_args(p, handle_size_len=3):
    if "drawers_params" in p:
        return dict(
            number_of_drawer=p["number_of_drawer"],
            drawers_params=list(p["drawers_params"]),
            position=p.get("position", [0, 0, 0]),
            rotation=p.get("rotation", [0, 0, 0]),
        )

    n = int(p["number_of_drawer"][0])
    def _item(v, i):
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)):
            return list(v[i])
        return list(v)
    def _scalar(v, i):
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)):
            return v[i][0] if len(v[i]) > 0 else 0.0
        return v[i]
    drawers_params = []
    for i in range(n):
        drawers_params.extend(_item(p["drawer_size"], i))
        drawers_params.extend([_scalar(p["bottom_size"], i)])
        drawers_params.extend(_item(p["front_size"], i))
        drawers_params.extend([_scalar(p["front_offset"], i)])
        drawers_params.extend([_scalar(p["left_right_inner_size"], i)])
        drawers_params.extend([_scalar(p["rear_front_inner_size"], i)])
        drawers_params.extend([_scalar(p["number_of_handle"], i)])
        handle_size_item = _item(p["handle_sizes"], i)
        if len(handle_size_item) < handle_size_len:
            handle_size_item = handle_size_item + [handle_size_item[-1] if len(handle_size_item) > 0 else 0.01] * (handle_size_len - len(handle_size_item))
        drawers_params.extend(handle_size_item[:handle_size_len])
        drawers_params.extend(_item(p["handle_offset"], i))
        drawers_params.extend([_scalar(p["handle_separation"], i)])
        drawers_params.extend(_item(p["drawer_offset"], i))
    return dict(
        number_of_drawer=p["number_of_drawer"],
        drawers_params=drawers_params,
        position=p.get("position", [0, 0, 0]),
        rotation=p.get("rotation", [0, 0, 0]),
    )


def _instantiate_concept_obj(template_name, parameters):
    cls = eval(template_name)
    if template_name == "Storagefurniture_body":
        return cls(**_build_storagefurniture_body_args(parameters))
    if template_name == "Regular_drawer":
        return cls(**_build_drawer_args(parameters, handle_size_len=3))
    if template_name == "Drawer_with_U_handle":
        return cls(**_build_drawer_args(parameters, handle_size_len=3))
    return cls(**parameters)


def _collect_real_objs(segmentation_dir):
    body_obj = os.path.join(segmentation_dir, "Body.obj")
    drawer_objs = []
    for name in os.listdir(segmentation_dir):
        m = re.fullmatch(r"Drawer_(\d+)\.obj", name)
        if m:
            drawer_objs.append((int(m.group(1)), os.path.join(segmentation_dir, name)))
    drawer_objs.sort(key=lambda x: x[0])
    return body_obj, drawer_objs


def _match_drawer_entries_to_real_indices(drawer_entries, drawer_obj_paths):
    if len(drawer_entries) == 0 or len(drawer_obj_paths) == 0:
        return []

    real_features = []
    for real_idx, obj_path in drawer_obj_paths:
        rv, _, _, _ = _load_obj_vertices_faces_uv(obj_path)
        real_features.append((real_idx, np.mean(rv, axis=0)))

    conceptual_features = []
    for ent_idx, (_, local_drawer_idx, drawer_obj, coll_v, _) in enumerate(drawer_entries):
        _ = local_drawer_idx
        _ = drawer_obj
        conceptual_features.append((ent_idx, np.mean(np.asarray(coll_v, dtype=np.float32), axis=0)))

    remaining_real = {idx for idx, _ in real_features}
    assigned = {}
    for ent_idx, c_center in conceptual_features:
        best_real = None
        best_dist = float("inf")
        for real_idx, r_center in real_features:
            if real_idx not in remaining_real:
                continue
            d = float(np.linalg.norm(c_center - r_center))
            if d < best_dist:
                best_dist = d
                best_real = real_idx
        if best_real is not None:
            assigned[ent_idx] = best_real
            remaining_real.remove(best_real)

    rem_real_sorted = sorted(list(remaining_real))
    for ent_idx, _ in conceptual_features:
        if ent_idx not in assigned and len(rem_real_sorted) > 0:
            assigned[ent_idx] = rem_real_sorted.pop(0)

    rem_used = set(assigned.values())
    next_idx = 0
    out = []
    for ent_idx, entry in enumerate(drawer_entries):
        if ent_idx in assigned:
            real_idx = assigned[ent_idx]
        else:
            while next_idx in rem_used:
                next_idx += 1
            real_idx = next_idx
            rem_used.add(real_idx)
            next_idx += 1
        out.append((real_idx, entry[1], entry[2], entry[3], entry[4]))
    return out


def export_with_simple_collision(
    texture_path,
    segmentation_dir,
    concept_pkl_path,
    output_usda_path,
    scale_to_meters=0.0015,
    init_pos=(0.0, 0.0, 0.0),
    init_euler=(0.0, 0.0, 0.0),
    anchor_base=True,
    base_mass_kg=2000.0,
    drawer_mass_kg=1.5,
    close_drawer_initially=True,
):
    texture_path = os.path.abspath(texture_path)
    segmentation_dir = os.path.abspath(segmentation_dir)
    concept_pkl_path = os.path.abspath(concept_pkl_path)
    output_usda_path = os.path.abspath(output_usda_path)

    for p in [texture_path, segmentation_dir, concept_pkl_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    body_obj_path, drawer_obj_paths = _collect_real_objs(segmentation_dir)
    if not os.path.exists(body_obj_path):
        raise FileNotFoundError(body_obj_path)

    concept_data = _load_concept_data(concept_pkl_path)

    os.makedirs(os.path.dirname(output_usda_path), exist_ok=True)
    stage = Usd.Stage.CreateNew(output_usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    root_path = "/StorageFurniture"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    _set_initial_pose(root, init_pos, init_euler)
    root.GetPrim().CreateAttribute("dataset:id", Sdf.ValueTypeNames.String).Set(str(concept_data.get("id", "unknown")))

    links_path = f"{root_path}/links"
    joints_path = f"{root_path}/joints"
    UsdGeom.Xform.Define(stage, links_path)
    UsdGeom.Xform.Define(stage, joints_path)

    components = []
    for c in concept_data["conceptualization"]:
        obj = _instantiate_concept_obj(c["template"], c["parameters"])
        components.append((c["template"], obj))

    base_collision_vertices = []
    base_collision_faces = []
    base_vert_offset = 0
    drawer_collision_entries = []
    global_drawer_counter = 0

    for template_name, obj in components:
        if template_name in ("Regular_drawer", "Drawer_with_U_handle"):
            for drawer_idx in range(obj.number_of_drawer[0]):
                cuboid_count = 6 + int(obj.number_of_handle[drawer_idx])
                drawer_faces = 12 * cuboid_count
                parts = _split_mesh_by_face_blocks(obj.overall_obj_mesh, drawer_faces, obj.number_of_drawer[0])
                verts, faces = parts[drawer_idx]
                if len(verts) == 0:
                    continue
                drawer_collision_entries.append((global_drawer_counter, drawer_idx, obj, verts, faces))
                global_drawer_counter += 1
            continue

        verts = np.asarray(obj.vertices, dtype=np.float32)
        faces = np.asarray(obj.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            continue
        base_collision_vertices.append(verts)
        base_collision_faces.append(faces + base_vert_offset)
        base_vert_offset += len(verts)

    if len(base_collision_vertices) == 0:
        raise RuntimeError("No base mesh found for StorageFurniture concept data.")

    base_link_path = f"{links_path}/base_link"
    _create_link_xform(stage, base_link_path, mass=base_mass_kg, kinematic=bool(anchor_base))

    body_verts, body_faces, body_uvs, body_face_uvs = _load_obj_vertices_faces_uv(body_obj_path)
    _create_mesh(stage, f"{base_link_path}/visual", body_verts, body_faces, scale_to_meters, collision=False, uvs=body_uvs, face_uvs=body_face_uvs)
    _bind_preview_texture(stage, stage.GetPrimAtPath(f"{base_link_path}/visual"), texture_path, "BaseMat")

    base_coll_v = np.concatenate(base_collision_vertices, axis=0)
    base_coll_f = np.concatenate(base_collision_faces, axis=0)
    _create_mesh(stage, f"{base_link_path}/collision", base_coll_v, base_coll_f, scale_to_meters, collision=True, approximation="convexDecomposition")

    drawer_obj_map = {idx: path for idx, path in drawer_obj_paths}
    drawer_collision_entries.sort(key=lambda x: x[0])
    drawer_collision_entries = _match_drawer_entries_to_real_indices(drawer_collision_entries, drawer_obj_paths)
    drawer_collision_prim_paths = []

    for real_drawer_idx, local_drawer_idx, drawer_obj, coll_v, coll_f in drawer_collision_entries:
        link_name = f"drawer_{real_drawer_idx}"
        link_path = f"{links_path}/{link_name}"
        _create_link_xform(
            stage,
            link_path,
            mass=drawer_mass_kg,
            kinematic=False,
            linear_damping=20.0,
            angular_damping=40.0,
        )

        # Some real segmented drawers are captured in a pulled-out pose.
        # Shift drawer geometry along joint axis so q=0 starts from a closed pose.
        lower, upper = _drawer_joint_limits(drawer_obj, local_drawer_idx)
        drawer_geom_offset = np.zeros((3,), dtype=np.float32)
        if close_drawer_initially:
            drawer_geom_offset[2] = -float(upper) 

        if real_drawer_idx in drawer_obj_map:
            dv, df, duvs, dfuvs = _load_obj_vertices_faces_uv(drawer_obj_map[real_drawer_idx])
            _create_mesh(
                stage,
                f"{link_path}/visual",
                dv + drawer_geom_offset[None, :],
                df,
                scale_to_meters,
                collision=False,
                uvs=duvs,
                face_uvs=dfuvs,
            )
            _bind_preview_texture(stage, stage.GetPrimAtPath(f"{link_path}/visual"), texture_path, f"DrawerMat_{real_drawer_idx}")

        _create_mesh(
            stage,
            f"{link_path}/collision",
            coll_v + drawer_geom_offset[None, :],
            coll_f,
            scale_to_meters,
            collision=True,
            approximation="convexDecomposition",
        )
        drawer_collision_prim_paths.append(f"{link_path}/collision")

        joint_path = f"{joints_path}/drawer_joint_{real_drawer_idx}"
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(base_link_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])

        origin = np.array(drawer_obj.drawer_offset[local_drawer_idx], dtype=float)
        r_part = Rot.from_euler("xyz", drawer_obj.rotation, degrees=False).as_matrix()
        p_part = np.array(drawer_obj.position, dtype=float)
        origin_world = p_part + r_part @ origin
        origin_world_scaled = origin_world * float(scale_to_meters)
        joint.CreateLocalPos0Attr().Set(_to_vec3f(origin_world_scaled))
        joint.CreateLocalPos1Attr().Set(_to_vec3f(origin_world_scaled))

        joint.CreateLowerLimitAttr().Set(float(lower * float(scale_to_meters)))
        joint.CreateUpperLimitAttr().Set(float(upper * float(scale_to_meters)))
        joint.CreateCollisionEnabledAttr().Set(False)

        grasp_obj = drawer_obj
        if close_drawer_initially:
            grasp_obj = copy.deepcopy(drawer_obj)
            grasp_obj.drawer_offset[local_drawer_idx][2] = float(grasp_obj.drawer_offset[local_drawer_idx][2]) - float(upper)

        _add_grasps(
            stage,
            link_path,
            grasp_obj,
            _build_grasp_params_for_drawer(grasp_obj, local_drawer_idx),
            parent_translate=(0.0, 0.0, 0.0),
            parent_orient_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=scale_to_meters
        )

    _disable_collision_between_prims(stage, drawer_collision_prim_paths)

    stage.GetRootLayer().Save()
    return output_usda_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(repo_root, "real_object_data", "StorageFurniture")

    texture_path = os.path.join(data_dir, "textures", "texture.jpg")
    segmentation_dir = os.path.join(data_dir, "segmentation")
    concept_pkl_path = os.path.join(data_dir, "conceptualization", "whk_new.pkl")
    output_usda_path = os.path.join(data_dir, "usda_output", "StorageFurniture_drawer_clean_simple_collision.usda")

    out = export_with_simple_collision(
        texture_path=texture_path,
        segmentation_dir=segmentation_dir,
        concept_pkl_path=concept_pkl_path,
        output_usda_path=output_usda_path,
        scale_to_meters=0.0020,
        init_pos=(0.0, 0.0, 0.0),
        init_euler=(0.0, 0.0, 0.0),
        anchor_base=True,
    )
    print(f"[OK] exported: {out}")


if __name__ == "__main__":
    main()
