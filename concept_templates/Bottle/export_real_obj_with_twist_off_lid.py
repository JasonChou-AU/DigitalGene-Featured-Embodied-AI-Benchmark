import os
import pickle
from pathlib import Path

import numpy as np
from concept_template import *
from geometry_template import *
from knowledge_definitions import *
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as Rot


def _to_vec3f(v):
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def _to_vec3d(v):
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def _set_initial_pose(prim_xform, position, euler_deg):
    prim_xform.ClearXformOpOrder()
    prim_xform.AddTranslateOp().Set(_to_vec3d(position))
    r = Rot.from_euler("xyz", euler_deg, degrees=True)
    q = r.as_quat()  # [x, y, z, w]
    quat_usd = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    prim_xform.AddOrientOp().Set(quat_usd)


def _set_identity_xform(prim_xform):
    prim_xform.ClearXformOpOrder()
    prim_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    prim_xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


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


def _add_visual_mesh(stage, prim_path, verts, faces, scale, uvs=None, face_uvs=None):
    geom = UsdGeom.Mesh.Define(stage, prim_path)
    verts = np.asarray(verts, dtype=np.float32) * float(scale)
    faces = np.asarray(faces, dtype=np.int32)

    geom.CreatePointsAttr([_to_vec3f(v) for v in verts])
    geom.CreateFaceVertexCountsAttr([3] * len(faces))
    geom.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    geom.CreateExtentAttr([
        _to_vec3f(np.min(verts, axis=0)),
        _to_vec3f(np.max(verts, axis=0)),
    ])

    if uvs is not None and face_uvs is not None and len(uvs) > 0 and len(face_uvs) == len(faces):
        st_data = []
        for tri_uv in face_uvs:
            for uv_i in tri_uv:
                if 0 <= uv_i < len(uvs):
                    st = uvs[uv_i]
                    st_data.append(Gf.Vec2f(float(st[0]), float(st[1])))
                else:
                    st_data.append(Gf.Vec2f(0.0, 0.0))
        primvars_api = UsdGeom.PrimvarsAPI(geom.GetPrim())
        st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st_primvar.Set(st_data)
    return geom


def _copy_visual_mesh(stage, prim_path, source_mesh):
    geom = UsdGeom.Mesh.Define(stage, prim_path)
    geom.CreatePointsAttr(source_mesh.GetPointsAttr().Get())
    geom.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get())
    geom.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get())

    points = source_mesh.GetPointsAttr().Get()
    if points:
        verts = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float32)
        geom.CreateExtentAttr([
            _to_vec3f(np.min(verts, axis=0)),
            _to_vec3f(np.max(verts, axis=0)),
        ])

    source_primvars = UsdGeom.PrimvarsAPI(source_mesh.GetPrim())
    dest_primvars = UsdGeom.PrimvarsAPI(geom.GetPrim())
    for source_pv in source_primvars.GetPrimvars():
        name = source_pv.GetBaseName()
        if name != "st":
            continue
        dest_pv = dest_primvars.CreatePrimvar(
            name,
            source_pv.GetTypeName(),
            source_pv.GetInterpolation(),
        )
        dest_pv.Set(source_pv.Get())
        if source_pv.GetIndicesAttr().IsValid():
            dest_pv.SetIndices(source_pv.GetIndices())
    return geom


def _add_collision_mesh(stage, prim_path, verts, faces, scale):
    geom = UsdGeom.Mesh.Define(stage, prim_path)
    verts = np.asarray(verts, dtype=np.float32) * float(scale)
    faces = np.asarray(faces, dtype=np.int32)

    geom.CreatePointsAttr([_to_vec3f(v) for v in verts])
    geom.CreateFaceVertexCountsAttr([3] * len(faces))
    geom.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    geom.CreateExtentAttr([
        _to_vec3f(np.min(verts, axis=0)),
        _to_vec3f(np.max(verts, axis=0)),
    ])

    prim = geom.GetPrim()
    UsdGeom.Imageable(geom).CreatePurposeAttr().Set("physics")
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexDecomposition")
    return geom


def _create_physics_material(stage, mat_path, static_friction=1.2, dynamic_friction=1.0, restitution=0.0):
    mat = UsdShade.Material.Define(stage, mat_path)
    mat_prim = mat.GetPrim()
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_prim.CreateAttribute("physics:staticFriction", Sdf.ValueTypeNames.Float).Set(float(static_friction))
    mat_prim.CreateAttribute("physics:dynamicFriction", Sdf.ValueTypeNames.Float).Set(float(dynamic_friction))
    mat_prim.CreateAttribute("physics:restitution", Sdf.ValueTypeNames.Float).Set(float(restitution))
    return mat


def _bind_physics_material(collision_prim, physics_material):
    UsdShade.MaterialBindingAPI(collision_prim).Bind(
        physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _bind_preview_texture(stage, mesh_prim, texture_path, mat_name):
    looks_scope = UsdGeom.Scope.Define(stage, "/Bottle/Looks")
    mat = UsdShade.Material.Define(stage, f"{looks_scope.GetPath()}/{mat_name}")
    shader = UsdShade.Shader.Define(stage, f"{looks_scope.GetPath()}/{mat_name}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, f"{looks_scope.GetPath()}/{mat_name}/TexCoordReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{looks_scope.GetPath()}/{mat_name}/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)


def _load_concept_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"empty pkl list: {pkl_path}")
        data = data[0]
    if not isinstance(data, dict) or "conceptualization" not in data:
        raise ValueError(f"invalid conceptualization format: {pkl_path}")
    return data


def _pick_single_file(base_dir, candidates, pattern):
    for rel in candidates:
        p = os.path.join(base_dir, rel)
        if os.path.exists(p):
            return os.path.abspath(p)

    matched = sorted(Path(base_dir).glob(pattern))
    if len(matched) == 0:
        raise FileNotFoundError(f"no file matched: base={base_dir}, pattern={pattern}")
    return str(matched[0].resolve())


def _build_simple_collision_from_concept(concept_data):
    body_meshes = []
    lid_meshes = []
    lid_obj = None

    for c in concept_data["conceptualization"]:
        template_name = c.get("template")
        if template_name is None:
            continue
        obj = eval(template_name)(**c["parameters"])
        if isinstance(obj, Cylindrical_Lid):
            lid_obj = obj
            lid_meshes.append((np.asarray(obj.vertices, dtype=np.float32), np.asarray(obj.faces, dtype=np.int32)))
        else:
            body_meshes.append((np.asarray(obj.vertices, dtype=np.float32), np.asarray(obj.faces, dtype=np.int32)))

    def _merge(mesh_list):
        if len(mesh_list) == 0:
            return None, None
        v_all = []
        f_all = []
        offset = 0
        for v, f in mesh_list:
            v_all.append(v)
            f_all.append(f + offset)
            offset += v.shape[0]
        return np.concatenate(v_all, axis=0), np.concatenate(f_all, axis=0)

    body_v, body_f = _merge(body_meshes)
    lid_v, lid_f = _merge(lid_meshes)
    return body_v, body_f, lid_v, lid_f, lid_obj


def _export_grasps(stage, bottle_root_path, lid_obj, scale_to_meters):
    if lid_obj is None:
        return 0

    grasp_root_path = f"{bottle_root_path}/grasps"
    UsdGeom.Xform.Define(stage, grasp_root_path)
    test_params = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (0.0, 1.0)]

    count = 0
    for idx, (p1, p2) in enumerate(test_params):
        spec = get_grasp_spec(lid_obj, manipulation_params=(p1, p2))
        if spec is None:
            continue

        g_xform = UsdGeom.Xform.Define(stage, f"{grasp_root_path}/grasp_{idx}")
        t = np.asarray(spec["world_position"], dtype=np.float64) * float(scale_to_meters)
        q = spec["world_rotation"]  # [x, y, z, w]
        quat_gf = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
        g_xform.AddTranslateOp().Set(_to_vec3d(t))
        g_xform.AddOrientOp().Set(quat_gf)

        prim = g_xform.GetPrim()
        prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(_to_vec3f(spec["world_approach_direction"]))

        t_mat = np.asarray(spec["world_transformation_matrix"], dtype=np.float64).copy()
        t_mat[:3, 3] *= float(scale_to_meters)
        prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(t_mat.flatten().tolist())
        count += 1
    return count


def _apply_rigid_body(prim, mass_kg):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass_kg))


def _create_link_xform(stage, link_path, mass_kg):
    link = UsdGeom.Xform.Define(stage, link_path)
    _set_identity_xform(link)
    _apply_rigid_body(link.GetPrim(), mass_kg=mass_kg)
    return link


def _apply_articulation_root(prim):
    UsdPhysics.ArticulationRootAPI.Apply(prim)


def _disable_collision_between_prims(stage, prim_paths):
    valid_paths = []
    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            valid_paths.append(p)

    for i, path_i in enumerate(valid_paths):
        prim_i = stage.GetPrimAtPath(path_i)
        rel = prim_i.GetRelationship("physics:filteredPairs")
        if not rel.IsValid():
            rel = prim_i.CreateRelationship("physics:filteredPairs", False)
        targets = set(rel.GetTargets())
        for j, path_j in enumerate(valid_paths):
            if i != j:
                targets.add(Sdf.Path(path_j))
        rel.SetTargets(sorted(targets, key=lambda p: str(p)))


def _get_lid_axis_and_anchor(lid_obj):
    if lid_obj is None:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r = Rot.from_euler("xyz", np.asarray(lid_obj.rotation, dtype=np.float64), degrees=False).as_matrix()
    axis = r @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-10:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        axis = axis / n
    anchor = np.asarray(lid_obj.position, dtype=np.float64)
    return axis, anchor


def _quatf_rotate_x_to(axis_vec):
    v = np.asarray(axis_vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    v = v / n
    rot = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])))
    qd = rot.GetQuat()
    qi = qd.GetImaginary()
    return Gf.Quatf(float(qd.GetReal()), float(qi[0]), float(qi[1]), float(qi[2]))


def _create_twist_off_joint(
    stage,
    joint_path,
    parent_link_path,
    lid_path,
    joint_anchor,
    joint_axis,
    lower_limit_deg,
    upper_limit_deg,
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(lid_path)])

    joint.CreateAxisAttr().Set("X")
    joint_rot = _quatf_rotate_x_to(joint_axis)
    joint.CreateLocalPos0Attr().Set(_to_vec3f(joint_anchor))
    joint.CreateLocalPos1Attr().Set(_to_vec3f(joint_anchor))
    joint.CreateLocalRot0Attr().Set(joint_rot)
    joint.CreateLocalRot1Attr().Set(joint_rot)

    joint.CreateLowerLimitAttr().Set(float(lower_limit_deg))
    joint.CreateUpperLimitAttr().Set(float(upper_limit_deg))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def _create_prismatic_joint(
    stage,
    joint_path,
    parent_link_path,
    child_link_path,
    joint_anchor,
    joint_axis,
    lower_limit_m,
    upper_limit_m,
):
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_link_path)])

    joint.CreateAxisAttr().Set("X")
    joint_rot = _quatf_rotate_x_to(joint_axis)
    joint.CreateLocalPos0Attr().Set(_to_vec3f(joint_anchor))
    joint.CreateLocalPos1Attr().Set(_to_vec3f(joint_anchor))
    joint.CreateLocalRot0Attr().Set(joint_rot)
    joint.CreateLocalRot1Attr().Set(joint_rot)

    joint.CreateLowerLimitAttr().Set(float(lower_limit_m))
    joint.CreateUpperLimitAttr().Set(float(upper_limit_m))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def _create_fixed_joint(stage, joint_path, child_link_path):
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_link_path)])
    joint.CreateLocalPos0Attr().Set(_to_vec3f((0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr().Set(_to_vec3f((0.0, 0.0, 0.0)))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def export_with_twist_off_lid(
    visual_tex_path,
    body_obj_path,
    prismatic_obj_path,
    lid_obj_path,
    concept_pkl_path,
    output_usda_path,
    source_visual_usda_path=None,
    scale_to_meters=0.01,
    body_mass_kg=0.518496,
    prismatic_mass_kg=1.0e-6,
    lid_mass_kg=0.011375,
    static_friction=1.2,
    dynamic_friction=1.0,
    restitution=0.0,
    lid_lift_lower_limit_m=0.0,
    lid_lift_upper_limit_m=0.15,
    lid_joint_lower_limit_deg=-360.0,
    lid_joint_upper_limit_deg=360.0,
    init_pos=(0.0, 0.2, 0.0),
    init_euler=(90.0, 0.0, 0.0),
):
    visual_tex_path = os.path.abspath(visual_tex_path)
    body_obj_path = os.path.abspath(body_obj_path)
    prismatic_obj_path = os.path.abspath(prismatic_obj_path)
    lid_obj_path = os.path.abspath(lid_obj_path)
    concept_pkl_path = os.path.abspath(concept_pkl_path)
    output_usda_path = os.path.abspath(output_usda_path)
    if source_visual_usda_path is not None:
        source_visual_usda_path = os.path.abspath(source_visual_usda_path)

    for p in [visual_tex_path, body_obj_path, prismatic_obj_path, lid_obj_path, concept_pkl_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if source_visual_usda_path is not None and not os.path.exists(source_visual_usda_path):
        raise FileNotFoundError(source_visual_usda_path)

    os.makedirs(os.path.dirname(output_usda_path), exist_ok=True)
    stage = Usd.Stage.CreateNew(output_usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    bottle_root_path = "/Bottle"
    root = UsdGeom.Xform.Define(stage, bottle_root_path)
    stage.SetDefaultPrim(root.GetPrim())
    _set_initial_pose(root, init_pos, init_euler)
    _apply_articulation_root(root.GetPrim())

    concept_data = _load_concept_from_pkl(concept_pkl_path)
    root.GetPrim().CreateAttribute("dataset:id", Sdf.ValueTypeNames.String).Set(str(concept_data.get("id", "unknown")))

    links_path = f"{bottle_root_path}/links"
    joints_path = f"{bottle_root_path}/joints"
    UsdGeom.Xform.Define(stage, links_path)
    UsdGeom.Xform.Define(stage, joints_path)

    body_link_path = f"{links_path}/body_link"
    prismatic_link_path = f"{links_path}/lid_prismatic_link"
    lid_link_path = f"{links_path}/lid_link"
    body_link = _create_link_xform(stage, body_link_path, mass_kg=body_mass_kg)
    _ = body_link
    _create_link_xform(stage, prismatic_link_path, mass_kg=prismatic_mass_kg)
    _create_link_xform(stage, lid_link_path, mass_kg=lid_mass_kg)

    # Real appearance (high fidelity visual). Prefer the simple-collision USDA,
    # which already has the correct global Bottle UV layout.
    copied_visuals = False
    if source_visual_usda_path is not None:
        source_stage = Usd.Stage.Open(source_visual_usda_path)
        if source_stage is None:
            raise RuntimeError(f"cannot open source visual USDA: {source_visual_usda_path}")
        source_body = UsdGeom.Mesh(source_stage.GetPrimAtPath("/Bottle/body/visual"))
        source_lid = UsdGeom.Mesh(source_stage.GetPrimAtPath("/Bottle/lid/visual"))
        if source_body and source_lid:
            _copy_visual_mesh(stage, f"{body_link_path}/visual", source_body)
            _copy_visual_mesh(stage, f"{lid_link_path}/visual", source_lid)
            copied_visuals = True

    if not copied_visuals:
        body_verts, body_faces, body_uvs, body_face_uvs = _load_obj_vertices_faces_uv(body_obj_path)
        lid_verts, lid_faces, lid_uvs, lid_face_uvs = _load_obj_vertices_faces_uv(lid_obj_path)
        _add_visual_mesh(stage, f"{body_link_path}/visual", body_verts, body_faces, scale_to_meters, body_uvs, body_face_uvs)
        _add_visual_mesh(stage, f"{lid_link_path}/visual", lid_verts, lid_faces, scale_to_meters, lid_uvs, lid_face_uvs)

    _bind_preview_texture(stage, stage.GetPrimAtPath(f"{body_link_path}/visual"), visual_tex_path, "BottleBodyMat")
    _bind_preview_texture(stage, stage.GetPrimAtPath(f"{lid_link_path}/visual"), visual_tex_path, "BottleLidMat")

    # Simple collision from conceptualization
    simple_body_v, simple_body_f, simple_lid_v, simple_lid_f, lid_obj = _build_simple_collision_from_concept(concept_data)
    physics_mat = _create_physics_material(
        stage,
        f"{bottle_root_path}/Looks/HighFrictionPhysics",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    collision_prim_paths = []
    if simple_body_v is not None and simple_body_f is not None:
        body_collision_path = f"{body_link_path}/collision"
        body_col = _add_collision_mesh(stage, body_collision_path, simple_body_v, simple_body_f, scale_to_meters)
        _bind_physics_material(body_col.GetPrim(), physics_mat)
        collision_prim_paths.append(body_collision_path)
    if simple_lid_v is not None and simple_lid_f is not None:
        lid_collision_path = f"{lid_link_path}/collision"
        lid_col = _add_collision_mesh(stage, lid_collision_path, simple_lid_v, simple_lid_f, scale_to_meters)
        _bind_physics_material(lid_col.GetPrim(), physics_mat)
        collision_prim_paths.append(lid_collision_path)

    lid_axis_local, lid_anchor_local = _get_lid_axis_and_anchor(lid_obj)
    lid_anchor_local_scaled = lid_anchor_local * float(scale_to_meters)

    _create_fixed_joint(stage, f"{joints_path}/world_to_body_joint", body_link_path)
    _create_prismatic_joint(
        stage=stage,
        joint_path=f"{joints_path}/lid_lift_joint",
        parent_link_path=body_link_path,
        child_link_path=prismatic_link_path,
        joint_anchor=lid_anchor_local_scaled,
        joint_axis=lid_axis_local,
        lower_limit_m=lid_lift_lower_limit_m,
        upper_limit_m=lid_lift_upper_limit_m,
    )
    _create_twist_off_joint(
        stage=stage,
        joint_path=f"{joints_path}/lid_twist_joint",
        parent_link_path=prismatic_link_path,
        lid_path=lid_link_path,
        joint_anchor=lid_anchor_local_scaled,
        joint_axis=lid_axis_local,
        lower_limit_deg=lid_joint_lower_limit_deg,
        upper_limit_deg=lid_joint_upper_limit_deg,
    )

    _disable_collision_between_prims(stage, collision_prim_paths)
    _export_grasps(stage, bottle_root_path, lid_obj, scale_to_meters)

    stage.GetRootLayer().Save()
    return output_usda_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(repo_root, "real_object_data", "Bottle")

    visual_tex = _pick_single_file(
        data_dir,
        candidates=["textures/texture.jpg", "sim_ready/meshes/texture.jpg"],
        pattern="**/texture.*",
    )
    body_obj = _pick_single_file(
        data_dir,
        candidates=["segmentation/Multilevel_Body.obj"],
        pattern="**/Multilevel_Body.obj",
    )
    prismatic_obj = _pick_single_file(
        data_dir,
        candidates=["segmentation/Cylindrical_Lid_virtual_prismatic.obj"],
        pattern="**/Cylindrical_Lid_virtual_prismatic.obj",
    )
    lid_obj = _pick_single_file(
        data_dir,
        candidates=["segmentation/Cylindrical_Lid.obj"],
        pattern="**/Cylindrical_Lid.obj",
    )
    concept_pkl = _pick_single_file(
        data_dir,
        candidates=["conceptualization/lemon_tea.pkl"],
        pattern="**/conceptualization/*.pkl",
    )
    source_visual_usda = _pick_single_file(
        data_dir,
        candidates=["usda_output/lemon_tea_simple_collision.usda"],
        pattern="**/*simple_collision.usda",
    )
    output_name = f"{Path(concept_pkl).stem}_twist_off_lid.usda"
    output_usda = os.path.join(data_dir, "usda_output", output_name)

    out = export_with_twist_off_lid(
        visual_tex_path=visual_tex,
        body_obj_path=body_obj,
        prismatic_obj_path=prismatic_obj,
        lid_obj_path=lid_obj,
        concept_pkl_path=concept_pkl,
        output_usda_path=output_usda,
        source_visual_usda_path=source_visual_usda,
        scale_to_meters=0.01,
        body_mass_kg=0.518496,
        prismatic_mass_kg=1.0e-6,
        lid_mass_kg=0.011375,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
        lid_lift_lower_limit_m=0.0,
        lid_lift_upper_limit_m=0.15,
        lid_joint_lower_limit_deg=-360.0,
        lid_joint_upper_limit_deg=360.0,
        init_pos=[0.0, 0.2, 0.0],
        init_euler=[90.0, 0.0, 0.0],
    )
    print(f"[OK] exported: {out}")


if __name__ == "__main__":
    main()
