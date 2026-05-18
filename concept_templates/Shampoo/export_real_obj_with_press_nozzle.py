import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from concept_template import *
from knowledge_definitions import get_grasp_spec
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as Rot

try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None


ROOT_PATH = "/Shampoo"
REAL_OBJ_MESHES = {
    "Cylindrical_body": "Cylindrical_Body.obj",
    "Regular_nozzle": "Regular_nozzle_0.obj",
    "Regular_nozzle_Head": "Regular_nozzle_1.obj",
}


def _to_vec3f(v):
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))


def _to_vec3d(v):
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def _to_quatf_xyzw(q_xyzw):
    return Gf.Quatf(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _set_initial_pose(prim_xform, position, euler_deg):
    prim_xform.ClearXformOpOrder()
    prim_xform.AddTranslateOp().Set(_to_vec3d(position))
    q = Rot.from_euler("xyz", euler_deg, degrees=True).as_quat()
    prim_xform.AddOrientOp().Set(_to_quatf_xyzw(q))
    prim_xform.AddScaleOp().Set(_to_vec3f((1.0, 1.0, 1.0)))


def _set_identity_xform(prim_xform):
    prim_xform.ClearXformOpOrder()
    prim_xform.AddTranslateOp().Set(_to_vec3d((0.0, 0.0, 0.0)))
    prim_xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim_xform.AddScaleOp().Set(_to_vec3f((1.0, 1.0, 1.0)))


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
                idx = []
                uv_idx = []
                for p in line.split()[1:]:
                    toks = p.split("/")
                    if not toks[0]:
                        continue
                    vi = int(toks[0])
                    vi = len(verts) + vi if vi < 0 else vi - 1
                    idx.append(vi)
                    if len(toks) > 1 and toks[1]:
                        ti = int(toks[1])
                        ti = len(uvs) + ti if ti < 0 else ti - 1
                        uv_idx.append(ti)
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


def _create_mesh(
    stage,
    mesh_path,
    verts,
    faces,
    scale,
    collision=False,
    approximation="convexDecomposition",
    uvs=None,
    face_uvs=None,
    physics_material=None,
    contact_offset=0.006,
    rest_offset=0.0,
):
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    verts = np.asarray(verts, dtype=np.float32) * float(scale)
    faces = np.asarray(faces, dtype=np.int32)

    mesh.CreatePointsAttr([_to_vec3f(v) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateExtentAttr([
        _to_vec3f(np.min(verts, axis=0)),
        _to_vec3f(np.max(verts, axis=0)),
    ])

    if uvs is not None and face_uvs is not None and len(uvs) > 0 and len(face_uvs) == len(faces):
        st_data = []
        for tri_uv in face_uvs:
            for uv_i in tri_uv:
                if 0 <= uv_i < len(uvs):
                    st_data.append(Gf.Vec2f(float(uvs[uv_i][0]), float(uvs[uv_i][1])))
                else:
                    st_data.append(Gf.Vec2f(0.0, 0.0))
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st_primvar = primvars_api.CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st_primvar.Set(st_data)

    if collision:
        prim = mesh.GetPrim()
        UsdGeom.Imageable(mesh).CreatePurposeAttr().Set("physics")
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approximation)
        _bind_physics_material(prim, physics_material)
        if PhysxSchema is not None:
            try:
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr().Set(float(contact_offset))
                physx_collision.CreateRestOffsetAttr().Set(float(rest_offset))
            except Exception:
                prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float(contact_offset))
                prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(float(rest_offset))
        else:
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float(contact_offset))
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(float(rest_offset))

    return mesh


def _create_preview_texture_material(stage, material_path, texture_path):
    mat = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/TexCoordReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _create_physics_material(stage, material_path, static_friction, dynamic_friction, restitution):
    mat = UsdShade.Material.Define(stage, material_path)
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(restitution))
    return mat


def _bind_visual_material(mesh_prim, material):
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)


def _bind_physics_material(collision_prim, physics_material):
    if physics_material is None:
        return
    try:
        UsdShade.MaterialBindingAPI(collision_prim).Bind(
            physics_material,
            materialPurpose=UsdShade.Tokens.physics,
        )
    except Exception:
        UsdShade.MaterialBindingAPI(collision_prim).Bind(physics_material)


def _apply_rigid_body(prim, mass_kg, linear_damping=None, angular_damping=None):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass_kg))
    if linear_damping is not None:
        prim.CreateAttribute("physics:linearDamping", Sdf.ValueTypeNames.Float).Set(float(linear_damping))
    if angular_damping is not None:
        prim.CreateAttribute("physics:angularDamping", Sdf.ValueTypeNames.Float).Set(float(angular_damping))


def _create_link(
    stage,
    link_path,
    mass_kg,
    link_name,
    translate=(0.0, 0.0, 0.0),
    linear_damping=0.04,
    angular_damping=0.04,
):
    link = UsdGeom.Xform.Define(stage, link_path)
    _set_initial_pose(link, translate, (0.0, 0.0, 0.0))
    _apply_rigid_body(
        link.GetPrim(),
        mass_kg=mass_kg,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
    )
    link.GetPrim().CreateAttribute("urdf:linkName", Sdf.ValueTypeNames.String).Set(link_name)
    return link


def _axis_token(axis):
    axis = np.asarray(axis, dtype=np.float64)
    if np.linalg.norm(axis) < 1.0e-10:
        return "X"
    idx = int(np.argmax(np.abs(axis)))
    return ("X", "Y", "Z")[idx]


def _quatf_from_rpy(rpy_rad):
    q = Rot.from_euler("xyz", np.asarray(rpy_rad, dtype=np.float64), degrees=False).as_quat()
    return _to_quatf_xyzw(q)


def _parse_urdf_joints(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint_el in root.findall("joint"):
        origin_el = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        limit_el = joint_el.find("limit")
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        axis = [1.0, 0.0, 0.0]
        lower = None
        upper = None
        effort = None
        velocity = None

        if origin_el is not None:
            xyz = [float(x) for x in origin_el.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin_el.get("rpy", "0 0 0").split()]
        if axis_el is not None:
            axis = [float(x) for x in axis_el.get("xyz", "1 0 0").split()]
        if limit_el is not None:
            lower = float(limit_el.get("lower", "0"))
            upper = float(limit_el.get("upper", "0"))
            effort = float(limit_el.get("effort", "0"))
            velocity = float(limit_el.get("velocity", "0"))

        joints.append(
            {
                "name": joint_el.get("name"),
                "type": joint_el.get("type"),
                "parent": parent_el.get("link") if parent_el is not None else None,
                "child": child_el.get("link") if child_el is not None else None,
                "xyz": np.asarray(xyz, dtype=np.float64),
                "rpy": np.asarray(rpy, dtype=np.float64),
                "axis": np.asarray(axis, dtype=np.float64),
                "lower": lower,
                "upper": upper,
                "effort": effort,
                "velocity": velocity,
            }
        )
    return joints


def _create_fixed_joint(
    stage,
    joint_path,
    parent_path,
    child_path,
    spec=None,
    distance_scale=1.0,
    local_pos0=None,
    local_pos1=None,
):
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    if parent_path is not None:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    if local_pos0 is not None:
        joint.CreateLocalPos0Attr().Set(_to_vec3f(local_pos0))
        joint.CreateLocalRot0Attr().Set(_quatf_from_rpy(spec["rpy"]) if spec is not None else Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    elif spec is None:
        joint.CreateLocalPos0Attr().Set(_to_vec3f((0.0, 0.0, 0.0)))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    else:
        joint.CreateLocalPos0Attr().Set(_to_vec3f(spec["xyz"] * float(distance_scale)))
        joint.CreateLocalRot0Attr().Set(_quatf_from_rpy(spec["rpy"]))
    joint.CreateLocalPos1Attr().Set(_to_vec3f(local_pos1 if local_pos1 is not None else (0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def _create_prismatic_joint(
    stage,
    joint_path,
    parent_path,
    child_path,
    spec,
    distance_scale,
    limit_scale,
    drive_damping,
    local_pos0=None,
    local_pos1=None,
):
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    joint.CreateAxisAttr().Set(_axis_token(spec["axis"]))
    joint.CreateLocalPos0Attr().Set(_to_vec3f(local_pos0 if local_pos0 is not None else spec["xyz"] * float(distance_scale)))
    joint.CreateLocalPos1Attr().Set(_to_vec3f(local_pos1 if local_pos1 is not None else (0.0, 0.0, 0.0)))
    joint.CreateLocalRot0Attr().Set(_quatf_from_rpy(spec["rpy"]))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if spec["lower"] is not None and spec["upper"] is not None:
        joint.CreateLowerLimitAttr().Set(float(spec["lower"]) * float(limit_scale))
        joint.CreateUpperLimitAttr().Set(float(spec["upper"]) * float(limit_scale))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(float(drive_damping))
    drive.CreateMaxForceAttr().Set(float(spec["effort"] or 10.0))
    if spec["velocity"] is not None:
        joint.GetPrim().CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(float(spec["velocity"]))
    return joint


def _create_revolute_joint(
    stage,
    joint_path,
    parent_path,
    child_path,
    spec,
    distance_scale,
    drive_damping,
    local_pos0=None,
    local_pos1=None,
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    joint.CreateAxisAttr().Set(_axis_token(spec["axis"]))
    joint.CreateLocalPos0Attr().Set(_to_vec3f(local_pos0 if local_pos0 is not None else spec["xyz"] * float(distance_scale)))
    joint.CreateLocalPos1Attr().Set(_to_vec3f(local_pos1 if local_pos1 is not None else (0.0, 0.0, 0.0)))
    joint.CreateLocalRot0Attr().Set(_quatf_from_rpy(spec["rpy"]))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if spec["lower"] is not None and spec["upper"] is not None:
        joint.CreateLowerLimitAttr().Set(float(np.degrees(spec["lower"])))
        joint.CreateUpperLimitAttr().Set(float(np.degrees(spec["upper"])))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(float(drive_damping))
    drive.CreateMaxForceAttr().Set(float(spec["effort"] or 10.0))
    if spec["velocity"] is not None:
        joint.GetPrim().CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(
            float(np.degrees(spec["velocity"]))
        )
    return joint


def _disable_collision_between_prims(stage, prim_paths):
    paths = [Sdf.Path(p) for p in prim_paths if stage.GetPrimAtPath(p).IsValid()]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        rel = prim.GetRelationship("physics:filteredPairs")
        if not rel.IsValid():
            rel = prim.CreateRelationship("physics:filteredPairs", False)
        rel.SetTargets([p for p in paths if p != path])


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


def _export_grasps(stage, root_path, concept_pkl_path, concept_scale=1.0):
    if concept_pkl_path is None or not os.path.exists(concept_pkl_path):
        return 0

    data = _load_concept_data(concept_pkl_path)
    stage.GetPrimAtPath(root_path).CreateAttribute("dataset:id", Sdf.ValueTypeNames.String).Set(
        str(data.get("id", "unknown"))
    )
    grasp_root = UsdGeom.Xform.Define(stage, f"{root_path}/grasps")
    _set_identity_xform(grasp_root)

    count = 0
    for entry in data["conceptualization"]:
        template_name = entry.get("template")
        if template_name is None:
            continue
        obj = eval(template_name)(**entry["parameters"])
        spec = get_grasp_spec(obj)
        if spec is None:
            continue

        t_mat = np.asarray(spec["world_transformation_matrix"], dtype=np.float64).copy()
        t_mat[:3, 3] *= float(concept_scale)
        q = spec["world_rotation"]
        g_xform = UsdGeom.Xform.Define(stage, f"{root_path}/grasps/grasp_{count}")
        g_xform.ClearXformOpOrder()
        g_xform.AddTranslateOp().Set(_to_vec3d(t_mat[:3, 3]))
        g_xform.AddOrientOp().Set(_to_quatf_xyzw(q))

        prim = g_xform.GetPrim()
        prim.CreateAttribute("grasp:source_template", Sdf.ValueTypeNames.String).Set(template_name)
        prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(
            _to_vec3f(np.asarray(spec["world_approach_direction"], dtype=np.float64))
        )
        if "world_finger_closing_direction" in spec:
            prim.CreateAttribute("grasp:finger_closing", Sdf.ValueTypeNames.Vector3f).Set(
                _to_vec3f(np.asarray(spec["world_finger_closing_direction"], dtype=np.float64))
            )
        if "grasp_width" in spec:
            prim.CreateAttribute("grasp:width", Sdf.ValueTypeNames.Float).Set(float(spec["grasp_width"]) * float(concept_scale))
        prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(t_mat.flatten().tolist())
        count += 1
    return count


def _pick_file(base_dir, candidates, pattern):
    for rel in candidates:
        p = os.path.join(base_dir, rel)
        if os.path.exists(p):
            return os.path.abspath(p)
    matches = sorted(Path(base_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matched in {base_dir}: {pattern}")
    return str(matches[0].resolve())


def _collect_real_obj_paths(segmentation_dir):
    paths = {}
    for link_name, obj_name in REAL_OBJ_MESHES.items():
        obj_path = os.path.join(segmentation_dir, obj_name)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(obj_path)
        paths[link_name] = obj_path
    return paths


def _mesh_centroid(verts):
    return np.mean(np.asarray(verts, dtype=np.float64), axis=0)


def _build_link_origins_unscaled(mesh_data):
    origins = {
        "Cylindrical_body": _mesh_centroid(mesh_data["Cylindrical_body"]["verts"]),
        "Regular_nozzle": _mesh_centroid(mesh_data["Regular_nozzle"]["verts"]),
        "Regular_nozzle_Head": _mesh_centroid(mesh_data["Regular_nozzle_Head"]["verts"]),
    }
    origins["Regular_nozzle_virtual_prismatic"] = origins["Regular_nozzle"].copy()
    origins["Regular_nozzle_Head_virtual_prismatic"] = origins["Regular_nozzle_Head"].copy()
    return origins


def _joint_anchor_unscaled(spec, parent, child, link_origins_unscaled):
    child_origin = link_origins_unscaled.get(child)
    parent_origin = link_origins_unscaled.get(parent)
    if parent_origin is None:
        return child_origin.copy() if child_origin is not None else np.zeros((3,), dtype=np.float64)
    if np.linalg.norm(spec["xyz"]) < 1.0e-10 and child_origin is not None:
        return child_origin.copy()
    return parent_origin + np.asarray(spec["xyz"], dtype=np.float64)


def _joint_local_positions_scaled(spec, parent, child, link_origins_unscaled, scale_to_meters):
    anchor = _joint_anchor_unscaled(spec, parent, child, link_origins_unscaled)
    parent_origin = link_origins_unscaled.get(parent, np.zeros((3,), dtype=np.float64))
    child_origin = link_origins_unscaled.get(child, np.zeros((3,), dtype=np.float64))
    return (
        (anchor - parent_origin) * float(scale_to_meters),
        (anchor - child_origin) * float(scale_to_meters),
    )


def export_press_nozzle_shampoo_with_real_collision(
    urdf_path,
    segmentation_dir,
    texture_path,
    concept_pkl_path,
    output_usda_path,
    scale_to_meters=0.01,
    joint_distance_scale=None,
    prismatic_limit_scale=None,
    init_pos=(0.0, 0.0, 0.0),
    init_euler=(0.0, 0.0, 0.0),
    body_mass_kg=0.45,
    nozzle_mass_kg=0.035,
    head_mass_kg=0.025,
    virtual_link_mass_kg=1.0e-5,
    static_friction=1.1,
    dynamic_friction=0.9,
    restitution=0.0,
    joint_drive_damping=0.35,
    collision_approximation="convexDecomposition",
    contact_offset=0.006,
    rest_offset=0.0,
    anchor_body=True,
    export_grasps=True,
):
    urdf_path = os.path.abspath(urdf_path)
    segmentation_dir = os.path.abspath(segmentation_dir)
    texture_path = os.path.abspath(texture_path)
    concept_pkl_path = os.path.abspath(concept_pkl_path) if concept_pkl_path is not None else None
    output_usda_path = os.path.abspath(output_usda_path)
    joint_distance_scale = scale_to_meters if joint_distance_scale is None else joint_distance_scale
    prismatic_limit_scale = 1.0 if prismatic_limit_scale is None else prismatic_limit_scale

    for path in [urdf_path, segmentation_dir, texture_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if concept_pkl_path is not None and not os.path.exists(concept_pkl_path):
        raise FileNotFoundError(concept_pkl_path)

    real_obj_paths = _collect_real_obj_paths(segmentation_dir)
    mesh_data = {}
    for link_name, obj_path in real_obj_paths.items():
        verts, faces, uvs, face_uvs = _load_obj_vertices_faces_uv(obj_path)
        mesh_data[link_name] = {
            "path": obj_path,
            "verts": verts,
            "faces": faces,
            "uvs": uvs,
            "face_uvs": face_uvs,
        }
    link_origins_unscaled = _build_link_origins_unscaled(mesh_data)

    os.makedirs(os.path.dirname(output_usda_path), exist_ok=True)
    stage = Usd.Stage.CreateNew(output_usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT_PATH)
    stage.SetDefaultPrim(root.GetPrim())
    _set_initial_pose(root, init_pos, init_euler)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute("asset:sourceUrdf", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(urdf_path))
    root.GetPrim().CreateAttribute("asset:scaleToMeters", Sdf.ValueTypeNames.Float).Set(float(scale_to_meters))
    root.GetPrim().CreateAttribute("asset:prismaticLimitScale", Sdf.ValueTypeNames.Float).Set(float(prismatic_limit_scale))
    root.GetPrim().CreateAttribute("asset:collisionSource", Sdf.ValueTypeNames.String).Set("segmentation_obj")

    UsdGeom.Xform.Define(stage, f"{ROOT_PATH}/links")
    UsdGeom.Xform.Define(stage, f"{ROOT_PATH}/joints")
    looks_scope = UsdGeom.Scope.Define(stage, f"{ROOT_PATH}/Looks")

    visual_mat = _create_preview_texture_material(stage, f"{looks_scope.GetPath()}/ShampooTexture", texture_path)
    physics_mat = _create_physics_material(
        stage,
        f"{looks_scope.GetPath()}/HighFrictionPhysics",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    masses = {
        "Cylindrical_body": body_mass_kg,
        "Regular_nozzle_virtual_prismatic": virtual_link_mass_kg,
        "Regular_nozzle": nozzle_mass_kg,
        "Regular_nozzle_Head_virtual_prismatic": virtual_link_mass_kg,
        "Regular_nozzle_Head": head_mass_kg,
    }
    link_paths = {}
    for link_name, mass in masses.items():
        link_path = f"{ROOT_PATH}/links/{link_name}"
        link_paths[link_name] = link_path
        _create_link(
            stage,
            link_path,
            mass_kg=mass,
            link_name=link_name,
            translate=link_origins_unscaled[link_name] * float(scale_to_meters),
        )

    collision_paths = []
    for link_name, data in mesh_data.items():
        obj_path = data["path"]
        local_verts = data["verts"] - link_origins_unscaled[link_name][None, :]
        visual = _create_mesh(
            stage,
            f"{link_paths[link_name]}/visual",
            local_verts,
            data["faces"],
            scale=scale_to_meters,
            collision=False,
            uvs=data["uvs"],
            face_uvs=data["face_uvs"],
        )
        _bind_visual_material(visual.GetPrim(), visual_mat)

        collision = _create_mesh(
            stage,
            f"{link_paths[link_name]}/collision",
            local_verts,
            data["faces"],
            scale=scale_to_meters,
            collision=True,
            approximation=collision_approximation,
            physics_material=physics_mat,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        )
        collision.GetPrim().CreateAttribute("asset:sourceObj", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(obj_path))
        collision_paths.append(str(collision.GetPath()))

    joints = _parse_urdf_joints(urdf_path)
    for spec in joints:
        child = spec["child"]
        parent = spec["parent"]
        if child not in link_paths:
            continue

        joint_path = f"{ROOT_PATH}/joints/{spec['name']}"
        child_path = link_paths[child]
        parent_path = link_paths.get(parent)
        joint_type = spec["type"]
        local_pos0, local_pos1 = _joint_local_positions_scaled(
            spec,
            parent,
            child,
            link_origins_unscaled,
            scale_to_meters,
        )
        if joint_type == "fixed":
            if parent == "world" and anchor_body:
                _create_fixed_joint(stage, joint_path, None, child_path, spec=None)
            else:
                _create_fixed_joint(
                    stage,
                    joint_path,
                    parent_path,
                    child_path,
                    spec=spec,
                    distance_scale=joint_distance_scale,
                    local_pos0=local_pos0,
                    local_pos1=local_pos1,
                )
        elif joint_type == "prismatic":
            _create_prismatic_joint(
                stage,
                joint_path,
                parent_path,
                child_path,
                spec,
                distance_scale=joint_distance_scale,
                limit_scale=prismatic_limit_scale,
                drive_damping=joint_drive_damping,
                local_pos0=local_pos0,
                local_pos1=local_pos1,
            )
        elif joint_type in ("revolute", "continuous"):
            _create_revolute_joint(
                stage,
                joint_path,
                parent_path,
                child_path,
                spec,
                distance_scale=joint_distance_scale,
                drive_damping=joint_drive_damping,
                local_pos0=local_pos0,
                local_pos1=local_pos1,
            )
        else:
            raise ValueError(f"unsupported joint type from URDF: {joint_type}")

        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_prim.CreateAttribute("urdf:jointName", Sdf.ValueTypeNames.String).Set(spec["name"])
        joint_prim.CreateAttribute("urdf:parent", Sdf.ValueTypeNames.String).Set(parent or "")
        joint_prim.CreateAttribute("urdf:child", Sdf.ValueTypeNames.String).Set(child or "")

    _disable_collision_between_prims(stage, collision_paths)
    grasp_count = _export_grasps(stage, ROOT_PATH, concept_pkl_path, concept_scale=1.0) if export_grasps else 0
    root.GetPrim().CreateAttribute("grasp:count", Sdf.ValueTypeNames.Int).Set(int(grasp_count))

    stage.GetRootLayer().Save()
    return output_usda_path


def export_press_nozzle_shampoo(*args, **kwargs):
    return export_press_nozzle_shampoo_with_real_collision(*args, **kwargs)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(repo_root, "real_object_data", "Shampoo")

    urdf_path = _pick_file(data_dir, ["sim_ready/Shampoo.urdf"], "**/Shampoo.urdf")
    segmentation_dir = os.path.join(data_dir, "segmentation")
    texture_path = _pick_file(
        data_dir,
        ["textures/texture.jpg", "sim_ready/configuration/materials/textures/material_0.jpeg"],
        "**/*texture*.*",
    )
    concept_pkl_path = _pick_file(
        data_dir,
        ["conceptualization/conceptualization.pkl"],
        "**/conceptualization/*.pkl",
    )
    output_path = os.path.join(data_dir, "usda_output", "shampoo_press_nozzle_real_collision.usda")

    out = export_press_nozzle_shampoo_with_real_collision(
        urdf_path=urdf_path,
        segmentation_dir=segmentation_dir,
        texture_path=texture_path,
        concept_pkl_path=concept_pkl_path,
        output_usda_path=output_path,
        scale_to_meters=0.001,
        prismatic_limit_scale=None,
        init_pos=[0.0, 0.0, 0.0],
        init_euler=[0.0, 0.0, 0.0],
        export_grasps=True,
    )
    print(f"[OK] exported: {out}")


if __name__ == "__main__":
    main()
