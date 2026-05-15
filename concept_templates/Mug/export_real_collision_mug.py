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


def _add_collision_mesh(stage, prim_path, verts, faces, scale, approximation="convexDecomposition"):
    geom = UsdGeom.Mesh.Define(stage, prim_path)
    verts = np.asarray(verts, dtype=np.float32) * float(scale)
    faces = np.asarray(faces, dtype=np.int32)

    geom.CreatePointsAttr([_to_vec3f(v) for v in verts])
    geom.CreateFaceVertexCountsAttr([3] * len(faces))
    geom.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

    prim = geom.GetPrim()
    UsdGeom.Imageable(geom).CreatePurposeAttr().Set("physics")
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approximation)
    return geom


def _create_physics_material(stage, mat_path, static_friction=2.0, dynamic_friction=2.0, restitution=0.0):
    mat = UsdShade.Material.Define(stage, mat_path)
    mat_prim = mat.GetPrim()
    mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(restitution))
    return mat


def _bind_physics_material(collision_prim, physics_material):
    UsdShade.MaterialBindingAPI(collision_prim).Bind(
        physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _bind_preview_texture(stage, mesh_prim, texture_path, mat_name):
    looks_scope = UsdGeom.Scope.Define(stage, "/Mug/Looks")
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


def _extract_handle_objs(concept_data):
    handle_objs = []
    for c in concept_data["conceptualization"]:
        template_name = c.get("template")
        if template_name is None:
            continue
        obj = eval(template_name)(**c["parameters"])
        if isinstance(obj, (Trifold_Handle, Curved_Handle)):
            handle_objs.append(obj)
    return handle_objs


def _export_grasps(stage, mug_root_path, handle_objs, scale_to_meters):
    if len(handle_objs) == 0:
        return 0

    grasp_root_path = f"{mug_root_path}/grasps"
    UsdGeom.Xform.Define(stage, grasp_root_path)
    test_params = [(-3, -1, 0), (-3, 1, 0), (0, -1, 0), (0, 1, 0)]

    count = 0
    for h_idx, h_obj in enumerate(handle_objs):
        for p_idx, (p1, p2, p3) in enumerate(test_params):
            spec = get_grasp_spec(h_obj, manipulation_params=(p1, p2, p3))
            if spec is None:
                continue
            if "world_position" not in spec or "world_rotation" not in spec:
                continue

            g_xform = UsdGeom.Xform.Define(stage, f"{grasp_root_path}/grasp_{p_idx}")
            t = np.asarray(spec["world_position"], dtype=np.float64) * float(scale_to_meters)
            q = np.asarray(spec["world_rotation"], dtype=np.float64)  # [x, y, z, w]
            quat_gf = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))

            g_xform.AddTranslateOp().Set(_to_vec3d(t))
            g_xform.AddOrientOp().Set(quat_gf)

            prim = g_xform.GetPrim()
            approach = np.asarray(spec.get("world_approach_direction", [0.0, 0.0, 1.0]), dtype=np.float64)
            prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(_to_vec3f(approach))

            t_mat = np.asarray(spec.get("world_transformation_matrix", np.eye(4)), dtype=np.float64).copy()
            if t_mat.shape == (4, 4):
                t_mat[:3, 3] *= float(scale_to_meters)
                prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(t_mat.flatten().tolist())
            count += 1

    return count


def export_real_collision_mug(
    visual_tex_path,
    mug_obj_path,
    concept_pkl_path,
    output_usda_path,
    scale_to_meters=0.01,
    mass_kg=0.05,
    static_friction=2.0,
    dynamic_friction=2.0,
    restitution=0.0,
    linear_damping=3000.0,
    angular_damping=3000.0,
    init_pos=(0.0, 0.2, 0.0),
    init_euler=(90.0, 0.0, 0.0),
):
    visual_tex_path = os.path.abspath(visual_tex_path)
    mug_obj_path = os.path.abspath(mug_obj_path)
    concept_pkl_path = os.path.abspath(concept_pkl_path)
    output_usda_path = os.path.abspath(output_usda_path)

    for p in [visual_tex_path, mug_obj_path, concept_pkl_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(os.path.dirname(output_usda_path), exist_ok=True)
    stage = Usd.Stage.CreateNew(output_usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    mug_root_path = "/Mug"
    root = UsdGeom.Xform.Define(stage, mug_root_path)
    stage.SetDefaultPrim(root.GetPrim())
    _set_initial_pose(root, init_pos, init_euler)

    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)
    root_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass_kg))
    if linear_damping is not None:
        root_prim.CreateAttribute("physics:linearDamping", Sdf.ValueTypeNames.Float).Set(float(linear_damping))
    if angular_damping is not None:
        root_prim.CreateAttribute("physics:angularDamping", Sdf.ValueTypeNames.Float).Set(float(angular_damping))

    mug_verts, mug_faces, mug_uvs, mug_face_uvs = _load_obj_vertices_faces_uv(mug_obj_path)
    _add_visual_mesh(stage, f"{mug_root_path}/visual", mug_verts, mug_faces, scale_to_meters, mug_uvs, mug_face_uvs)
    _bind_preview_texture(stage, stage.GetPrimAtPath(f"{mug_root_path}/visual"), visual_tex_path, "MugMat")

    physics_mat = _create_physics_material(
        stage,
        f"{mug_root_path}/Looks/HighFrictionPhysics",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )
    collision = _add_collision_mesh(stage, f"{mug_root_path}/collision", mug_verts, mug_faces, scale_to_meters)
    _bind_physics_material(collision.GetPrim(), physics_mat)

    concept_data = _load_concept_from_pkl(concept_pkl_path)
    _export_grasps(stage, mug_root_path, _extract_handle_objs(concept_data), scale_to_meters)

    stage.GetRootLayer().Save()
    return output_usda_path


def export_with_real_collision(*args, **kwargs):
    return export_real_collision_mug(*args, **kwargs)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(repo_root, "real_object_data", "Mug")

    visual_tex = _pick_single_file(
        data_dir,
        candidates=["textures/texture.jpg"],
        pattern="**/texture.*",
    )
    mug_obj = _pick_single_file(
        data_dir,
        candidates=["segmentation/Mug.obj"],
        pattern="**/*.obj",
    )
    concept_pkl = _pick_single_file(
        data_dir,
        candidates=["conceptualization/Mug_conceptualization.pkl"],
        pattern="**/conceptualization/*.pkl",
    )
    output_name = f"{Path(concept_pkl).stem}_real_collision.usda"
    output_usda = os.path.join(data_dir, "usda_output", output_name)

    out = export_real_collision_mug(
        visual_tex_path=visual_tex,
        mug_obj_path=mug_obj,
        concept_pkl_path=concept_pkl,
        output_usda_path=output_usda,
        scale_to_meters=0.01,
        mass_kg=0.05,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
        linear_damping=3000.0,
        angular_damping=3000.0,
        init_pos=[0.0, 0.2, 0.0],
        init_euler=[90.0, 0.0, 0.0],
    )
    print(f"[OK] exported: {out}")


if __name__ == "__main__":
    main()
