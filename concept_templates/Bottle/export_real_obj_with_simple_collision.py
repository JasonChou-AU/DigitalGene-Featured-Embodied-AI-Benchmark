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


def _add_collision_mesh(stage, prim_path, verts, faces, scale):
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


def export_with_simple_collision(
    visual_tex_path,
    body_obj_path,
    lid_obj_path,
    concept_pkl_path,
    output_usda_path,
    scale_to_meters=0.01,
    mass_kg=0.5,
    static_friction=1.2,
    dynamic_friction=1.0,
    restitution=0.0,
    init_pos=(0.0, 0.2, 0.0),
    init_euler=(90.0, 0.0, 0.0),
):
    visual_tex_path = os.path.abspath(visual_tex_path)
    body_obj_path = os.path.abspath(body_obj_path)
    lid_obj_path = os.path.abspath(lid_obj_path)
    concept_pkl_path = os.path.abspath(concept_pkl_path)
    output_usda_path = os.path.abspath(output_usda_path)

    for p in [visual_tex_path, body_obj_path, lid_obj_path, concept_pkl_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(os.path.dirname(output_usda_path), exist_ok=True)
    stage = Usd.Stage.CreateNew(output_usda_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    bottle_root_path = "/Bottle"
    root = UsdGeom.Xform.Define(stage, bottle_root_path)
    stage.SetDefaultPrim(root.GetPrim())
    _set_initial_pose(root, init_pos, init_euler)

    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)
    root_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(mass_kg))

    # Real appearance (high fidelity visual)
    body_verts, body_faces, body_uvs, body_face_uvs = _load_obj_vertices_faces_uv(body_obj_path)
    lid_verts, lid_faces, lid_uvs, lid_face_uvs = _load_obj_vertices_faces_uv(lid_obj_path)

    _add_visual_mesh(stage, f"{bottle_root_path}/body/visual", body_verts, body_faces, scale_to_meters, body_uvs, body_face_uvs)
    _bind_preview_texture(stage, stage.GetPrimAtPath(f"{bottle_root_path}/body/visual"), visual_tex_path, "BottleBodyMat")

    _add_visual_mesh(stage, f"{bottle_root_path}/lid/visual", lid_verts, lid_faces, scale_to_meters, lid_uvs, lid_face_uvs)
    _bind_preview_texture(stage, stage.GetPrimAtPath(f"{bottle_root_path}/lid/visual"), visual_tex_path, "BottleLidMat")

    # Simple collision from conceptualization
    concept_data = _load_concept_from_pkl(concept_pkl_path)
    simple_body_v, simple_body_f, simple_lid_v, simple_lid_f, lid_obj = _build_simple_collision_from_concept(concept_data)
    physics_mat = _create_physics_material(
        stage,
        f"{bottle_root_path}/Looks/HighFrictionPhysics",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    if simple_body_v is not None and simple_body_f is not None:
        body_col = _add_collision_mesh(stage, f"{bottle_root_path}/body/collision", simple_body_v, simple_body_f, scale_to_meters)
        _bind_physics_material(body_col.GetPrim(), physics_mat)
    if simple_lid_v is not None and simple_lid_f is not None:
        lid_col = _add_collision_mesh(stage, f"{bottle_root_path}/lid/collision", simple_lid_v, simple_lid_f, scale_to_meters)
        _bind_physics_material(lid_col.GetPrim(), physics_mat)

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
        candidates=["segmentation/body.obj"],
        pattern="**/body.obj",
    )
    lid_obj = _pick_single_file(
        data_dir,
        candidates=["segmentation/lid.obj"],
        pattern="**/lid.obj",
    )
    concept_pkl = _pick_single_file(
        data_dir,
        candidates=["conceptualization/lemon_tea.pkl"],
        pattern="**/conceptualization/*.pkl",
    )
    output_name = f"{Path(concept_pkl).stem}_simple_collision.usda"
    output_usda = os.path.join(data_dir, "usda_output", output_name)

    out = export_with_simple_collision(
        visual_tex_path=visual_tex,
        body_obj_path=body_obj,
        lid_obj_path=lid_obj,
        concept_pkl_path=concept_pkl,
        output_usda_path=output_usda,
        scale_to_meters=0.01,
        mass_kg=0.1,
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
        init_pos=[0.0, 0.2, 0.0],
        init_euler=[90.0, 0.0, 0.0],
    )
    print(f"[OK] exported: {out}")


if __name__ == "__main__":
    main()
