# This file is used to generate primitive collsions to test which is better: mesh collison or conbination of primitive collision 
import os
import pickle
import inspect
import copy
import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from scipy.spatial.transform import Rotation as Rot

try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None

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


def _create_visual_mesh(stage, mesh_path, verts, faces):
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([to_vec3f(v) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int32).flatten().tolist())
    return mesh


def _create_collision_mesh(
    stage,
    mesh_path,
    verts,
    faces,
    physics_material,
    contact_offset,
    rest_offset,
    approximation="convexHull",
):
    mesh = _create_visual_mesh(stage, mesh_path, verts, faces)
    prim = mesh.GetPrim()
    UsdGeom.Imageable(mesh).CreatePurposeAttr().Set("physics")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(str(approximation))
    _configure_collision_prim(
        prim,
        physics_material=physics_material,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
    )
    return mesh


def _set_xform_pose(xformable, translate, orient_xyzw=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0)):
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(to_vec3d(translate))
    xformable.AddOrientOp().Set(to_quatf_xyzw(orient_xyzw))
    xformable.AddScaleOp().Set(to_vec3f(scale))


def _ensure_physics_material(
    stage,
    material_path="/StorageFurniture/Looks/PhysicsMaterial",
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.0,
):
    mat = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    mat_prim = mat.GetPrim()
    mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(restitution))
    return mat


def _bind_physics_material(prim, physics_material):
    if physics_material is None:
        return
    try:
        rel = prim.CreateRelationship("physics:material:binding", False)
        rel.SetTargets([physics_material.GetPath()])
    except Exception:
        pass
    try:
        if hasattr(UsdShade.Tokens, "physics"):
            UsdShade.MaterialBindingAPI(prim).Bind(physics_material, materialPurpose=UsdShade.Tokens.physics)
        else:
            UsdShade.MaterialBindingAPI(prim).Bind(physics_material)
    except Exception:
        pass


def _configure_collision_prim(
    prim,
    physics_material=None,
    contact_offset=0.003,
    rest_offset=0.0,
):
    UsdPhysics.CollisionAPI.Apply(prim)
    _bind_physics_material(prim, physics_material)

    if PhysxSchema is not None:
        try:
            c_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            # Slightly larger contact offset improves pre-contact generation and avoids deep penetration.
            c_api.CreateContactOffsetAttr().Set(float(contact_offset))
            # Keep rest offset at 0 for neutral resting distance.
            c_api.CreateRestOffsetAttr().Set(float(rest_offset))
            # Stabilize frictional torque response at small patch contacts (handle pinches).
            c_api.CreateTorsionalPatchRadiusAttr().Set(0.0015)
            c_api.CreateMinTorsionalPatchRadiusAttr().Set(0.0008)
            return
        except Exception:
            pass

    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float(contact_offset))
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(float(rest_offset))


def _bbox_from_verts(verts, min_extent=1e-4):
    verts = np.asarray(verts, dtype=np.float32)
    if len(verts) == 0:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32) * float(min_extent)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = np.maximum(vmax - vmin, float(min_extent))
    return center.astype(np.float32), extent.astype(np.float32)


def _obb_from_verts(verts, min_extent=1e-4):
    verts = np.asarray(verts, dtype=np.float32)
    if len(verts) == 0:
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32) * float(min_extent)

    center = verts.mean(axis=0)
    shifted = verts - center[None, :]
    cov = np.matmul(shifted.T, shifted) / max(len(verts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].astype(np.float32)
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]

    projected = np.matmul(shifted, axes)
    pmin = projected.min(axis=0)
    pmax = projected.max(axis=0)
    extents = np.maximum(pmax - pmin, float(min_extent))
    center_local = 0.5 * (pmin + pmax)
    center_world = center + np.matmul(axes, center_local)
    return center_world.astype(np.float32), axes.astype(np.float32), extents.astype(np.float32)


def _safe_shrink_extent(extent, shrink):
    e = np.asarray(extent, dtype=np.float32).copy()
    e = np.maximum(e - 2.0 * float(shrink), 0.002)
    return e


def _quat_xyzw_from_axes(axes):
    q = Rot.from_matrix(np.asarray(axes, dtype=np.float64)).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _apply_shape_collision_defaults(
    prim,
    physics_material,
    contact_offset,
    rest_offset,
):
    _configure_collision_prim(
        prim,
        physics_material=physics_material,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
    )


def _create_box_collider(
    stage,
    path,
    center,
    axes,
    extents,
    physics_material,
    contact_offset,
    rest_offset,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube.GetPrim())
    _set_xform_pose(
        xformable,
        translate=center,
        orient_xyzw=_quat_xyzw_from_axes(axes),
        scale=extents,
    )
    UsdGeom.Imageable(cube).CreatePurposeAttr().Set("physics")
    _apply_shape_collision_defaults(
        cube.GetPrim(),
        physics_material=physics_material,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
    )
    return cube


def _create_capsule_collider(
    stage,
    path,
    center,
    axes,
    extents,
    primary_axis,
    physics_material,
    contact_offset,
    rest_offset,
):
    axis_map = ["X", "Y", "Z"]
    a = int(primary_axis)
    other = [i for i in range(3) if i != a]
    length = float(extents[a])
    radius = float(max(min(extents[other[0]], extents[other[1]]) * 0.5, 0.0015))
    height = float(max(length - 2.0 * radius, 0.0005))

    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateAxisAttr().Set(axis_map[a])
    capsule.CreateRadiusAttr().Set(radius)
    capsule.CreateHeightAttr().Set(height)
    xformable = UsdGeom.Xformable(capsule.GetPrim())
    _set_xform_pose(
        xformable,
        translate=center,
        orient_xyzw=_quat_xyzw_from_axes(axes),
        scale=(1.0, 1.0, 1.0),
    )
    UsdGeom.Imageable(capsule).CreatePurposeAttr().Set("physics")
    _apply_shape_collision_defaults(
        capsule.GetPrim(),
        physics_material=physics_material,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
    )
    return capsule


def _create_box_visual(
    stage,
    path,
    center,
    axes,
    extents,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube.GetPrim())
    _set_xform_pose(
        xformable,
        translate=center,
        orient_xyzw=_quat_xyzw_from_axes(axes),
        scale=extents,
    )
    # Keep render/default purpose so debug visuals are visible in viewport.
    UsdGeom.Imageable(cube).CreatePurposeAttr().Set("default")
    return cube


def _create_capsule_visual(
    stage,
    path,
    center,
    axes,
    extents,
    primary_axis,
):
    axis_map = ["X", "Y", "Z"]
    a = int(primary_axis)
    other = [i for i in range(3) if i != a]
    length = float(extents[a])
    radius = float(max(min(extents[other[0]], extents[other[1]]) * 0.5, 0.0015))
    height = float(max(length - 2.0 * radius, 0.0005))

    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.CreateAxisAttr().Set(axis_map[a])
    capsule.CreateRadiusAttr().Set(radius)
    capsule.CreateHeightAttr().Set(height)
    xformable = UsdGeom.Xformable(capsule.GetPrim())
    _set_xform_pose(
        xformable,
        translate=center,
        orient_xyzw=_quat_xyzw_from_axes(axes),
        scale=(1.0, 1.0, 1.0),
    )
    UsdGeom.Imageable(capsule).CreatePurposeAttr().Set("default")
    return capsule


def _create_primitive_visuals(
    stage,
    visual_root_path,
    visual_specs,
):
    visual_root = UsdGeom.Xform.Define(stage, visual_root_path)
    group_paths = set()
    for spec in visual_specs:
        name = str(spec["name"])
        if "/" in name:
            parts = name.split("/")[:-1]
            running = str(visual_root_path)
            for p in parts:
                running = f"{running}/{p}"
                group_paths.add(running)
    for gp in sorted(group_paths):
        UsdGeom.Xform.Define(stage, gp)

    for spec in visual_specs:
        name = str(spec["name"])
        shape = str(spec["shape"])
        center = np.asarray(spec["center"], dtype=np.float32)
        axes = np.asarray(spec["axes"], dtype=np.float32)
        extents = np.asarray(spec["extents"], dtype=np.float32)

        if shape == "box":
            _create_box_visual(
                stage,
                f"{visual_root_path}/{name}",
                center=center,
                axes=axes,
                extents=extents,
            )
            continue

        if shape == "capsule":
            _create_capsule_visual(
                stage,
                f"{visual_root_path}/{name}",
                center=center,
                axes=axes,
                extents=extents,
                primary_axis=int(spec.get("primary_axis", 0)),
            )
            continue

    return visual_root


def _create_primitive_colliders(
    stage,
    collision_root_path,
    collider_specs,
    physics_material,
    contact_offset,
    rest_offset,
):
    collision_root = UsdGeom.Xform.Define(stage, collision_root_path)
    # Explicit grouping keeps a clean hierarchy: collision/body, collision/front_panel, collision/handles.
    group_paths = set()
    for spec in collider_specs:
        name = str(spec["name"])
        if "/" in name:
            parts = name.split("/")[:-1]
            running = str(collision_root_path)
            for p in parts:
                running = f"{running}/{p}"
                group_paths.add(running)
    for gp in sorted(group_paths):
        UsdGeom.Xform.Define(stage, gp)

    for spec in collider_specs:
        name = str(spec["name"])
        shape = str(spec["shape"])
        center = np.asarray(spec["center"], dtype=np.float32)
        axes = np.asarray(spec["axes"], dtype=np.float32)
        extents = np.asarray(spec["extents"], dtype=np.float32)

        if shape == "box":
            _create_box_collider(
                stage,
                f"{collision_root_path}/{name}",
                center=center,
                axes=axes,
                extents=extents,
                physics_material=physics_material,
                contact_offset=contact_offset,
                rest_offset=rest_offset,
            )
            continue

        if shape == "capsule":
            _create_capsule_collider(
                stage,
                f"{collision_root_path}/{name}",
                center=center,
                axes=axes,
                extents=extents,
                primary_axis=int(spec.get("primary_axis", 0)),
                physics_material=physics_material,
                contact_offset=contact_offset,
                rest_offset=rest_offset,
            )
            continue

    return collision_root


def _estimate_mass_from_bbox(verts, density_kg_m3, fill_ratio, min_mass_kg, max_mass_kg):
    verts = np.asarray(verts, dtype=np.float32)
    if len(verts) == 0:
        return float(min_mass_kg)
    _, extents = _bbox_from_verts(verts)
    volume_m3 = float(extents[0] * extents[1] * extents[2])
    mass = volume_m3 * float(density_kg_m3) * float(fill_ratio)
    return float(np.clip(mass, float(min_mass_kg), float(max_mass_kg)))


def _box_diagonal_inertia(mass, extents_xyz):
    ex, ey, ez = [float(x) for x in extents_xyz]
    ixx = (mass / 12.0) * (ey * ey + ez * ez)
    iyy = (mass / 12.0) * (ex * ex + ez * ez)
    izz = (mass / 12.0) * (ex * ex + ey * ey)
    # Keep inertia above a small floor to avoid near-singular tensors.
    return np.array([max(ixx, 1.0e-4), max(iyy, 1.0e-4), max(izz, 1.0e-4)], dtype=np.float32)


def _apply_rigidbody_stability(
    prim,
    solver_pos_iters=64,
    solver_vel_iters=16,
    max_depenetration_velocity=0.25,
    enable_ccd=True,
):
    if PhysxSchema is not None:
        try:
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            # More position iterations reduce interpenetration jitter under gripper contacts.
            rb_api.CreateSolverPositionIterationCountAttr().Set(int(solver_pos_iters))
            # Velocity iterations improve impulse/friction convergence.
            rb_api.CreateSolverVelocityIterationCountAttr().Set(int(solver_vel_iters))
            # Clamp depenetration speed to prevent explosive push-back.
            rb_api.CreateMaxDepenetrationVelocityAttr().Set(float(max_depenetration_velocity))
            if enable_ccd:
                rb_api.CreateEnableCCDAttr().Set(True)
                rb_api.CreateEnableSpeculativeCCDAttr().Set(True)
            return
        except Exception:
            pass

    prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(int(solver_pos_iters))
    prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(int(solver_vel_iters))
    prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(float(max_depenetration_velocity))
    if enable_ccd:
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)


def _create_link(
    stage,
    link_path,
    verts,
    faces,
    mass,
    diagonal_inertia,
    kinematic=False,
    translate=(0.0, 0.0, 0.0),
    orient_xyzw=(0.0, 0.0, 0.0, 1.0),
    linear_damping=None,
    angular_damping=None,
    solver_pos_iters=64,
    solver_vel_iters=16,
    max_depenetration_velocity=0.25,
    physics_material=None,
    contact_offset=0.003,
    rest_offset=0.0,
    collision_approximation="convexHull",
    collision_mode="mesh",
    collision_specs=None,
    visual_match_collision=False,
):
    link = UsdGeom.Xform.Define(stage, link_path)
    _set_xform_pose(link, translate=translate, orient_xyzw=orient_xyzw, scale=(1.0, 1.0, 1.0))

    prim = link.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateDiagonalInertiaAttr().Set(to_vec3f(diagonal_inertia))
    mass_api.CreateCenterOfMassAttr().Set(to_vec3f((0.0, 0.0, 0.0)))

    if linear_damping is not None:
        prim.CreateAttribute("physics:linearDamping", Sdf.ValueTypeNames.Float).Set(float(linear_damping))
    if angular_damping is not None:
        prim.CreateAttribute("physics:angularDamping", Sdf.ValueTypeNames.Float).Set(float(angular_damping))
    if kinematic:
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    _apply_rigidbody_stability(
        prim,
        solver_pos_iters=solver_pos_iters,
        solver_vel_iters=solver_vel_iters,
        max_depenetration_velocity=max_depenetration_velocity,
        enable_ccd=(not kinematic),
    )

    if bool(visual_match_collision) and str(collision_mode) == "primitive" and collision_specs is not None and len(collision_specs) > 0:
        # Debug-focused option: visual hierarchy mirrors primitive collision specs 1:1.
        _create_primitive_visuals(
            stage,
            f"{link_path}/visual",
            visual_specs=collision_specs,
        )
    else:
        _create_visual_mesh(stage, f"{link_path}/visual", verts, faces)

    if str(collision_mode) == "primitive" and collision_specs is not None and len(collision_specs) > 0:
        _create_primitive_colliders(
            stage,
            f"{link_path}/collision",
            collider_specs=collision_specs,
            physics_material=physics_material,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        )
    else:
        _create_collision_mesh(
            stage,
            f"{link_path}/collision",
            verts,
            faces,
            physics_material=physics_material,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
            approximation=collision_approximation,
        )
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


def _split_drawers_into_blocks(drawer_obj, template_name):
    all_verts = np.asarray(drawer_obj.overall_obj_mesh.vertices, dtype=np.float32)
    all_faces = np.asarray(drawer_obj.overall_obj_mesh.faces, dtype=np.int32)
    drawer_infos = []
    cursor = 0

    for drawer_idx in range(drawer_obj.number_of_drawer[0]):
        handle_num = int(drawer_obj.number_of_handle[drawer_idx]) if drawer_idx < len(drawer_obj.number_of_handle) else 1
        if template_name == "Drawer_with_U_handle":
            cuboid_count = 6 + 3 * handle_num
        else:
            cuboid_count = 6 + handle_num

        face_count = cuboid_count * 12
        f0, f1 = cursor, min(cursor + face_count, len(all_faces))
        drawer_faces_global = all_faces[f0:f1]
        cursor = f1

        if len(drawer_faces_global) == 0:
            drawer_infos.append({
                "verts": np.zeros((0, 3), dtype=np.float32),
                "faces": np.zeros((0, 3), dtype=np.int32),
                "blocks": [],
                "handle_num": handle_num,
                "template_name": template_name,
            })
            continue

        unique_vids = np.unique(drawer_faces_global.reshape(-1))
        d_verts = all_verts[unique_vids]
        mapping = -np.ones((len(all_verts),), dtype=np.int32)
        mapping[unique_vids] = np.arange(len(unique_vids), dtype=np.int32)
        d_faces = mapping[drawer_faces_global]

        blocks = []
        local_cursor = 0
        for _ in range(cuboid_count):
            b0, b1 = local_cursor, min(local_cursor + 12, len(d_faces))
            block_faces = d_faces[b0:b1]
            local_cursor = b1
            if len(block_faces) == 0:
                blocks.append((np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)))
                continue
            b_vids = np.unique(block_faces.reshape(-1))
            b_verts = d_verts[b_vids]
            b_map = -np.ones((len(d_verts),), dtype=np.int32)
            b_map[b_vids] = np.arange(len(b_vids), dtype=np.int32)
            b_faces = b_map[block_faces]
            blocks.append((b_verts, b_faces))

        drawer_infos.append({
            "verts": d_verts,
            "faces": d_faces,
            "blocks": blocks,
            "handle_num": handle_num,
            "template_name": template_name,
        })

    return drawer_infos


def _split_part_into_blocks(part_verts, part_faces, faces_per_block=12):
    part_verts = np.asarray(part_verts, dtype=np.float32)
    part_faces = np.asarray(part_faces, dtype=np.int32)
    blocks = []
    cursor = 0
    while cursor < len(part_faces):
        b0 = cursor
        b1 = min(cursor + int(faces_per_block), len(part_faces))
        cursor = b1
        block_faces = part_faces[b0:b1]
        if len(block_faces) == 0:
            continue
        vids = np.unique(block_faces.reshape(-1))
        b_verts = part_verts[vids]
        remap = -np.ones((len(part_verts),), dtype=np.int32)
        remap[vids] = np.arange(len(vids), dtype=np.int32)
        b_faces = remap[block_faces]
        blocks.append((b_verts, b_faces))
    return blocks


def _collider_spec_from_block(name, verts, shrink_by=0.0):
    center, axes, extents = _obb_from_verts(verts, min_extent=2.0e-4)
    if shrink_by > 0.0:
        extents = _safe_shrink_extent(extents, shrink=shrink_by)
    return {
        "name": str(name),
        "shape": "box",
        "center": center,
        "axes": axes,
        "extents": extents,
    }


def _capsule_spec_from_verts(name, verts, shrink_by=0.0):
    center, axes, extents = _obb_from_verts(verts, min_extent=2.0e-4)
    if shrink_by > 0.0:
        extents = _safe_shrink_extent(extents, shrink=shrink_by)
    primary_axis = int(np.argmax(extents))
    return {
        "name": str(name),
        "shape": "capsule",
        "center": center,
        "axes": axes,
        "extents": extents,
        "primary_axis": primary_axis,
    }


def _push_handle_specs_forward(
    specs,
    panel_name,
    handle_prefix,
    contact_offset,
    min_push=0.004,
    push_ratio=1.2,
):
    panel_spec = None
    for s in specs:
        if str(s.get("name", "")) == str(panel_name):
            panel_spec = s
            break
    if panel_spec is None:
        return specs

    panel_center = np.asarray(panel_spec["center"], dtype=np.float32)
    panel_axes = np.asarray(panel_spec["axes"], dtype=np.float32)
    panel_extents = np.asarray(panel_spec["extents"], dtype=np.float32)
    if panel_axes.shape != (3, 3) or panel_extents.shape[0] != 3:
        return specs

    thin_axis = int(np.argmin(panel_extents))
    panel_normal = panel_axes[:, thin_axis]
    norm = float(np.linalg.norm(panel_normal))
    if norm < 1e-8:
        return specs
    panel_normal = panel_normal / norm

    forward_push = max(
        float(min_push),
        float(push_ratio) * float(panel_extents[thin_axis]),
        float(contact_offset) * 2.5,
    )

    for s in specs:
        name = str(s.get("name", ""))
        if not name.startswith(str(handle_prefix)):
            continue
        center = np.asarray(s["center"], dtype=np.float32)
        direction = panel_normal.copy()
        if float(np.dot(center - panel_center, direction)) < 0.0:
            direction = -direction
        s["center"] = center + direction * float(forward_push)

    return specs


def _build_drawer_primitive_collision_specs(drawer_info, contact_offset):
    blocks = list(drawer_info.get("blocks", []))
    template_name = str(drawer_info.get("template_name", "Regular_drawer"))
    handle_num = int(drawer_info.get("handle_num", 1))
    if len(blocks) < 6:
        return []

    specs = []
    body_shrink = max(float(contact_offset) * 0.55, 0.0009)
    panel_shrink = max(float(contact_offset) * 0.40, 0.0007)
    handle_shrink = max(float(contact_offset) * 0.25, 0.0004)

    # 5 shell pieces for drawer body (left, right, rear, inner-front rail, bottom).
    for body_idx in range(5):
        b_verts = np.asarray(blocks[body_idx][0], dtype=np.float32)
        if len(b_verts) == 0:
            continue
        specs.append(_collider_spec_from_block(f"body/box_{body_idx}", b_verts, shrink_by=body_shrink))

    # Front decorative panel as a thin dedicated box collider for stable grasps.
    front_panel_verts = np.asarray(blocks[5][0], dtype=np.float32)
    if len(front_panel_verts) > 0:
        specs.append(_collider_spec_from_block("front_panel/box", front_panel_verts, shrink_by=panel_shrink))

    if template_name == "Drawer_with_U_handle":
        for h in range(max(handle_num, 1)):
            b0 = 6 + 3 * h
            b1 = min(b0 + 3, len(blocks))
            group_verts = [np.asarray(blocks[k][0], dtype=np.float32) for k in range(b0, b1) if len(blocks[k][0]) > 0]
            if len(group_verts) == 0:
                continue
            specs.append(
                _capsule_spec_from_verts(
                    f"handles/handle_{h}",
                    np.concatenate(group_verts, axis=0),
                    shrink_by=handle_shrink,
                )
            )
    else:
        for h in range(max(handle_num, 1)):
            idx = 6 + h
            if idx >= len(blocks):
                continue
            h_verts = np.asarray(blocks[idx][0], dtype=np.float32)
            if len(h_verts) == 0:
                continue
            specs.append(_capsule_spec_from_verts(f"handles/handle_{h}", h_verts, shrink_by=handle_shrink))

    return _push_handle_specs_forward(
        specs,
        panel_name="front_panel/box",
        handle_prefix="handles/",
        contact_offset=contact_offset,
    )


def _build_door_primitive_collision_specs(local_verts, local_faces, contact_offset):
    blocks = _split_part_into_blocks(local_verts, local_faces, faces_per_block=12)
    if len(blocks) == 0:
        return []
    specs = []
    panel_shrink = max(float(contact_offset) * 0.45, 0.0007)
    handle_shrink = max(float(contact_offset) * 0.30, 0.0005)

    panel_verts = np.asarray(blocks[0][0], dtype=np.float32)
    if len(panel_verts) > 0:
        specs.append(_collider_spec_from_block("panel/box", panel_verts, shrink_by=panel_shrink))

    if len(blocks) > 1:
        handle_verts = np.asarray(blocks[1][0], dtype=np.float32)
        if len(handle_verts) > 0:
            specs.append(_capsule_spec_from_verts("handle/capsule", handle_verts, shrink_by=handle_shrink))
    return _push_handle_specs_forward(
        specs,
        panel_name="panel/box",
        handle_prefix="handle/",
        contact_offset=contact_offset,
    )


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
    # Keep opening travel < full depth to avoid hard-stop singular contacts at rail end.
    travel_proportion = 0.80
    drawer_len = float(part.drawer_size[drawer_idx][2])
    travel_total = drawer_len * travel_proportion
    return 0.0, max(0.0, travel_total)


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

        t_grasp_parent = t_parent_inv @ spec["world_transformation_matrix"]
        t_grasp_parent_scaled = t_grasp_parent.copy()
        t_grasp_parent_scaled[:3, 3] = t_grasp_parent_scaled[:3, 3] * float(scale)
        pos_local = t_grasp_parent_scaled[:3, 3]
        rot_local = t_grasp_parent[:3, :3]
        quat_local = Rot.from_matrix(rot_local).as_quat()
        approach_local = rot_local @ np.array([0.0, 0.0, 1.0], dtype=float)
        finger_local = rot_local @ np.array([1.0, 0.0, 0.0], dtype=float)

        grasp_path = f"{link_path}/grasps/grasp_{valid_idx}"
        g_xform = UsdGeom.Xform.Define(stage, grasp_path)
        g_xform.ClearXformOpOrder()
        g_xform.AddTranslateOp().Set(to_vec3d(pos_local))
        g_xform.AddOrientOp().Set(to_quatf_xyzw(quat_local))
        g_prim = g_xform.GetPrim()
        g_prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(to_vec3f(approach_local))
        if "world_finger_closing_direction" in spec:
            g_prim.CreateAttribute("grasp:finger_closing", Sdf.ValueTypeNames.Vector3f).Set(to_vec3f(finger_local))
        if "grasp_width" in spec:
            g_prim.CreateAttribute("grasp:width", Sdf.ValueTypeNames.Float).Set(float(spec["grasp_width"]))
        if "manip_params_size" in spec:
            g_prim.CreateAttribute("grasp:manip_params_size", Sdf.ValueTypeNames.Int).Set(int(spec["manip_params_size"]))
        g_prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(t_grasp_parent_scaled.flatten().tolist())
        valid_idx += 1


def _build_grasp_params_for_door(door_obj, door_idx):
    params = []
    for trans_ratio in (-0.5, 0.0, 0.5):
        params.append((trans_ratio, 0.0, door_idx))
    return params


def _build_grasp_params_for_drawer(drawer_obj, drawer_idx):
    params = []
    handle_num = int(drawer_obj.number_of_handle[drawer_idx]) if drawer_idx < len(drawer_obj.number_of_handle) else 1
    for handle_idx in range(max(handle_num, 1)):
        for trans_ratio in (-0.5, 0.0, 0.5):
            params.append((trans_ratio, 0.0, drawer_idx, handle_idx))
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


def _create_physx_joint_tuning(joint_prim):
    if PhysxSchema is not None:
        try:
            j_api = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
            j_api.CreateEnableProjectionAttr().Set(True)
            j_api.CreateProjectionLinearToleranceAttr().Set(0.003)
            j_api.CreateProjectionAngularToleranceAttr().Set(8.0)
            return
        except Exception:
            pass

    joint_prim.CreateAttribute("physxJoint:enableProjection", Sdf.ValueTypeNames.Bool).Set(True)
    joint_prim.CreateAttribute("physxJoint:projectionLinearTolerance", Sdf.ValueTypeNames.Float).Set(0.003)
    joint_prim.CreateAttribute("physxJoint:projectionAngularTolerance", Sdf.ValueTypeNames.Float).Set(8.0)


def export_storagefurniture_usda(
    data,
    save_path,
    scale=1.0,
    init_pos=(0.0, 0.0, 0.0),
    init_euler=(0.0, 0.0, 0.0),
    anchor_base=True,
    dynamic_base_mass=3.0,
    base_mass_kg=None,
    door_mass_kg=None,
    drawer_mass_kg=None,
    door_joint_drive_damping=45.0,
    door_joint_drive_stiffness=0.0,
    drawer_joint_drive_damping=120.0,
    drawer_joint_drive_stiffness=0.0,
    close_drawer_initially=True,
    contact_offset=0.002,
    rest_offset=0.0,
):
    stage = Usd.Stage.CreateNew(save_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetMetadata("metersPerUnit", 1.0)

    root_path = "/StorageFurniture"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    set_initial_pose(root, init_pos, init_euler)
    root.GetPrim().CreateAttribute("dataset:id", Sdf.ValueTypeNames.String).Set(str(data.get("id", "unknown")))

    links_path = f"{root_path}/links"
    joints_path = f"{root_path}/joints"
    UsdGeom.Xform.Define(stage, links_path)
    UsdGeom.Xform.Define(stage, joints_path)

    physics_material = _ensure_physics_material(stage)

    components = []
    for c in data["conceptualization"]:
        obj = _instantiate_template_object(c["template"], c["parameters"])
        components.append((c["template"], obj))

    base_parts = []
    door_links = []
    drawer_links = []
    global_door_counter = 0
    global_drawer_counter = 0

    # Build movable links (door / drawer) and collect remaining parts for base link.
    for template_name, obj in components:
        if template_name == "Regular_door":
            per_door_faces = 12 * 2
            door_parts = _split_mesh_by_face_blocks(obj.overall_obj_mesh, per_door_faces, obj.number_of_door[0])

            for door_idx, (verts, faces) in enumerate(door_parts):
                if len(verts) == 0 or len(faces) == 0:
                    continue

                global_door_idx = global_door_counter
                global_door_counter += 1
                link_name = f"door_{global_door_idx}"
                link_path = f"{links_path}/{link_name}"

                hinge_world = _door_joint_origin_world(obj, door_idx)
                local_verts = verts - hinge_world[None, :].astype(np.float32)
                local_verts = local_verts * float(scale)

                mass_est = _estimate_mass_from_bbox(
                    local_verts,
                    density_kg_m3=650.0,
                    fill_ratio=0.45,
                    min_mass_kg=0.35,
                    max_mass_kg=9.0,
                )
                mass_val = float(door_mass_kg) if door_mass_kg is not None else float(mass_est)
                _, ext = _bbox_from_verts(local_verts)
                inertia = _box_diagonal_inertia(mass_val, ext)

                _create_link(
                    stage,
                    link_path,
                    local_verts,
                    faces,
                    mass=mass_val,
                    diagonal_inertia=inertia,
                    translate=np.array(hinge_world, dtype=float) * float(scale),
                    orient_xyzw=(0.0, 0.0, 0.0, 1.0),
                    linear_damping=0.8,
                    angular_damping=3.5,
                    solver_pos_iters=64,
                    solver_vel_iters=16,
                    max_depenetration_velocity=0.25,
                    physics_material=physics_material,
                    contact_offset=contact_offset,
                    rest_offset=rest_offset,
                    collision_mode="primitive",
                    collision_specs=_build_door_primitive_collision_specs(local_verts, faces, contact_offset=contact_offset),
                    visual_match_collision=True,
                )

                door_links.append((global_door_idx, door_idx, link_name, obj, hinge_world))
            continue

        if template_name in ("Regular_drawer", "Drawer_with_U_handle"):
            drawer_infos = _split_drawers_into_blocks(obj, template_name)

            for drawer_idx, info in enumerate(drawer_infos):
                verts = np.asarray(info["verts"], dtype=np.float32)
                faces = np.asarray(info["faces"], dtype=np.int32)
                if len(verts) == 0 or len(faces) == 0:
                    continue

                global_drawer_idx = global_drawer_counter
                global_drawer_counter += 1
                link_name = f"drawer_{global_drawer_idx}"
                link_path = f"{links_path}/{link_name}"

                lower, upper = _drawer_joint_limits(obj, drawer_idx)
                shifted_blocks = []
                if close_drawer_initially:
                    verts = verts.copy()
                    verts[:, 2] = verts[:, 2] - float(upper)
                    for b_verts, b_faces in info["blocks"]:
                        b_verts = np.asarray(b_verts, dtype=np.float32).copy()
                        if len(b_verts) > 0:
                            b_verts[:, 2] = b_verts[:, 2] - float(upper)
                        shifted_blocks.append((b_verts, np.asarray(b_faces, dtype=np.int32)))
                else:
                    shifted_blocks = [(np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32)) for v, f in info["blocks"]]

                scaled_verts = verts * float(scale)
                scaled_drawer_info = dict(info)
                scaled_drawer_info["blocks"] = [
                    (np.asarray(v, dtype=np.float32) * float(scale), np.asarray(f, dtype=np.int32))
                    for v, f in shifted_blocks
                ]
                scaled_drawer_info["verts"] = scaled_verts

                mass_est = _estimate_mass_from_bbox(
                    scaled_verts,
                    density_kg_m3=650.0,
                    fill_ratio=0.19,
                    min_mass_kg=1.2,
                    max_mass_kg=4.5,
                )
                mass_val = float(drawer_mass_kg) if drawer_mass_kg is not None else float(mass_est)
                _, ext = _bbox_from_verts(scaled_verts)
                inertia = _box_diagonal_inertia(mass_val, ext)

                _create_link(
                    stage,
                    link_path,
                    scaled_verts,
                    faces,
                    mass=mass_val,
                    diagonal_inertia=inertia,
                    linear_damping=0.6,
                    angular_damping=4.0,
                    solver_pos_iters=80,
                    solver_vel_iters=16,
                    max_depenetration_velocity=0.20,
                    physics_material=physics_material,
                    contact_offset=contact_offset,
                    rest_offset=rest_offset,
                    collision_mode="primitive",
                    collision_specs=_build_drawer_primitive_collision_specs(
                        scaled_drawer_info,
                        contact_offset=contact_offset,
                    ),
                    visual_match_collision=True,
                )

                drawer_links.append((global_drawer_idx, drawer_idx, link_name, obj))
            continue

        base_parts.append((template_name, obj))

    if len(base_parts) == 0:
        raise RuntimeError("No base mesh found for StorageFurniture.")

    base_v_list = []
    base_f_list = []
    base_offset = 0
    for _, obj in base_parts:
        v = np.asarray(obj.vertices, dtype=np.float32)
        f = np.asarray(obj.faces, dtype=np.int32)
        if len(v) == 0 or len(f) == 0:
            continue
        base_v_list.append(v)
        base_f_list.append(f + base_offset)
        base_offset += len(v)

    if len(base_v_list) == 0:
        raise RuntimeError("No valid base geometry to export.")

    base_verts = np.concatenate(base_v_list, axis=0) * float(scale)
    base_faces = np.concatenate(base_f_list, axis=0)

    base_mass_est = _estimate_mass_from_bbox(
        base_verts,
        density_kg_m3=650.0,
        fill_ratio=0.30,
        min_mass_kg=10.0,
        max_mass_kg=180.0,
    )
    base_mass_val = float(base_mass_kg) if base_mass_kg is not None else float(base_mass_est)
    _, base_ext = _bbox_from_verts(base_verts)
    base_inertia = _box_diagonal_inertia(base_mass_val, base_ext)

    _create_link(
        stage,
        f"{links_path}/base_link",
        base_verts,
        base_faces,
        mass=base_mass_val if anchor_base else float(dynamic_base_mass),
        diagonal_inertia=base_inertia,
        kinematic=bool(anchor_base),
        linear_damping=None if anchor_base else 2.5,
        angular_damping=None if anchor_base else 8.0,
        solver_pos_iters=48,
        solver_vel_iters=12,
        max_depenetration_velocity=0.3,
        physics_material=physics_material,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
        collision_approximation="convexDecomposition",
    )

    # Door joints.
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

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(door_joint_drive_stiffness))
        drive.CreateDampingAttr().Set(float(door_joint_drive_damping))
        drive.CreateMaxForceAttr().Set(800.0)

        _create_physx_joint_tuning(joint.GetPrim())

        _add_grasps(
            stage,
            f"{links_path}/{link_name}",
            door_obj,
            _build_grasp_params_for_door(door_obj, local_door_idx),
            parent_translate=hinge_world,
            parent_orient_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=scale,
        )

    # Drawer joints.
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

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        # Zero stiffness avoids spring explosions; damping dissipates impact/handle chatter.
        drive.CreateStiffnessAttr().Set(float(drawer_joint_drive_stiffness))
        drive.CreateDampingAttr().Set(float(drawer_joint_drive_damping))
        drive.CreateMaxForceAttr().Set(800.0)

        _create_physx_joint_tuning(joint.GetPrim())

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "whk_new.pkl"), "rb") as f:
        data_list = pickle.load(f)

    output_dir = os.path.join(script_dir, "storagefurniture_usda_outputs")
    os.makedirs(output_dir, exist_ok=True)

    SCALE_FACTOR = 0.0023

    for i, data in enumerate(data_list[:1]):
        save_name = f"storagefurniture_{i}.usda"
        export_storagefurniture_usda(
            data,
            os.path.join(output_dir, save_name),
            scale=SCALE_FACTOR,
            init_pos=(0.0, 0.0, 0.0),
            init_euler=(0.0, 0.0, 0.0),
            anchor_base=True,
            dynamic_base_mass=35.0,
            contact_offset=0.002,
            rest_offset=0.0,
        )
