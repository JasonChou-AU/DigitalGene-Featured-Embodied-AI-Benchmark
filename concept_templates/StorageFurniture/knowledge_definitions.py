from concept_template import *
from geometry_template import *
from knowledge_utils import *
from scipy.spatial.transform import Rotation as Rot


def _normalize(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _get_rodrigues_matrix(axis, angle):
    axis = _normalize(axis)
    identity = np.eye(3, dtype=float)
    s1 = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    s2 = np.matmul(axis[:, None], axis[None, :])
    return np.cos(angle) * identity + np.sin(angle) * s1 + (1.0 - np.cos(angle)) * s2


def _compose_world_grasp_spec(world_pos, world_rot):
    world_pos = np.array(world_pos, dtype=float)
    world_rot = np.array(world_rot, dtype=float)
    world_quat = Rot.from_matrix(world_rot).as_quat()  # [x, y, z, w]
    world_approach = world_rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    t_world = np.eye(4, dtype=float)
    t_world[:3, :3] = world_rot
    t_world[:3, 3] = world_pos
    return {
        "world_position": world_pos,
        "world_rotation": world_quat,
        "world_approach_direction": world_approach,
        "world_transformation_matrix": t_world,
    }


def _build_door_grasp_sequence_specs(grasp_spec, pivot, axis, pose_count=10, open_angle_deg=90.0):
    """
    Generate a quarter-circle opening sequence from a grasp pose.
    Follow doric_knowledge interpolation:
      P_new = pivot + R(axis, theta) * (P0 - pivot)
      R_new = R(axis, theta) * R0
    """
    if pose_count <= 0:
        return []

    pivot = np.array(pivot, dtype=float)
    axis = _normalize(axis)
    p0 = np.array(grasp_spec["world_position"], dtype=float)
    r0 = Rot.from_quat(np.array(grasp_spec["world_rotation"], dtype=float)).as_matrix()

    vec_radius = p0 - pivot
    sign = 1.0 if vec_radius[0] < 0.0 else -1.0
    max_theta = sign * np.deg2rad(float(open_angle_deg))

    sequence = []
    for progress in np.linspace(0.0, 1.0, int(pose_count)):
        theta = max_theta * progress
        r_motion = _get_rodrigues_matrix(axis, theta)
        p_new = pivot + np.matmul(r_motion, vec_radius)
        r_new = np.matmul(r_motion, r0)
        sequence.append(_compose_world_grasp_spec(p_new, r_new))
    return sequence


def door_affordance(obj, pt):

    def is_affordance(obj, pt):

        if (isinstance(obj, Regular_door)):
            _pt = apply_transformation(np.array([pt]), -np.array(obj.position), -np.array(obj.rotation), rotation_order='ZYX', offset_first=True)[0]

            for door_idx in range(obj.number_of_door[0]):
                mesh_position = [
                    obj.door_offset[door_idx][0] + obj.handle_offset[door_idx][0] * np.cos(obj.door_rotation[door_idx]) + obj.handle_size[door_idx][2] / 2 * np.sin(obj.door_rotation[door_idx]),
                    obj.door_offset[door_idx][1] + obj.handle_offset[door_idx][1],
                    obj.door_offset[door_idx][2] - obj.handle_offset[door_idx][0] * np.sin(obj.door_rotation[door_idx]) + obj.handle_size[door_idx][2] / 2 * np.cos(obj.door_rotation[door_idx])
                ]
                mesh_rotation = [0, obj.door_rotation[door_idx], 0]
                __pt = inverse_transformation(_pt, mesh_position, mesh_rotation)

                if (__pt[0] >= -obj.handle_size[door_idx][0]/2 - AFFORDACE_PROXIMITY_THRES and 
                    __pt[0] <= obj.handle_size[door_idx][0]/2 + AFFORDACE_PROXIMITY_THRES and 
                    __pt[1] >= -obj.handle_size[door_idx][1]/2 - AFFORDACE_PROXIMITY_THRES and 
                    __pt[1] <= obj.handle_size[door_idx][1]/2 + AFFORDACE_PROXIMITY_THRES and 
                    __pt[2] >= -obj.handle_size[door_idx][2]/2 - AFFORDACE_PROXIMITY_THRES and 
                    __pt[2] <= obj.handle_size[door_idx][2]/2 + AFFORDACE_PROXIMITY_THRES):
                    return True
            
            return False
    
        
        else:
            return False
    
    
    return is_affordance(obj, pt)


def drawer_affordance(obj, pt):

    def is_affordance(obj, pt):

        if (isinstance(obj, (Regular_drawer, Drawer_with_U_handle))):
            _pt = apply_transformation(np.array([pt]), -np.array(obj.position), -np.array(obj.rotation), rotation_order='ZYX', offset_first=True)[0]

            for drawer_idx in range(obj.number_of_drawer[0]):
                for mesh_idx in range(6, 6 + obj.number_of_handle[drawer_idx]):
                    if obj.number_of_handle[drawer_idx] == 2:
                        position_sign = 1 if mesh_idx == 6 else -1
                    else:
                        position_sign = 0
                    mesh_position = [
                        obj.drawer_offset[drawer_idx][0] + obj.handle_offset[drawer_idx][0] + position_sign * obj.handle_separation[drawer_idx] / 2,
                        obj.drawer_offset[drawer_idx][1] + obj.handle_offset[drawer_idx][1],
                        obj.drawer_offset[drawer_idx][2] + obj.drawer_size[drawer_idx][2] / 2 + obj.front_size[drawer_idx][2] + obj.front_size[drawer_idx][2] / 2
                    ]
                    mesh_rotation = [0, 0, 0]
                    __pt = inverse_transformation(_pt, mesh_position, mesh_rotation)

                    if (__pt[0] >= -obj.handle_sizes[drawer_idx][0]/2 - AFFORDACE_PROXIMITY_THRES and 
                        __pt[0] <= obj.handle_sizes[drawer_idx][0]/2 + AFFORDACE_PROXIMITY_THRES and 
                        __pt[1] >= -obj.handle_sizes[drawer_idx][1]/2 - AFFORDACE_PROXIMITY_THRES and 
                        __pt[1] <= obj.handle_sizes[drawer_idx][1]/2 + AFFORDACE_PROXIMITY_THRES and 
                        __pt[2] >= -obj.handle_sizes[drawer_idx][2]/2 - AFFORDACE_PROXIMITY_THRES and 
                        __pt[2] <= obj.handle_sizes[drawer_idx][2]/2 + AFFORDACE_PROXIMITY_THRES):
                        return True
            
            return False
    
        
        else:
            return False
    
    
    return is_affordance(obj, pt)


def part_pose(obj):
    RT = transformation_matrix(obj.position, obj.rotation)
    return RT


def articulation_state(obj):
    """
    Return joint-level articulation states for supported StorageFurniture parts.

    Output format:
    {
        "joint_type": "revolute" | "prismatic",
        "joint_count": int,
        "joints": [
            {
                "joint_index": int,
                "joint_axis_local": np.ndarray(3,),
                "joint_axis_world": np.ndarray(3,),
                "joint_origin_local": np.ndarray(3,),
                "joint_origin_world": np.ndarray(3,),
                "x": float
            },
            ...
        ]
    }

    x semantics:
    - Drawer: pulled distance, closed state is 0 (meters-like model unit).
    - Door: opening angle, closed state is 0 and outward opening is positive (radians).
    """
    if isinstance(obj, (Regular_drawer, Drawer_with_U_handle)):
        obj_rot_mat = Rot.from_euler("xyz", obj.rotation, degrees=False).as_matrix()
        obj_pos = np.array(obj.position, dtype=float)
        axis_local = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_world = obj_rot_mat @ axis_local

        joints = []
        for drawer_idx in range(obj.number_of_drawer[0]):
            origin_local = np.array(obj.drawer_offset[drawer_idx], dtype=float)
            origin_world = obj_rot_mat @ origin_local + obj_pos
            # Drawer closed state is defined as x = 0; opening is positive.
            state = max(float(origin_local[2]), 0.0)
            joints.append(
                {
                    "joint_index": drawer_idx,
                    "joint_axis_local": axis_local.copy(),
                    "joint_axis_world": axis_world.copy(),
                    "joint_origin_local": origin_local,
                    "joint_origin_world": origin_world,
                    "state": state,
                }
            )

        return {"joint_type": "prismatic", "joint_count": len(joints), "joints": joints}

    if isinstance(obj, Regular_door):
        obj_rot_mat = Rot.from_euler("xyz", obj.rotation, degrees=False).as_matrix()
        obj_pos = np.array(obj.position, dtype=float)
        axis_local = np.array([0.0, 1.0, 0.0], dtype=float)
        axis_world = obj_rot_mat @ axis_local

        joints = []
        for door_idx in range(obj.number_of_door[0]):
            if abs(obj.handle_offset[door_idx][0]) > 1e-6:
                hinge_sign = -1.0 if obj.handle_offset[door_idx][0] > 0 else 1.0
            else:
                hinge_sign = -1.0 if obj.door_offset[door_idx][0] >= 0 else 1.0

            origin_local = np.array(
                [
                    obj.door_offset[door_idx][0] + hinge_sign * obj.door_size[door_idx][0] / 2.0,
                    obj.door_offset[door_idx][1],
                    obj.door_offset[door_idx][2],
                ],
                dtype=float,
            )
            origin_world = obj_rot_mat @ origin_local + obj_pos
            # Door closed state is 0; outward opening is represented as positive angle.
            state = abs(float(obj.door_rotation[door_idx]))

            joints.append(
                {
                    "joint_index": door_idx,
                    "joint_axis_local": axis_local.copy(),
                    "joint_axis_world": axis_world.copy(),
                    "joint_origin_local": origin_local,
                    "joint_origin_world": origin_world,
                    "state": state,
                }
            )

        return {"joint_type": "revolute", "joint_count": len(joints), "joints": joints}

    return None


def _compose_grasp_spec(obj, geometry_position, geometry_rotation, local_position, local_rotation, grasp_width, manip_params_size):
    obj_rot_mat = Rot.from_euler("xyz", obj.rotation, degrees=False).as_matrix()
    obj_pos = np.array(obj.position, dtype=float)
    t_obj_world = np.eye(4)
    t_obj_world[:3, :3] = obj_rot_mat
    t_obj_world[:3, 3] = obj_pos

    t_geo_obj = np.eye(4)
    t_geo_obj[:3, :3] = Rot.from_euler("xyz", geometry_rotation, degrees=False).as_matrix()
    t_geo_obj[:3, 3] = np.array(geometry_position, dtype=float)

    t_local = np.eye(4)
    t_local[:3, :3] = np.array(local_rotation, dtype=float)
    t_local[:3, 3] = np.array(local_position, dtype=float)

    t_world = t_obj_world @ t_geo_obj @ t_local
    world_rot_mat = t_world[:3, :3]
    world_pos = t_world[:3, 3]
    world_quat = Rot.from_matrix(world_rot_mat).as_quat()
    world_approach = world_rot_mat @ np.array([0.0, 0.0, 1.0])
    world_closing = world_rot_mat @ np.array([1.0, 0.0, 0.0])

    return {
        "local_position": np.array(local_position, dtype=float),
        "local_rotation": np.array(local_rotation, dtype=float),
        "local_force_direction": np.array([0.0, 0.0, 1.0]),
        "world_transformation_matrix": t_world,
        "world_position": world_pos,
        "world_rotation": world_quat,
        "world_approach_direction": world_approach,
        "world_finger_closing_direction": world_closing,
        "grasp_width": grasp_width,
        "manip_params_size": manip_params_size,
    }


def get_grasp_spec(obj, manipulation_params=None):
    if isinstance(obj, Drawer_with_U_handle):
        trans_ratio, rot_ratio, drawer_idx, handle_idx = manipulation_params if manipulation_params is not None else (0.0, 0.0, 0, 0)
        rot_angle = rot_ratio * np.pi / 4
        drawer_idx = int(np.clip(int(drawer_idx), 0, obj.number_of_drawer[0] - 1))
        handle_idx = int(np.clip(int(handle_idx), 0, obj.number_of_handle[drawer_idx] - 1))
        mesh_idx = 6 + handle_idx
        if obj.number_of_handle[drawer_idx] == 2:
            position_sign = 1 if mesh_idx == 6 else -1
        else:
            position_sign = 0

        mesh_position = [
            obj.drawer_offset[drawer_idx][0] + obj.handle_offset[drawer_idx][0] + position_sign * obj.handle_separation[drawer_idx] / 2,
            obj.drawer_offset[drawer_idx][1] + obj.handle_offset[drawer_idx][1],
            obj.drawer_offset[drawer_idx][2] + obj.drawer_size[drawer_idx][2] / 2 + obj.front_size[drawer_idx][2] + obj.front_size[drawer_idx][2] / 2,
        ]
        mesh_rotation = [0.0, 0.0, 0.0]

        # U-handle bars are arranged around handle center; grasp around the opening center
        # and avoid biasing downward as regular cuboid-handle settings do.
        local_position = np.array([
            trans_ratio * obj.handle_sizes[drawer_idx][0] / 2 * 0.7,
            -obj.handle_sizes[drawer_idx][1] / 8,
            obj.handle_sizes[drawer_idx][2] * 2,
        ])
        local_rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        offset = Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        return _compose_grasp_spec(
            obj,
            mesh_position,
            mesh_rotation,
            offset @ local_position,
            offset @ local_rotation,
            obj.handle_sizes[drawer_idx][0],
            4,
        )

    if isinstance(obj, Regular_drawer):
        trans_ratio, rot_ratio, drawer_idx, handle_idx = manipulation_params if manipulation_params is not None else (0.0, 0.0, 0, 0)
        rot_angle = rot_ratio * np.pi / 4
        drawer_idx = int(np.clip(int(drawer_idx), 0, obj.number_of_drawer[0] - 1))
        handle_idx = int(np.clip(int(handle_idx), 0, obj.number_of_handle[drawer_idx] - 1))
        mesh_idx = 6 + handle_idx
        if obj.number_of_handle[drawer_idx] == 2:
            position_sign = 1 if mesh_idx == 6 else -1
        else:
            position_sign = 0

        mesh_position = [
            obj.drawer_offset[drawer_idx][0] + obj.handle_offset[drawer_idx][0] + position_sign * obj.handle_separation[drawer_idx] / 2,
            obj.drawer_offset[drawer_idx][1] + obj.handle_offset[drawer_idx][1],
            obj.drawer_offset[drawer_idx][2] + obj.drawer_size[drawer_idx][2] / 2 + obj.front_size[drawer_idx][2] + obj.front_size[drawer_idx][2] / 2,
        ]
        mesh_rotation = [0.0, 0.0, 0.0]

        local_position = np.array([
            trans_ratio * obj.handle_sizes[drawer_idx][0] / 2 * 0.7,
            -obj.handle_sizes[drawer_idx][1],
            obj.handle_sizes[drawer_idx][2] / 2 + 0.09,
        ])
        local_rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        offset = Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        return _compose_grasp_spec(
            obj,
            mesh_position,
            mesh_rotation,
            offset @ local_position,
            offset @ local_rotation,
            obj.handle_sizes[drawer_idx][0],
            4,
        )

    if isinstance(obj, Regular_door):
        trans_ratio, rot_ratio, door_idx = manipulation_params if manipulation_params is not None else (0.0, 0.0, 0)
        rot_angle = rot_ratio * np.pi / 4
        door_idx = int(np.clip(int(door_idx), 0, obj.number_of_door[0] - 1))
        mesh_rotation = [0.0, obj.door_rotation[door_idx], 0.0]
        mesh_position = [
            obj.door_offset[door_idx][0] + obj.handle_offset[door_idx][0] * np.cos(obj.door_rotation[door_idx]) + obj.handle_size[door_idx][2] / 2 * np.sin(obj.door_rotation[door_idx]),
            obj.door_offset[door_idx][1] + obj.handle_offset[door_idx][1],
            obj.door_offset[door_idx][2] - obj.handle_offset[door_idx][0] * np.sin(obj.door_rotation[door_idx]) + obj.handle_size[door_idx][2] / 2 * np.cos(obj.door_rotation[door_idx]),
        ]

        if obj.handle_size[door_idx][1] > obj.handle_size[door_idx][0]:
            local_position = np.array([0.0, trans_ratio * obj.handle_size[door_idx][1] / 2 * 0.7, obj.handle_size[door_idx][2] / 2 + 0.09])
            local_rotation = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            offset = Rot.from_euler("y", rot_angle, degrees=False).as_matrix()
        else:
            local_position = np.array([trans_ratio * obj.handle_size[door_idx][0] / 2 * 0.7, 0.0, obj.handle_size[door_idx][2] / 2 + 0.09])
            local_rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            offset = Rot.from_euler("x", rot_angle, degrees=False).as_matrix()
        return _compose_grasp_spec(
            obj,
            mesh_position,
            mesh_rotation,
            offset @ local_position,
            offset @ local_rotation,
            obj.handle_size[door_idx][0],
            3,
        )

    return None
