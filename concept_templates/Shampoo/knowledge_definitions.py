from concept_template import *
from geometry_template import *
from knowledge_utils import *
from utils import *
from scipy.spatial.transform import Rotation as Rot
import numpy as np


def _build_transformation_matrix(approach_dir, finger_closing_dir, position):
    z_axis = np.asarray(approach_dir, dtype=float)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    x_axis = np.asarray(finger_closing_dir, dtype=float)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)

    t = np.eye(4, dtype=float)
    t[:3, 0] = x_axis
    t[:3, 1] = y_axis
    t[:3, 2] = z_axis
    t[:3, 3] = np.asarray(position, dtype=float)
    return t


def nozzle_affordance(obj, pt):

    def is_affordance(obj, pt):

        if (isinstance(obj, Cylindrical_cap)):
            _pt = apply_transformation(np.array([pt]), -np.array(obj.position), -np.array(obj.rotation), rotation_order='ZXY')[0]

            nozzle_mesh_position = np.array([0, obj.inner_size[2] / 2, 0])
            nozzle_mesh_rotation = np.array([0, 0, 0])
            __pt = inverse_transformation(_pt, nozzle_mesh_position, nozzle_mesh_rotation)

            _pt_radius = np.linalg.norm(__pt[[0, 2]])

            return (_pt_radius <= obj.outer_size[0] and
                    __pt[1] >= - AFFORDACE_PROXIMITY_THRES and
                    __pt[1] <= (obj.outer_size[2] - obj.inner_size[2])/2 + AFFORDACE_PROXIMITY_THRES)
        
        

        elif (isinstance(obj, Regular_cap)):
            _pt = apply_transformation(np.array([pt]), -np.array(obj.position), -np.array(obj.rotation), rotation_order='ZXY')[0]

            _pt_radius = np.linalg.norm(__pt[[0, 2]])

            return (_pt_radius <= obj.size[0] and
                    __pt[1] >= - AFFORDACE_PROXIMITY_THRES and
                    __pt[1] <= obj.size[1]/2 + AFFORDACE_PROXIMITY_THRES)
    
        
        else:
            return False
    
    
    return is_affordance(obj, pt)


def part_pose(obj):
    RT = transformation_matrix(obj.position, obj.rotation)
    return RT


def get_grasp_spec(obj, manipulation_params=None):
    # Shampoo: one top grasp above cap, approach points down to cap.
    if isinstance(obj, Cylindrical_cap):
        top_offset = 0.18
        local_position = np.array([0.0, obj.outer_size[2] / 2 + top_offset, 0.0], dtype=float)
        approach_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        finger_closing_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        t_local = _build_transformation_matrix(approach_dir, finger_closing_dir, local_position)

        t_obj = transformation_matrix(obj.position, obj.rotation)
        t_world = t_obj @ t_local
        r_world = t_world[:3, :3]

        return {
            "world_transformation_matrix": t_world,
            "world_position": t_world[:3, 3],
            "world_rotation": Rot.from_matrix(r_world).as_quat(),
            "world_approach_direction": r_world @ np.array([0.0, 0.0, 1.0], dtype=float),
            "world_finger_closing_direction": r_world @ np.array([1.0, 0.0, 0.0], dtype=float),
            "grasp_width": obj.outer_size[0] * 2,
            "manip_params_size": 1,
        }

    if isinstance(obj, Regular_cap):
        top_offset = 0.18
        local_position = np.array([0.0, obj.size[1] / 2 + top_offset, 0.0], dtype=float)
        approach_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        finger_closing_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        t_local = _build_transformation_matrix(approach_dir, finger_closing_dir, local_position)

        t_obj = transformation_matrix(obj.position, obj.rotation)
        t_world = t_obj @ t_local
        r_world = t_world[:3, :3]

        return {
            "world_transformation_matrix": t_world,
            "world_position": t_world[:3, 3],
            "world_rotation": Rot.from_matrix(r_world).as_quat(),
            "world_approach_direction": r_world @ np.array([0.0, 0.0, 1.0], dtype=float),
            "world_finger_closing_direction": r_world @ np.array([1.0, 0.0, 0.0], dtype=float),
            "grasp_width": obj.size[0] * 2,
            "manip_params_size": 1,
        }

    if isinstance(obj, Regular_nozzle):
        top_offset = 0.18
        verts = np.asarray(obj.vertices, dtype=float)
        center_x = 0.5 * (float(np.min(verts[:, 0])) + float(np.max(verts[:, 0])))
        center_z = 0.5 * (float(np.min(verts[:, 2])) + float(np.max(verts[:, 2])))
        top_y = float(np.max(verts[:, 1]))
        world_pos = np.array([center_x, top_y + top_offset, center_z], dtype=float)

        approach_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        finger_closing_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        world_t = _build_transformation_matrix(approach_dir, finger_closing_dir, world_pos)
        world_r = world_t[:3, :3]

        width = max(float(np.max(verts[:, 0]) - np.min(verts[:, 0])), float(np.max(verts[:, 2]) - np.min(verts[:, 2])))
        width = max(width, 0.02)

        return {
            "world_transformation_matrix": world_t,
            "world_position": world_pos,
            "world_rotation": Rot.from_matrix(world_r).as_quat(),
            "world_approach_direction": world_r @ np.array([0.0, 0.0, 1.0], dtype=float),
            "world_finger_closing_direction": world_r @ np.array([1.0, 0.0, 0.0], dtype=float),
            "grasp_width": width,
            "manip_params_size": 1,
        }

    return None
