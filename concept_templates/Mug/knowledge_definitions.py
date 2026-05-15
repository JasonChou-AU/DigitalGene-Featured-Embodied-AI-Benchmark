from concept_template import *
from geometry_template import *
from knowledge_utils import *
from utils import *
from scipy.spatial.transform import Rotation as Rot


def _build_transformation_matrix(approach, closing, position):
    z_axis = approach / (np.linalg.norm(approach) + 1e-12)
    x_axis = closing / (np.linalg.norm(closing) + 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, z_axis)

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = position
    return T


def _rotate_vector_about_axis(v, axis, angle_rad):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v_par = np.dot(v, axis) * axis
    v_perp = v - v_par
    v_perp_rot = v_perp * np.cos(angle_rad) + np.cross(axis, v_perp) * np.sin(angle_rad)
    return v_par + v_perp_rot


def _make_transformation_matrix(position, rotation, rotation_order="xyz"):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler(rotation_order, rotation).as_matrix()
    T[:3, 3] = np.array(position, dtype=np.float64)
    return T


def handle_affordance(obj, pt):

    def is_affordance(obj, pt):

        if (isinstance(obj, Trifold_Handle)):
            _pt = apply_transformation(np.array([pt]), -np.array(obj.position), -np.array(obj.rotation), rotation_order='ZXY', offset_first=True)[0]

            vertical_y_offset = (-obj.horizontal_length[0] * np.sin(obj.horizontal_rotation[0]) - obj.horizontal_length[1] * np.sin(obj.horizontal_rotation[1])) / 2
            vertical_z_offset = (obj.horizontal_length[1] * np.cos(obj.horizontal_rotation[1]) + obj.mounting_offset[0] + obj.horizontal_length[0] * np.cos(obj.horizontal_rotation[0])) / 2
            delta_y = obj.horizontal_separation[0] - obj.horizontal_length[0] * np.sin(obj.horizontal_rotation[0]) + obj.horizontal_length[1] * np.sin(obj.horizontal_rotation[1])
            delta_z = obj.mounting_offset[0] - obj.horizontal_length[1] * np.cos(obj.horizontal_rotation[1]) + obj.horizontal_length[0] * np.cos(obj.horizontal_rotation[0])
            vertical_rotation = np.arctan(delta_z / delta_y)
            vertical_length = np.sqrt(delta_y * delta_y + delta_z * delta_z) + obj.horizontal_thickness[1]
            vertical_mesh_position = np.array([
                0, 
                vertical_y_offset, 
                vertical_z_offset + obj.vertical_thickness[1] / 2
            ])
            vertical_mesh_rotation = np.array([-vertical_rotation, 0, 0])
            _pt = inverse_transformation(_pt, vertical_mesh_position, vertical_mesh_rotation)

            return (_pt[0] >= -obj.vertical_thickness[0]/2 - AFFORDACE_PROXIMITY_THRES and
                    _pt[0] <= obj.vertical_thickness[0]/2 + AFFORDACE_PROXIMITY_THRES and 
                    _pt[1] >= -vertical_length/2 - AFFORDACE_PROXIMITY_THRES and
                    _pt[1] <= vertical_length/2 + AFFORDACE_PROXIMITY_THRES and
                    _pt[2] >= -obj.vertical_thickness[1]/2 - AFFORDACE_PROXIMITY_THRES and
                    _pt[2] <= obj.vertical_thickness[1]/2 + AFFORDACE_PROXIMITY_THRES)
        

        elif (isinstance(obj, Curved_Handle)):
            _pt = apply_transformation(np.array([pt]), -np.array(obj.position), -np.array(obj.rotation), rotation_order='ZYX', offset_first=True)[0]

            mesh_position = np.array([0, 0, 0])
            mesh_rotation = np.array([0, 0, -np.pi / 2])
            _pt = apply_transformation(np.array([_pt]), -mesh_position, -mesh_rotation, rotation_order='ZYX', offset_first=True)[0]

            _pt_radius = np.linalg.norm(_pt[[0, 2]])
            _pt_angle = np.arctan2(_pt[2], _pt[0])

            return (_pt_radius <= obj.radius[0] + obj.radius[1] + AFFORDACE_PROXIMITY_THRES and 
                    _pt_radius >= obj.radius[0] - obj.radius[1] - AFFORDACE_PROXIMITY_THRES and 
                    _pt_angle >= obj.central_angle[0]/3 and _pt_angle <= obj.central_angle[0]*2/3)
        
        else:
            return False
    
    
    return is_affordance(obj, pt)



def part_pose(obj):
    RT = transformation_matrix(obj.position, obj.rotation)
    return RT


import numpy as np
from scipy.spatial.transform import Rotation as Rot


def get_grasp_spec(obj, manipulation_params=None):
    if manipulation_params is None:
        manipulation_params = [0, 0]

    # default
    rot_param_1 = manipulation_params[0]
    rot_param_2 = manipulation_params[1] if len(manipulation_params) > 1 else 0
    rot_param_3 = manipulation_params[2] if len(manipulation_params) > 2 else 0

    # convert to angles
    rot_angle_1 = np.pi / 2 * rot_param_1
    rot_angle_2 = np.pi / 4 * rot_param_2

    # =========================
    # Trifold Handle
    # =========================
    if isinstance(obj, Trifold_Handle):

        h_rot = obj.horizontal_rotation
        h_len = obj.horizontal_length

        # Match Trifold_Handle geometry in concept_template.py.
        top_mesh_position = np.array([
            0,
            obj.horizontal_separation[0] / 2 - h_len[0] * np.sin(h_rot[0]) / 2,
            obj.mounting_offset[0] + h_len[0] * np.cos(h_rot[0]) / 2,
        ], dtype=np.float64)
        top_mesh_rotation = np.array([h_rot[0], 0, 0], dtype=np.float64)

        delta_y = (
            obj.horizontal_separation[0]
            - h_len[0] * np.sin(h_rot[0])
            + h_len[1] * np.sin(h_rot[1])
        )

        delta_z = (
            obj.mounting_offset[0]
            - h_len[1] * np.cos(h_rot[1])
            + h_len[0] * np.cos(h_rot[0])
        )

        vertical_length = np.sqrt(delta_y**2 + delta_z**2) + obj.horizontal_thickness[1]
        vertical_rotation = np.arctan2(delta_z, delta_y)
        vertical_y_offset = (-h_len[0] * np.sin(h_rot[0]) - h_len[1] * np.sin(h_rot[1])) / 2
        vertical_z_offset = (h_len[1] * np.cos(h_rot[1]) + obj.mounting_offset[0] + h_len[0] * np.cos(h_rot[0])) / 2

        if len(manipulation_params) >= 3:
            mode_param = rot_param_2
            close_param = rot_param_3
        else:
            mode_param = -1.0
            close_param = rot_param_2

        # p1 selects the handle segment and position:
        #   [-3, -1] -> top horizontal mesh, moving along its local Z axis.
        #   (-1, 1] -> middle vertical mesh, moving along its local Y axis.
        if rot_param_1 <= -1.0:
            is_top_mesh_grasp = True
            top_position_param = np.clip(rot_param_1 + 2.0, -1.0, 1.0)
            grasp_position_local = np.array([
                0.0,
                0.0,
                top_position_param * h_len[0] / 2,
            ], dtype=np.float64)
            geometry_position = top_mesh_position
            geometry_rotation = top_mesh_rotation
            tangent_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            in_plane_width = float(obj.horizontal_thickness[1])
            plane_normal_width = float(obj.horizontal_thickness[0])
        else:
            is_top_mesh_grasp = False
            vertical_position_param = np.clip(rot_param_1, -1.0, 1.0)
            grasp_position_local = np.array([
                0.0,
                vertical_position_param * vertical_length / 2,
                0.0,
            ], dtype=np.float64)
            geometry_position = np.array([
                0,
                vertical_y_offset,
                vertical_z_offset + obj.vertical_thickness[1] / 2,
            ], dtype=np.float64)
            geometry_rotation = np.array([vertical_rotation, 0, 0], dtype=np.float64)
            tangent_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            in_plane_width = float(obj.vertical_thickness[1])
            plane_normal_width = float(obj.vertical_thickness[0])

        plane_normal_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        in_plane_cross_local = np.cross(tangent_local, plane_normal_local)
        in_plane_cross_local = in_plane_cross_local / (np.linalg.norm(in_plane_cross_local) + 1e-12)

        # mode_param controls finger closing direction:
        #   < 0 -> close along the handle plane
        #   >=0 -> close perpendicular to the handle plane
        if mode_param >= 0.0:
            approach_local = in_plane_cross_local
            closing_base_local = plane_normal_local
            grasp_width = plane_normal_width
        else:
            approach_local = plane_normal_local
            closing_base_local = in_plane_cross_local
            grasp_width = in_plane_width

        close_spin_angle = np.pi * close_param
        if is_top_mesh_grasp:
            close_spin_angle += np.pi / 2

        finger_closing_local = _rotate_vector_about_axis(
            closing_base_local,
            approach_local,
            close_spin_angle,
        )

        T_local = _build_transformation_matrix(
            approach=approach_local,
            closing=finger_closing_local,
            position=grasp_position_local,
        )

        T_geom = _make_transformation_matrix(geometry_position, geometry_rotation)
        T_obj_world = _make_transformation_matrix(obj.position, obj.rotation)
        grasp_pose = T_obj_world @ T_geom @ T_local

        grasp_rotation = grasp_pose[:3, :3]
        grasp_position = grasp_pose[:3, 3]
        world_quat = Rot.from_matrix(grasp_rotation).as_quat()
        world_approach = grasp_rotation @ np.array([0.0, 0.0, 1.0])
        world_closing = grasp_rotation @ np.array([1.0, 0.0, 0.0])

        return {
            "world_transformation_matrix": grasp_pose,
            "world_position": grasp_position,
            "world_rotation": world_quat,  # quaternion [x, y, z, w]
            "world_approach_direction": world_approach,
            "world_finger_closing_direction": world_closing,
            "grasp_width": grasp_width,
            "manip_params_size": 3,
        }

    # =========================
    # Curved Handle
    # =========================
    elif isinstance(obj, Curved_Handle):

        central_radius = obj.radius[0]
        tube_radius = obj.radius[1]
        angle_range = obj.central_angle[0]
        grasp_radius  =central_radius - tube_radius * 0.3        

        # choose angle along arc
        theta = angle_range / 2 + rot_param_1 * (angle_range / 6)

        # local grasp point on arc (in torus local coordinates)
        # Keep grasp position around the middle ring surface.
        grasp_ring_radius = grasp_radius
        x = grasp_ring_radius * np.cos(theta)
        z = grasp_ring_radius * np.sin(theta)

        grasp_position_local = np.array([x, 0, z])

        # Build local grasp frame in torus coordinates.
        # New 3-param interface for curved handle:
        #   p1 (rot_param_1): position along arc
        #   p2 (mode_param):  <0 -> radial (toward center), >=0 -> normal to handle circle plane
        #   p3 (close_param): continuous finger-closing spin around approach
        # Backward compatibility:
        #   If only 2 params are provided, treat p2 as close_param and keep radial mode.
        if len(manipulation_params) >= 3:
            mode_param = rot_param_2
            close_param = rot_param_3
        else:
            mode_param = -1.0
            close_param = rot_param_2

        radial_approach = -grasp_position_local
        radial_approach[1] = 0.0
        radial_approach = radial_approach / (np.linalg.norm(radial_approach) + 1e-12)

        # In torus local frame, the handle circle lies in XZ plane, so its plane normal is +Y/-Y.
        plane_normal_approach = np.array([0.0, 1.0 if mode_param >= 0.0 else -1.0, 0.0], dtype=np.float64)

        if mode_param >= 0.0:
            approach_local = plane_normal_approach
            closing_base_local = radial_approach
        else:
            approach_local = radial_approach
            closing_base_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        close_spin_angle = np.pi * close_param
        finger_closing_local = _rotate_vector_about_axis(
            closing_base_local, approach_local, close_spin_angle + np.pi / 2
        )

        T_local = _build_transformation_matrix(
            approach=approach_local,
            closing=finger_closing_local,
            position=grasp_position_local,
        )

        # Match concept_template Curved_Handle construction:
        # Torus local frame is first rotated by [0, 0, pi/2] (xyz),
        # then transformed by object pose with xyz convention.
        T_geom = np.eye(4)
        T_geom[:3, :3] = Rot.from_euler("xyz", [0, 0, np.pi / 2]).as_matrix()

        T_obj_world = np.eye(4)
        T_obj_world[:3, :3] = Rot.from_euler("xyz", obj.rotation).as_matrix()
        T_obj_world[:3, 3] = np.array(obj.position, dtype=np.float64)

        grasp_pose = T_obj_world @ T_geom @ T_local

        grasp_rotation = grasp_pose[:3, :3]
        grasp_position = grasp_pose[:3, 3]
        world_quat = Rot.from_matrix(grasp_rotation).as_quat()
        world_approach = grasp_rotation @ np.array([0.0, 0.0, 1.0])
        world_closing = grasp_rotation @ np.array([1.0, 0.0, 0.0])
        grasp_width = float(tube_radius * 2.0)

        return {
            "world_transformation_matrix": grasp_pose,
            "world_position": grasp_position,
            "world_rotation": world_quat,  # quaternion [x, y, z, w]
            "world_approach_direction": world_approach,
            "world_finger_closing_direction": world_closing,
            "grasp_width": grasp_width,
            "manip_params_size": 3,
        }

    else:
        return None
