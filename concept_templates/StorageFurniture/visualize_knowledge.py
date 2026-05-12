from concept_template import *
from geometry_template import *
from scipy.spatial.transform import Rotation as Rot
import pickle
import os
from copy import deepcopy
from utils import COLOR20
import open3d as o3d
from knowledge_utils import *
from knowledge_definitions import *
import numpy as np


current_script_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(current_folder))
gripper_stl_path = os.path.join(project_root, "standard_gripper.stl")


if __name__ == "__main__":

    with open("conceptualization.pkl", "rb") as f:
        data_list = pickle.load(f)

    for data in data_list:

        conceptualization = data['conceptualization']


        objs = {}
        region_knowledge_wrappers = {}


        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        templates = []
        
        for c in conceptualization:
            templates.append(c['template'])
            module = eval(c['template'])
            obj = module(**c['parameters'])
            vertices_list.append(obj.vertices)
            faces_list.append(obj.faces + total_num_vertices)
            total_num_vertices += len(obj.vertices)

            if (c['template'] not in objs.keys()):
                objs[c['template']] = []
            if (c['template'] not in region_knowledge_wrappers.keys()):
                region_knowledge_wrappers[c['template']] = []
            objs[c['template']].append(obj)
            region_knowledge_wrappers[c['template']].append(Region_Knowledge_Wrapper(obj))


        final_vertices = np.concatenate(vertices_list)
        final_faces = np.concatenate(faces_list)
        final_mesh = trimesh.Trimesh(final_vertices, final_faces)
        pts = np.array(final_mesh.sample(20000))
        

        # affordance
        affordance_label = np.zeros(pts.shape[0])
        for template, obj_list in objs.items():
            for obj_idx, obj in enumerate(obj_list):
                res = np.full((pts.shape[0]), False)
                if (obj.semantic == 'Door'):
                    res = region_knowledge_wrappers[template][obj_idx].check(door_affordance, pts)
                elif (obj.semantic == 'Drawer'):
                    res = region_knowledge_wrappers[template][obj_idx].check(drawer_affordance, pts)
                for idx in range(pts.shape[0]):
                    if (affordance_label[idx] == 0 and res[idx] != False):
                        affordance_label[idx] = res[idx]

        affordance_label = affordance_label.astype(np.int32)
        color = COLOR20[affordance_label]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])



        # part pose
        coordinates = []
        poses = []
        for template, obj_list in objs.items():
            for obj_idx, obj in enumerate(obj_list):
                pose = part_pose(obj)
                poses.append(pose)

                coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                v = np.array(coordinate.vertices)
                v = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
                v = (pose @ v.T).T
                coordinate.vertices = o3d.utility.Vector3dVector(v[:, :3])
                coordinates.append(coordinate)
            
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.visualization.draw_geometries([pcd] + coordinates)

        # grasp pose + direction
        def draw_arrow(origin, direction, color=[1, 0, 0], length=0.3):
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end = origin + direction * length
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([origin, end])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([color])
            return line_set

        def draw_axis_line(origin, axis, color, half_length=0.35):
            axis = np.array(axis, dtype=float)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            start = np.array(origin, dtype=float) - axis * half_length
            end = np.array(origin, dtype=float) + axis * half_length
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([start, end])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([color])
            return line_set

        gripper_mesh = o3d.io.read_triangle_mesh(gripper_stl_path)
        gripper_mesh.scale(0.2 / 3.0, center=[0, 0, 0])
        gripper_mesh.paint_uniform_color([1, 0, 0])
        gripper_mesh.compute_vertex_normals()

        grasp_assets = []
        for template, obj_list in objs.items():
            for obj in obj_list:
                grasp_spec = get_grasp_spec(obj)
                if grasp_spec is None:
                    continue

                manip_params_size = grasp_spec.get("manip_params_size", 0)
                if manip_params_size == 4:
                    # Drawer: (trans_ratio, rot_ratio, drawer_idx, handle_idx)
                    # 直接在这里写死测试参数，便于手工控制演示内容
                    test_params = [
                        (0.0, 0.0, 0, 0),
                        (-0.5, 0.0, 0, 0),
                        (0.5, 0.0, 0, 0),
                        (0.0, 0.0, 1, 0),
                    ]
                elif manip_params_size == 3:
                    # Door: (trans_ratio, rot_ratio, door_idx)
                    # 直接在这里写死测试参数，便于手工控制演示内容
                    test_params = [
                        (0.0, 0.0, 0),
                        (-0.5, 0.0, 0),
                        (0.5, 0.0, 0),
                        (0.0, 0.0, 1),
                    ]
                else:
                    test_params = [()]

                for param in test_params:
                    grasp_spec = get_grasp_spec(obj, manipulation_params=param)
                    if grasp_spec is None:
                        continue

                    grasp_pose = grasp_spec.get("world_transformation_matrix")
                    if grasp_pose is None:
                        continue
                    origin = grasp_spec.get("world_position", grasp_pose[:3, 3])
                    app_dir = grasp_spec.get("world_approach_direction", grasp_pose[:3, 2])
                    cls_dir = grasp_spec.get("world_finger_closing_direction", grasp_pose[:3, 0])

                    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
                    coord.transform(grasp_pose)
                    grasp_assets.append(coord)

                    g = deepcopy(gripper_mesh)
                    g.transform(grasp_pose)
                    grasp_assets.append(g)

                    grasp_assets.append(draw_arrow(origin, app_dir, color=[0.5, 0.0, 0.5], length=0.5))
                    grasp_assets.append(draw_arrow(origin, cls_dir, color=[1.0, 0.5, 0.0], length=0.5))

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.visualization.draw_geometries([pcd] + coordinates + grasp_assets)

        # articulation axis (4th Open3D view)
        articulation_assets = []
        for _, obj_list in objs.items():
            for obj in obj_list:
                state = articulation_state(obj)
                if state is None:
                    continue
                if state["joint_type"] == "revolute":
                    color = [1.0, 0.0, 0.0]  # door hinge axis: red
                elif state["joint_type"] == "prismatic":
                    color = [0.0, 0.0, 1.0]  # drawer pull axis: blue
                else:
                    continue

                for joint in state["joints"]:
                    articulation_assets.append(
                        draw_axis_line(
                            joint["joint_origin_world"],
                            joint["joint_axis_world"],
                            color=color,
                            half_length=0.35,
                        )
                    )

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.visualization.draw_geometries([pcd] + articulation_assets)
