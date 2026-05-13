from concept_template import *
from geometry_template import *
from scipy.spatial.transform import Rotation as Rot
import pickle
from utils import COLOR20
import open3d as o3d
from knowledge_utils import *
from knowledge_definitions import *
from copy import deepcopy

import os  # 确保顶部导入了os

# 1. 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在文件夹（code/Bottle）
current_folder = os.path.dirname(current_script_path)
# 3. 向上跳2级，到达项目根目录（template_rich_knowledge/）
project_root = os.path.dirname(os.path.dirname(current_folder))
# 4. 拼接机械手STL文件路径
gripper_stl_path = os.path.join(project_root, "standard_gripper.stl")


if __name__ == "__main__":

    with open("lemon_tea.pkl", "rb") as f:
        data_list = pickle.load(f)

    for data in data_list:

        conceptualization = data['conceptualization']

        objs = {}
        region_knowledge_wrappers = {}


        vertices_list = []
        faces_list = []
        total_num_vertices = 0
        
        for c in conceptualization:
            module = eval(c['template'])
            obj = module(**c['parameters'])
            vertices_list.append(obj.vertices)
            faces_list.append(obj.faces + total_num_vertices)
            total_num_vertices += len(obj.vertices)

            objs[c['template']] = obj
            region_knowledge_wrappers[c['template']] = Region_Knowledge_Wrapper(obj)

        final_vertices = np.concatenate(vertices_list)
        final_faces = np.concatenate(faces_list)
        final_mesh = trimesh.Trimesh(final_vertices, final_faces)
        pts = np.array(final_mesh.sample(20000))
        

        # affordance
        affordance_label = np.zeros(pts.shape[0])
        for template, obj in objs.items():
            res = np.full((pts.shape[0]), False)
            if (obj.semantic == 'Lid'):
                res = region_knowledge_wrappers[template].check(lid_affordance, pts)
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
        for template, obj in objs.items():

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

        # =========================
        # Grasp Pose + Direction Visualization (Modified)
        # =========================

        def draw_arrow(origin, direction, color=[1, 0, 0], length=0.2):
            """创建一条简单的直线段来表示方向向量"""
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end = origin + direction * length

            points = [origin, end]
            lines = [[0, 1]]
            colors = [color]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            return line_set

        # 准备夹爪基础模型
        gripper_mesh = o3d.io.read_triangle_mesh(gripper_stl_path)
        gripper_mesh.scale(10, center=[0, 0, 0])
        gripper_mesh.paint_uniform_color([1, 0, 0]) # 红色夹爪
        gripper_mesh.compute_vertex_normals()

        grasp_assets = [] # 存储所有可视化物体

        for template, obj in objs.items():
            # 仅针对 Cylindrical_Lid 进行可视化测试
            if not isinstance(obj, Cylindrical_Lid):
                continue

            # 测试不同的参数 (例如：侧面抓取和顶部抓取)
            test_params = [(0.0, 0.0), (0.5, 0), (1.0, 0.0), (1.5, 0), (0.0, 1.0)] 
            
            for p1, p2 in test_params:
                grasp_spec = get_grasp_spec(obj, manipulation_params=(p1, p2))

                if grasp_spec is None:
                    continue

                # 提取位姿矩阵
                grasp_pose = grasp_spec["world_transformation_matrix"]
                origin = grasp_spec["world_position"]

                # 1. 添加坐标系 (Grasp Coordinate Frame)
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                coord.transform(grasp_pose)
                grasp_assets.append(coord)

                # 2. 添加夹爪模型 (Gripper Mesh)
                g = deepcopy(gripper_mesh)
                g.transform(grasp_pose)
                grasp_assets.append(g)

                # 3. 添加语义方向直线
                # 获取世界坐标系下的方向
                app_dir = grasp_spec["world_approach_direction"]
                cls_dir = grasp_spec["world_finger_closing_direction"]

                # 紫色直线: Approach 方向 (长度设为 0.5 方便观察)
                arrow_approach = draw_arrow(origin, app_dir, color=[0.5, 0, 0.5], length=20) 
                
                # 橙色直线: Finger Close 方向
                arrow_finger = draw_arrow(origin, cls_dir, color=[1, 0.5, 0], length=20)

                grasp_assets.extend([arrow_approach, arrow_finger])

        # =========================
        # Final Visualization
        # =========================
        # 使用分部件 mesh 代替点云:
        # Body -> 灰色, Lid -> 蓝色
        part_meshes = []
        for _, obj in objs.items():
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(obj.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(obj.faces)
            mesh.compute_vertex_normals()

            if obj.semantic == 'Body':
                mesh.paint_uniform_color([0.6, 0.6, 0.6])  # gray
            elif obj.semantic == 'Lid':
                mesh.paint_uniform_color([0.1, 0.4, 0.9])  # blue
            else:
                mesh.paint_uniform_color([0.75, 0.75, 0.75])  # fallback

            part_meshes.append(mesh)

        # 加上物体本身的坐标系 coordinates (之前循环生成的)
        o3d.visualization.draw_geometries(part_meshes + coordinates + grasp_assets)
