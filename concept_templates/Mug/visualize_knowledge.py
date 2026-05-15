from concept_template import *
from geometry_template import *
from scipy.spatial.transform import Rotation as Rot
import pickle
from utils import COLOR20
import open3d as o3d
from knowledge_utils import *
from knowledge_definitions import *
import os
from copy import deepcopy

# ===== 自动路径 =====
current_script_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(current_folder))
gripper_stl_path = os.path.join(project_root, "standard_gripper.stl")

if __name__ == "__main__":

    with open("Mug_conceptualization.pkl", "rb") as f:
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
            if (obj.semantic == 'Handle'):
                res = region_knowledge_wrappers[template].check(handle_affordance, pts)
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
            poses.append(poses)

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            v = np.array(coordinate.vertices)
            v = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
            v = (pose @ v.T).T
            coordinate.vertices = o3d.utility.Vector3dVector(v[:, :3])
            coordinates.append(coordinate)
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.visualization.draw_geometries([pcd] + coordinates)

        # =========================
        # Grasp Pose + Direction Visualization
        # =========================
        def draw_arrow(origin, direction, color=[1, 0, 0], length=0.2):
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            end = origin + direction * length
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([origin, end])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([color])
            return line_set

        gripper_mesh = o3d.io.read_triangle_mesh(gripper_stl_path)
        gripper_mesh.scale(5, center=[0, 0, 0])
        gripper_mesh.paint_uniform_color([1, 0, 0])
        gripper_mesh.compute_vertex_normals()

        grasp_assets = []

        for template, obj in objs.items():

            if obj.semantic != "Handle":
                continue

            for x in np.arange(-3, -0.5, 0.5):
                manip_params_list = [
                    [x, -1.0, 0.0],  # top horizontal mesh, in-handle-plane grasp
                    [x, +1.0, 0.0],  # top horizontal mesh, perpendicular-to-handle-plane grasp
                ]
                for manip_param in manip_params_list:
                    spec = get_grasp_spec(obj, manip_param)
                    if spec is None or "world_transformation_matrix" not in spec:
                        continue
                    grasp_pose = np.asarray(spec["world_transformation_matrix"], dtype=np.float64)
                    if grasp_pose.shape != (4, 4):
                        continue
    
                    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
                    v = np.asarray(coord.vertices)
                    v = np.concatenate([v, np.ones((v.shape[0],1))], axis=1)
    
                    v = (grasp_pose @ v.T).T
    
                    coord.vertices = o3d.utility.Vector3dVector(v[:, :3])
    
                    grasp_assets.append(coord)
    
                    g = deepcopy(gripper_mesh)
                    g.transform(grasp_pose)
                    grasp_assets.append(g)
    
                    origin = np.asarray(spec["world_position"], dtype=np.float64)
                    app_dir = np.asarray(spec["world_approach_direction"], dtype=np.float64)
                    cls_dir = np.asarray(spec["world_finger_closing_direction"], dtype=np.float64)
    
                    arrow_approach = draw_arrow(origin, app_dir, color=[0.5, 0.0, 0.5], length=0.5)  # purple
                    arrow_finger = draw_arrow(origin, cls_dir, color=[1.0, 0.5, 0.0], length=0.5)    # orange
                    grasp_assets.extend([arrow_approach, arrow_finger])

            # visualize
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            o3d.visualization.draw_geometries([pcd] + coordinates + grasp_assets)
