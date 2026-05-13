from concept_template import *
from geometry_template import *
from knowledge_utils import *
from scipy.spatial.transform import Rotation as Rot


def lid_affordance(obj, pt):

    def is_affordance(obj, pt):
        if (isinstance(obj, Cylindrical_Lid)):
            _pt = inverse_transformation(pt, obj.position, obj.rotation)

            lid_y_offset = -(obj.outer_size[2] - obj.inner_size[2]) / 2
            _pt_radius = np.linalg.norm(_pt[[0, 2]])

            return (_pt_radius >= obj.inner_size[1] - AFFORDACE_PROXIMITY_THRES and
                    _pt_radius <= obj.outer_size[0] + AFFORDACE_PROXIMITY_THRES and
                    _pt[1] >= -obj.inner_size[2]/2 + lid_y_offset - AFFORDACE_PROXIMITY_THRES and
                    _pt[1] <= obj.outer_size[2]/2 + AFFORDACE_PROXIMITY_THRES)
    
    return is_affordance(obj, pt)


def part_pose(obj):
    RT = transformation_matrix(obj.position, obj.rotation)
    return RT


def get_grasp_spec(obj, manipulation_params=None):
    if isinstance(obj, Cylindrical_Lid):
        para1, para2 = manipulation_params if manipulation_params is not None else (0.0, 0.0)
        
        # 获取物体的世界变换
        # 假设 obj.rotation 是弧度制的 [rx, ry, rz]
        obj_rot_mat = Rot.from_euler('xyz', obj.rotation).as_matrix()
        obj_pos = np.array(obj.position)
        
        # 物体局部到世界的变换矩阵
        T_obj_world = np.eye(4)
        T_obj_world[:3, :3] = obj_rot_mat
        T_obj_world[:3, 3] = obj_pos

        if para2 == 0: 
            # --- 模式 1: 侧面抓取 (Side Grasp) ---
            # 根据 para1 计算绕 Y 轴的角度 (0-2 对应 0-360度)
            angle = para1 * np.pi 
            
            # 基础局部位置贴合在外表面
            # 注意：原点在衔接处，y 向上，所以 -obj.inner_size[2]/2 是底座中心
            r = obj.outer_size[0] + 10
            local_pos = np.array([r * np.sin(angle), -obj.inner_size[2] / 2, -r * np.cos(angle)])
            
            # 趋近方向指向圆心
            approach_dir = np.array([-np.sin(angle), 0.0, np.cos(angle)])
            # 闭合方向沿 Y 轴
            finger_closing_dir = np.array([0.0, 1.0, 0.0])
            
            grasp_width = obj.outer_size[0] * 2
            
        else:
            # --- 模式 2: 顶部抓取 (Top-down Grasp) ---
            # 基础局部位置贴合在顶盖表面（不包含夹爪长度后退量）
            local_pos = np.array([0.0, (obj.outer_size[2] - obj.inner_size[2]) / 2 + 15, 0.0])
            
            # 趋近方向向下
            approach_dir = np.array([0.0, -1.0, 0.0])
            
            # 根据 para2 计算夹爪自身的旋转角度
            angle = para2 * np.pi
            finger_closing_dir = np.array([np.cos(angle), 0.0, np.sin(angle)])
            
            grasp_width = obj.outer_size[0] * 2

        # 构建局部变换矩阵
        T_local = build_transformation_matrix(approach_dir, finger_closing_dir, local_pos)
        
        # 转换到世界坐标系: T_world = T_object * T_local
        T_world = T_obj_world @ T_local
        
        world_pos = T_world[:3, 3]
        world_rot_mat = T_world[:3, :3]
        world_quat = Rot.from_matrix(world_rot_mat).as_quat()
        
        # 提取世界坐标系下的方向向量
        world_approach = world_rot_mat @ np.array([0, 0, 1])
        world_closing = world_rot_mat @ np.array([1, 0, 0])

        return {
            "world_transformation_matrix": T_world,
            "world_position": world_pos,
            "world_rotation": world_quat, # 四元数 [x, y, z, w]
            "world_approach_direction": world_approach,
            "world_finger_closing_direction": world_closing,
            "grasp_width": grasp_width,
            "manip_params_size": 2,
        }

    return None
