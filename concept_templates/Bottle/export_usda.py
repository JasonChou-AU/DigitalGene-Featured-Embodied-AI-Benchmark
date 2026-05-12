
from concept_template import *
from geometry_template import *
from knowledge_definitions import *
import os
import numpy as np
import pickle

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from scipy.spatial.transform import Rotation as Rot


# =========================
# 工具函数
# =========================
def to_vec3f(v):
    return Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))

def to_vec3d(v):
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))

def set_initial_pose(prim_xform, position, euler_deg):
    """
    根据 Isaac Sim 中调好的位姿设置 USD 节点的初始变换
    Args:
        prim_xform: UsdGeom.Xform 节点
        position: [x, y, z] 位置
        euler_deg: [rx, ry, rz] 欧拉角 (角度制)
    """
    # 清除旧的变换属性（防止重复调用报错）
    prim_xform.ClearXformOpOrder()
    
    # 1. 设置平移
    prim_xform.AddTranslateOp().Set(to_vec3d(position))
    
    # 2. 设置旋转 (Isaac Sim 默认顺规为 XYZ)
    r = Rot.from_euler('xyz', euler_deg, degrees=True)
    q = r.as_quat() # 返回 [x, y, z, w]
    
    # 转换为 USD 要求的 Quatf (w, x, y, z)
    quat_usd = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    prim_xform.AddOrientOp().Set(quat_usd)

# =========================
# 导出核心函数
# =========================
def export_bottle_usda(data, save_path, scale=0.1, init_pos=[0,0,0], init_euler=[0,0,0]):
    """
    Args:
        data: 瓶子的概念化数据
        save_path: USDA 保存路径
        scale: 顶点缩放倍数
        init_pos: 初始位置 [x, y, z]
        init_euler: 初始欧拉角 [rx, ry, rz] (角度制)
    """
    stage = Usd.Stage.CreateNew(save_path)
    ####
    bottle_root_path = "/Bottle" 
    bottle_root = UsdGeom.Xform.Define(stage, bottle_root_path)
    
    # 将此节点设为 defaultPrim，这样引用时最清晰
    stage.SetDefaultPrim(bottle_root.GetPrim())   
    ####
    # 场景基础设置
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y) 
    stage.SetMetadata("metersPerUnit", 1.0)
    
    # world = UsdGeom.Xform.Define(stage, "/World")
    
    # # 1. 定义 Bottle 根节点
    # bottle_root_path = "/World/Bottle"
    # bottle_root = UsdGeom.Xform.Define(stage, bottle_root_path)
    
    # --- 设置初始位姿 ---
    set_initial_pose(bottle_root, init_pos, init_euler)
    
    # 应用物理属性
    UsdPhysics.RigidBodyAPI.Apply(bottle_root.GetPrim())
    UsdPhysics.MassAPI.Apply(bottle_root.GetPrim())
    bottle_root.GetPrim().CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(0.5)

    lid_obj = None
    
    # 2. 导出几何体
    for i, c in enumerate(data["conceptualization"]):
        # 动态加载模板组件
        module = eval(c["template"])
        component = module(**c["parameters"])
        
        semantic_name = "lid" if "Lid" in c["template"] else "body"
        path = f"{bottle_root_path}/{semantic_name}"
        
        xform = UsdGeom.Xform.Define(stage, path)
        
        # 缩放顶点数据
        # verts = np.asarray(component.vertices, dtype=np.float32) * scale
        verts = np.asarray(component.vertices, dtype=np.float32) 
        faces = np.asarray(component.faces, dtype=np.int32)
        
        # Visual Mesh
        visual = UsdGeom.Mesh.Define(stage, f"{path}/visual")
        visual.CreatePointsAttr([to_vec3f(v) for v in verts])
        visual.CreateFaceVertexCountsAttr([3] * len(faces))
        visual.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        
        # Collision Mesh
        collision = UsdGeom.Mesh.Define(stage, f"{path}/collision")
        collision.CreatePointsAttr([to_vec3f(v) for v in verts])
        collision.CreateFaceVertexCountsAttr([3] * len(faces))
        collision.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        
        col_prim = collision.GetPrim()
        UsdGeom.Imageable(collision).CreatePurposeAttr().Set("physics")
        UsdPhysics.MeshCollisionAPI.Apply(col_prim)
        UsdPhysics.CollisionAPI.Apply(col_prim)
        col_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexDecomposition")

        if semantic_name == "lid":
            lid_obj = component

    # 3. 导出抓取位姿 (注意：抓取点是相对于 bottle_root 的局部坐标)
    if lid_obj is not None:
        grasp_root_path = f"{bottle_root_path}/grasps"
        UsdGeom.Xform.Define(stage, grasp_root_path)
        
        # 假设这里有获取抓取参数的函数
        test_params = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (0.0, 1.0)]
        
        for idx, (p1, p2) in enumerate(test_params):
            # 注意：此处需确保 get_grasp_spec 已定义
            spec = get_grasp_spec(lid_obj, manipulation_params=(p1, p2))
            if spec is None: continue
            
            grasp_path = f"{grasp_root_path}/grasp_{idx}"
            g_xform = UsdGeom.Xform.Define(stage, grasp_path)
            
            # 平移缩放
            # t = spec["world_position"] * scale
            t = spec["world_position"]
            q = spec["world_rotation"] 
            quat_gf = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            
            g_xform.AddTranslateOp().Set(to_vec3d(t))
            g_xform.AddOrientOp().Set(quat_gf)
            
            prim = g_xform.GetPrim()
            prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(
                to_vec3f(spec["world_approach_direction"])
            )
            
            t_mat = spec["world_transformation_matrix"].copy()
            # t_mat[:3, 3] *= scale
            prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(t_mat.flatten().tolist())

    # stage.GetRootLayer().defaultPrim = "World"
    stage.GetRootLayer().Save()
    print(f"Exported with Pose [Pos:{init_pos}, Rot:{init_euler}] to: {save_path}")

# =========================
# 执行
# =========================
if __name__ == "__main__":
    # 加载你的数据
    with open("conceptualization.pkl", "rb") as f:
        data_list = pickle.load(f)

    output_dir = "bottle_usda_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --- 用户配置区 ---
    SCALE_FACTOR = 0.2 
    
    # 在 Isaac Sim 中调好的平放数值：
    # 假设瓶子长轴原先沿 Y 轴，平放需要绕 X 轴转 90 度
    # 并为了防止掉出地面，中心点高度设为 0.05
    ADJUSTED_POS = [-0.1, 0.2, 0.0]
    ADJUSTED_ROT = [90.0, 0.0, 0.0] 
    # ----------------

    for i, data in enumerate(data_list[1:4]):
        file_name = f"bottle_{i}.usda"
        export_bottle_usda(
            data, 
            os.path.join(output_dir, file_name), 
            scale=SCALE_FACTOR,
            init_pos=ADJUSTED_POS,
            init_euler=ADJUSTED_ROT
        )