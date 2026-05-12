import numpy as np
from scipy.spatial.transform import Rotation as Rot


PROXIMITY_THRES = 0.02
AFFORDACE_PROXIMITY_THRES = 0.02
SAMPLENUM = 10000


class Region_Knowledge_Wrapper():
    def __init__(self, template_instance):
        self.instance = template_instance
    
    def check(self, func, pts, *args, **kwargs):
        res = []
        for pt in pts:
            if (not self.instance.proximation(pt)):
                res.append(False)
            else:
                res.append(func(self.instance, pt, *args, **kwargs))

        return res


def transformation_matrix(position, rotation):
    RT = np.eye(4)
    RT[:3, :3] = Rot.from_euler('xyz', rotation, degrees=False).as_matrix()
    RT[:3, -1] = position
    return RT


def inverse_transformation(pt, position, rotation):
    RT = transformation_matrix(position, rotation)
    _pt = np.array([pt[0], pt[1], pt[2], 1])
    _pt = np.linalg.inv(RT) @ _pt
    return _pt[:3]



def build_transformation_matrix(approach, closing, position):
    """
    构建 4x4 齐次变换矩阵
    approach: 趋近方向 (局部 z 轴)
    closing: 手指闭合方向 (局部 x 轴)
    position: 抓取点位置 (局部坐标)
    """
    z_axis = approach / np.linalg.norm(approach)
    x_axis = closing / np.linalg.norm(closing)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis) # 重新校准确保正交

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = position
    return T