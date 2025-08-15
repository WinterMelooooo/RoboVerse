import sys
sys.path.append('.')
from roboverse_learn.algorithms.utils.franka_ros import FrankaRobot
import rclpy
import numpy as np
import time
rclpy.init()
robot = FrankaRobot()

def get_mat(grasp17):
    assert grasp17.shape[0] == 17, "输入必须是一维、长度为17的向量"

    # 取出旋转矩阵 R 和平移向量 t
    R = grasp17[4:13].reshape(3, 3) # 元素 4~12（共9个）
    t = grasp17[13:16] # 元素 13~15（共3个）

    # 构造 4x4 齐次矩阵
    H = np.eye(4, dtype=R.dtype)
    H[:3, :3] = R
    H[:3, 3] = t

    return H


# target = [0.04, 0.04, 0, 0, 0, -90, 0, 90, -135]
# robot.goto_joint_degree(target)
# time.sleep(2.5)

# target = np.array([ 2.56520195e+01, 1.13507688e-01, 1.99999996e-02, 3.99999991e-02,
#                 -9.17071640e-01, -3.29150021e-01, -2.25032955e-01, -3.16067547e-01,
#                 9.44172084e-01, -9.29537714e-02, 2.43065566e-01, -1.41196605e-02,
#                 -9.69907105e-01, 4.57741231e-01, 1.39805019e-01, 2.27555692e-01,
#                 -1.00000000e+00])
# #target[15] = target[15]
# #target[13]  = target[13] + 0.10  # 调整平移向量 t 的 z 分量
# #print(get_mat(target))

# target = np.array([ 2.44836750e+01,  1.25071585e-01,  1.99999996e-02,  3.99999991e-02,
#                    -9.32647586e-01, -3.42663050e-01,  1.12918101e-01, -3.02560031e-01,
#                     9.13321316e-01,  2.72583187e-01, -1.96534693e-01,  2.20059559e-01,
#                    -9.55483079e-01,  6.00693345e-01,  5.79447821e-02,  1.66872978e-01,
#                    -1.00000000e+00])
# target[13] = target[13] + 0.05  # 调整平移向量 t 的 x 分量
# target[14] = target[14] + 0.10  # 调整平移向量 t 的 y 分量
# target[15] = target[15] + 0.15
# mat = get_mat(target)
# print(mat)
# # robot.goto_ee_pos(get_mat(target))
# robot.goto_ee_pos(mat)


target = [0.015022234991192818, 0.015022234991192818, -0.09432021463409695, 0.886769713421453, 0.014089397593840982, -1.607296119702405, -0.06133104177759363, 2.1841289903536296, -2.333810640080136]
robot.goto(target)
