import sys
sys.path.append('.')
from roboverse_learn.algorithms.utils.franka_ros import FrankaRobot
import rclpy
import numpy as np
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


target = [0.04, 0.04, 0, 0, 0, -90, 0, 90, -45]
robot.goto_joint_degree(target)

# target = np.array([ 2.69396076e+01, 5.65867983e-02, 1.99999996e-02, 3.99999991e-02,
# -5.21948443e-01, 3.11749896e-01, -7.93973728e-01, -7.67303452e-01,
# 2.34977720e-01, 5.96678788e-01, 3.72579920e-01, 9.20652225e-01,
# 1.16564146e-01, 7.55163377e-01, 2.79683444e-01, 3.61096983e-01,
# -1.00000000e+00])


#robot.goto_ee_pos(get_mat(target))
