#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
raw_extr_data = {
    "D455_dexhand": {
        "rvec": [ 0.02322675, -0.00136824,  0.0387247 ],
        "tvec": [-0.18066608, -0.18902881,  0.45980965],
    }
}


def project_and_draw_axes(img, K, dist, rvec, tvec, axis_length=1.0):
    """
    在 img 上绘制世界坐标系的三条轴
    axis_length 单位与外参一致（比如米）。
    """
    # 世界坐标系的 4 个点：原点、X、Y、Z
    pts_world = np.array([
        [0,0,0],
        [axis_length,0,0],
        [0,axis_length,0],
        [0,0,axis_length]
    ], dtype=np.float32)

    # 投影到图像平面
    imgpts, _ = cv2.projectPoints(
        pts_world,    # (4,3)
        rvec,         # (3,1)
        tvec,         # (3,1)
        K,            # (3,3)
        dist          # (1×5) 或 (k,)
    )
    imgpts = imgpts.reshape(-1,2).astype(int)

    o = tuple(imgpts[0])  # 原点
    x = tuple(imgpts[1])  # X 轴
    y = tuple(imgpts[2])  # Y 轴
    z = tuple(imgpts[3])  # Z 轴

    # 画线：原点→各轴端点
    cv2.line(img, o, x, (0,0,255), 2)  # 红色 X
    cv2.line(img, o, y, (0,255,0), 2)  # 绿色 Y
    cv2.line(img, o, z, (255,0,0), 2)  # 蓝色 Z

    return img

def main():
    cam_intr = np.array([[654.6406,   0.0000, 642.2062],
        [  0.0000, 654.6406, 362.4909],
        [  0.0000,   0.0000,   1.0000]], dtype=np.float32)
    cam_extr_raw = raw_extr_data["D455_dexhand"]
    tvec = np.array(cam_extr_raw["tvec"])
    rvec = np.array(cam_extr_raw["rvec"])
    dist = np.array([0., 0., 0., 0., 0.])
    img_path = "/home/balen/Projects/yktang/RoboVerse/charuco_vis.png"
    img = cv2.imread(img_path)
    projected_img = project_and_draw_axes(img, cam_intr, dist, rvec, tvec)
    cv2.imshow("Projected Axes", projected_img)


if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
