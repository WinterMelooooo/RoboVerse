import numpy as np
import cv2
import sys
import os
import json
def draw_world_frame_axes(
    img: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    axis_length: float = 1.0,
    base_thickness: int = 5,
    thickness_falloff: float = 2.0,
    save_path = None
):
    """
    在图像上绘制世界坐标系的 X/Y/Z 轴，每条轴使用梯形：
      - 近端（原点）宽度 = base_thickness
      - 远端（终点）宽度 = base_thickness * (scale ** thickness_falloff)
    参数:
        img: 输入图像（BGR 格式）。
        K: 相机内参矩阵，3×3。
        extrinsic: 相机外参，3×4 或 4×4。
        axis_length: 世界坐标轴长度。
        base_thickness: 近端厚度（像素）。
        thickness_falloff: 厚度衰减指数 (>1 时差异更明显)。
        save_path: 若需保存，可打开最后的 imwrite。
    返回:
        带梯形轴的图像副本。
    """
    # 1. 提取 R, t
    if extrinsic.shape == (4,4):
        R = extrinsic[:3,:3]
        t = extrinsic[:3,3].reshape(3,1)
    else:
        R = extrinsic[:3,:3]
        t = extrinsic[:3,3].reshape(3,1)

    # 2. 世界原点和三轴终点
    axes_world = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],   # X 轴
        [0, axis_length, 0],   # Y 轴
        [0, 0, axis_length]    # Z 轴
    ], dtype=np.float32).T      # 3×4

    # 3. 变换到相机坐标
    pts_cam = R @ axes_world + t  # 3×4

    # 4. 投影到像素，并归一化
    pts_proj = K @ pts_cam        # 3×4
    pts_pix = (pts_proj[:2] / pts_proj[2:3]).T.astype(int)  # 4×2

    # 5. 计算深度缩放比例
    depths = pts_cam[2]           # 4
    scale = depths[0] / depths    # scale[0]=1, 其余<1

    # 6. 绘制梯形轴
    img_out = img.copy()
    origin = tuple(pts_pix[0])
    colors = [(0,0,255), (0,255,0), (255,0,0)]  # BGR for X, Y, Z

    for i, color in enumerate(colors, start=1):
        pt_far = tuple(pts_pix[i])
        dx, dy = pt_far[0] - origin[0], pt_far[1] - origin[1]
        length = np.hypot(dx, dy)
        if length < 1:
            continue

        # 单位方向向量和垂直向量
        vx, vy = dx/length, dy/length
        px, py = -vy, vx

        # 近端和更显著衰减后的远端宽度
        th_near = base_thickness
        th_far  = max(1, int(base_thickness * (scale[i] ** thickness_falloff)))

        # 梯形四个顶点
        v1 = (int(origin[0] + px*th_near/2), int(origin[1] + py*th_near/2))
        v2 = (int(origin[0] - px*th_near/2), int(origin[1] - py*th_near/2))
        v3 = (int(pt_far[0] - px*th_far/2),   int(pt_far[1] - py*th_far/2))
        v4 = (int(pt_far[0] + px*th_far/2),   int(pt_far[1] + py*th_far/2))

        cv2.fillPoly(img_out, [np.array([v1, v2, v3, v4], np.int32)], color)

    # 若需保存：
    if save_path is not None:
        cv2.imwrite(save_path, img_out)
    return img_out

def rot_matrix_from_euler(dx, dy, dz):
    """生成绕世界 X/Y/Z 轴旋转的旋转矩阵（角度单位：度）"""
    rx, ry, rz = np.deg2rad([dx, dy, dz])
    Rx = np.array([[1,0,0],
                   [0,np.cos(rx),-np.sin(rx)],
                   [0,np.sin(rx), np.cos(rx)]])
    Ry = np.array([[ np.cos(ry),0,np.sin(ry)],
                   [0,1,0],
                   [-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                   [np.sin(rz), np.cos(rz),0],
                   [0,0,1]])
    # 注意旋转顺序：先绕 X，再绕 Y，再绕 Z
    return Rz @ Ry @ Rx

def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    # 1. 获取总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("视频帧数未知或为 0")
    if frame_idx < 0 or frame_idx >= total_frames:
        raise ValueError(f"帧索引 {frame_idx} 超出范围 [0, {total_frames - 1}]")

    # 2. 定位到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    # 3. 读取帧
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"读取第 {frame_idx} 帧失败")

    return frame, frame_idx


def main():
    demo_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_rlbench/sweep_to_dustpan_of_size/robot_franka/demo_0000"
    rgbs = os.path.join(demo_dir, "rgb.mp4")
    metadata = os.path.join(demo_dir, "metadata.json")
    rgb, idx = read_frame(rgbs, 0)
    with open(metadata, "r") as f:
        meta = json.load(f)


    cam_intr = np.array(meta["cam_intr"][0], dtype=np.float64)

    cam_extr = np.array(meta["cam_extr"][0], dtype=np.float64)
    c2w = np.linalg.inv(cam_extr)

    init_img = rgb[..., ::-1]

    # 创建控制窗口和滑杆
    win = "Real World Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # 先显示一帧，让后台窗口真正创建
    cv2.imshow(win, init_img)
    cv2.waitKey(1)

    # 旋转滑杆：0–360，对应 -180° 到 +180°
    for name in ("RotX","RotY","RotZ"):
        cv2.createTrackbar(name, win, 180, 720, lambda x: None)
    # 平移滑杆：0–20，对应 -0.5 到 +0.5（步长 0.05m）
    for name in ("TransX","TransY","TransZ"):
        cv2.createTrackbar(name, win, 10, 100, lambda x: None)

    while True:
        frame = init_img.copy()
        # 读滑杆
        rx = cv2.getTrackbarPos("RotX", win) - 180
        ry = cv2.getTrackbarPos("RotY", win) - 180
        rz = cv2.getTrackbarPos("RotZ", win) - 180
        tx = (cv2.getTrackbarPos("TransX", win) - 10) * 0.05
        ty = (cv2.getTrackbarPos("TransY", win) - 10) * 0.05
        tz = (cv2.getTrackbarPos("TransZ", win) - 10) * 0.05

        # 计算新的 c2w：先绕世界轴旋转，再平移
        R_add = rot_matrix_from_euler(rx, ry, rz)
        new_c2w = c2w.copy()
        # 先把相机坐标系的旋转也做全局旋转
        new_c2w[:3, :3] = R_add @ c2w[:3, :3]

        # 再把相机位置一起绕世界原点旋转，再加上你的平移滑杆偏移
        new_c2w[:3, 3] = R_add @ c2w[:3, 3] + np.array([tx, ty, tz])
        new_w2c = np.linalg.inv(new_c2w)

        # 叠加坐标轴
        out = draw_world_frame_axes(frame, cam_intr, new_w2c, axis_length=1)
        # 显示 c2w 与 w2c（只显示前三行）
        text = ["c2w:"] + [f"{row[0]:6.4f}, {row[1]:6.4f}, {row[2]:6.4f}, {row[3]:6.4f}"
                           for row in new_c2w[:4]] \
               + ["w2c:"] + [f"{row[0]:6.4f}, {row[1]:6.4f}, {row[2]:6.4f}, {row[3]:6.4f}"
                             for row in new_w2c[:4]]
        for i, line in enumerate(text):
            cv2.putText(out, line, (10, 25*(i+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        print(new_c2w)

        cv2.imshow(win, out[...,::-1])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
