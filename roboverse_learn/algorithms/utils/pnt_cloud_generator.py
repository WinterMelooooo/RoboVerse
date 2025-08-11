# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
from typing import List

import numpy as np
import open3d as o3d
from PIL import Image as PIL_Image

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    """
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    """

    # This function is lifted directly from scipy source code
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [
        x2 - y2 - z2 + w2,
        2 * (xy - zw),
        2 * (xz + yw),
        2 * (xy + zw),
        -x2 + y2 - z2 + w2,
        2 * (yz - xw),
        2 * (xz - yw),
        2 * (yz + xw),
        -x2 - y2 + z2 + w2,
    ]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


"""
Generates numpy transformation matrix from position list len(3) and
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""


def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


#
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""


class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """

    def __init__(self, cam_names: List, img_size=256):
        super(PointCloudGenerator, self).__init__()

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_names = cam_names

    def generateCroppedPointCloud(
        self,
        rgb,
        depth,
        cam_intr,
        cam_extr,
        save_img_dir=None,
        device_id=0,
        debug=False,
    ):
        od_cammat = cammat2o3d(cam_intr, self.img_width, self.img_height)
        od_depth = o3d.geometry.Image(depth)

        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
        c2w = np.linalg.inv(cam_extr)
        transformed_cloud = o3d_cloud.transform(c2w)
        # get numpy array of point cloud, (position, color)
        combined_cloud_points = np.asarray(transformed_cloud.points)
        # color is automatically normalized to [0,1] by open3d
        H, W, _ = rgb.shape
        N = len(combined_cloud_points)  # number of points
        idx = np.arange(N, dtype=np.int32)  # (N,)
        u = idx // W
        v = idx % W
        uv = np.stack([u, v], axis=1)  # (N,2)
        # combined_cloud_colors = np.asarray(combined_cloud.colors)  # Get the colors, ranging [0,1].
        combined_cloud_colors = rgb.reshape(-1, 3)  # range [0, 255]
        if not isinstance(combined_cloud_colors, np.ndarray):
            combined_cloud_colors = combined_cloud_colors.cpu().numpy()
        depth_map = depth[:, :, 0] if depth.shape[-1] == 1 else depth[...]
        mask = depth_map > 0  # bool, (H, W)
        flat_mask = mask.reshape(-1)  # bool, (H*W,)
        combined_cloud_colors = combined_cloud_colors[flat_mask]
        # —— 5. 计算每个有效像素的 (u, v) ——
        idxs = np.nonzero(flat_mask)[0]  # (M,)
        us = idxs // W  # 行 (u)
        vs = idxs % W  # 列 (v)
        uv = np.stack([us, vs], axis=1)  # (M, 2)
        combined_cloud = np.concatenate(
            (combined_cloud_points, combined_cloud_colors, uv), axis=1
        )
        if debug:
            # print(f"RGB shape: ({H}, {W})")
            rows = combined_cloud[:, 6].astype(np.int64)  # (N,)
            cols = combined_cloud[:, 7].astype(np.int64)  # (N,)
            pixel_colors = rgb[rows, cols, :]  # (N,3)

            # 再把 combined_cloud 里存的颜色取出来 (N,3)
            stored_colors = combined_cloud[:, 3:6]  # (N,3)

            # 逐行比较是否完全相等，axis=1 后返回 (N,) 的布尔数组
            equal_mask = np.all(pixel_colors == stored_colors, axis=1)  # (N,)
            if not np.all(equal_mask):
                print("Warning: Not all colors match the original RGB image!")
                self.plot_rgb_comparison(rgb, combined_cloud)
                raise ValueError("Colors do not match the original RGB image!")
        return combined_cloud, depth

    def plot_rgb_comparison(self, rgb_original, combined_cloud):
        """
        Plots the original RGB image and a reconstructed RGB image based on combined_cloud colors.
        """
        import matplotlib.pyplot as plt

        # Extract dimensions
        H, W, _ = rgb_original.shape

        # Initialize a blank reconstructed image
        reconstructed = np.zeros_like(rgb_original)

        # Extract rows, cols, and stored colors from combined_cloud
        rows = combined_cloud[:, 6].astype(np.int64)
        cols = combined_cloud[:, 7].astype(np.int64)
        stored_colors = combined_cloud[:, 3:6].astype(np.uint8)

        # Assign stored colors to reconstructed image at the specified (row, col) positions
        reconstructed[rows, cols] = stored_colors

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_original.astype(np.uint8))
        axes[0].set_title("Original RGB Image")
        axes[0].axis("off")

        axes[1].imshow(reconstructed)
        axes[1].set_title("Reconstructed from combined_cloud")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True, device_id=0):
        rendered_images = self.sim.render(
            self.img_width,
            self.img_height,
            camera_name=camera_name,
            depth=capture_depth,
            device_id=device_id,
        )
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)

            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)
            return img, depth_convert
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img / img.max() * 255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + "/" + filename + ".jpg")
