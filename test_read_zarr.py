import zarr
import imageio as iio
from roboverse_learn.algorithms.utils.visualizer import visualizer

zarr_path = "/home/ghr/yktang/RoboVerse/data_policy/LiberoPickButterFrankaL0_obs:joint_pos_act:joint_pos_100.zarr"
root = zarr.open(zarr_path, mode="r")

data = root["data/head_camera_pnt_cloud"][:]
print(data.shape)
visualizer.visualize_pointcloud(data[..., :3][5])
# import pdb; pdb.set_trace()
# img = root["data/head_camera"][:]
# rgb = img[5]
# iio.imwrite("test.png", rgb)


# depth = root["data/head_camera_depth"][:]
# depth_img = depth[5]
# iio.imwrite("test_depth.png", depth_img)
