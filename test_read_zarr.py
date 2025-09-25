import zarr
import imageio as iio
from roboverse_learn.algorithms.utils.visualizer import visualizer

zarr_path = "/datasets/startrack/current/yanbinghan/RoboVerse-Store/data_policy/RealworldPour_franka_dp3_rost3r_40_0914.zarr"
root = zarr.open(zarr_path, mode="r")

data = root["data/head_camera"][:]
print(data.shape)
import pdb; pdb.set_trace()
visualizer.visualize_pointcloud(data[..., :3][5])
# import pdb; pdb.set_trace()
# img = root["data/head_camera"][:]
# rgb = img[5]
# iio.imwrite("test.png", rgb)


# depth = root["data/head_camera_depth"][:]
# depth_img = depth[5]
# iio.imwrite("test_depth.png", depth_img)
