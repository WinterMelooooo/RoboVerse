import zarr
import numpy as np
save_dir = "/home/ghr/yktang/RoboVerse/data_policy/RealworldLiberoPickButterFrankaRealWorld_obs:joint_pos_act:joint_pos_48.zarr"
input_name = '/home/ghr/yktang/RoboVerse/data_policy/Backup_RealworldLiberoPickButterFrankaRealWorld_obs:joint_pos_act:joint_pos_48.zarr'
dataset_name = 'data/head_camera_pnt_cloud'
f = zarr.open(input_name)
raw = f[dataset_name]
print(raw.shape)
raw_data = raw[:]
bbox = {
    "min": [-100, -100, -100],
    "max": [100, 100, 100]
}


compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
zarr_root = zarr.group(save_dir)
zarr_data = zarr_root.create_group("data")
zarr_data.create_dataset(
                    "head_camera_pnt_cloud",
                    shape=(0, *raw_data.shape[1:]),
                    chunks=(batch_size, *raw_data.shape[1:]),
                    dtype=raw_data.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
zarr_data["head_camera_pnt_cloud"].append(raw_data)
