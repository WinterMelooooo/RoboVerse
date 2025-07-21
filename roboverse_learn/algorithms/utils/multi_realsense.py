import os
import random
from collections import defaultdict
import numpy as np
import torch
import pyrealsense2 as rs
import sys
sys.path.append("./")
from roboverse_learn.algorithms.utils.attr_dict import AttrDict
from diffusion_policy.common.pytorch_util import dict_apply
from PIL import Image

def gather_realsense_cameras(enable_rgb = True, enable_depth = True, **stream_kwargs):
    """
    Gather all available Realsense cameras.
    :param enable_rgb: Whether to enable RGB stream.
    :param enable_depth: Whether to enable Depth stream.
    :param stream_kwargs: Additional arguments for the streams.
    :return: List of RealsenseCamera objects.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    cameras = []
    if len(devices) == 0:
        print("未检测到任何 RealSense 设备。")
    else:
        # 假设你只连接了一个设备，或者你知道 serial number
        device = devices[0]
        sensors = device.query_sensors()

        # for sensor in sensors:
        #     print(f"\n传感器名称: {sensor.get_info(rs.camera_info.name)}")
        #     for profile in sensor.get_stream_profiles():
        #         vprofile = profile.as_video_stream_profile()
        #         fmt = vprofile.format()
        #         fmt_name = str(fmt).split('.')[-1]
        #         width = vprofile.width()
        #         height = vprofile.height()
        #         fps = vprofile.fps()
        #         stream_type = vprofile.stream_type()
        #         print(f"流类型: {stream_type}, 分辨率: {width}x{height}, 格式: {fmt_name}, 帧率: {fps}fps")

    for device in devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        camera = RealsenseCamera(serial_number, enable_rgb=enable_rgb, enable_depth=enable_depth, **stream_kwargs)
        cameras.append(camera)

    return cameras


class RealsenseCamera:
    def __init__(self, serial, enable_rgb=True, enable_depth=True, target_width=256, target_height=256, fps=30):
        self.serial_number = serial
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.target_width = target_width
        self.target_height = target_height

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)

        if self.enable_rgb:
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, fps)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self._started = False

    def open(self):
        if not self._started:
            self.profile = self.pipeline.start(self.config)
            self._started = True
            ds = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            cs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = ds.get_intrinsics()
            fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
            self.intrinsics = torch.tensor([[fx, 0, cx],
                                             [0, fy, cy],
                                             [0, 0, 1]], dtype=torch.float32)
            self.cam_pos = (1.27, 0, 0.93)  # hard coded camera position
            self.cam_look_at = (0, 0, 0)  # hard coded camera look at
            self.extrinsics = self.get_extrinsics(
                cam_pos=self.cam_pos,
                cam_look_at=self.cam_look_at,
            )
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

    def close(self):
        if self._started:
            self.pipeline.stop()
            self._started = False

    def read_camera(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        timestamp = aligned_frames.get_timestamp()
        depth = aligned_frames.get_depth_frame() if self.enable_depth else None
        color = aligned_frames.get_color_frame() if self.enable_rgb else None
        depth = np.asanyarray(depth.get_data()).copy() if depth is not None else None
        color = np.asanyarray(color.get_data()).copy() if color is not None else None
        color = color[..., ::-1]
        if depth is not None:
            depth = depth * self.depth_scale
        depth_min = depth.min() if depth is not None else None
        depth_max = depth.max() if depth is not None else None

        depth_img = (depth-depth.min()) / (depth.max() - depth.min()) if depth is not None else None

        color = self._center_crop_and_resize(Image.fromarray(color), self.target_width, self.target_height) if color is not None else None
        depth_img = self._center_crop_and_resize(Image.fromarray(depth_img), self.target_width, self.target_height) if depth_img is not None else None

        cam_intr = self.intrinsics
        cam_extr = self.extrinsics

        time_stamp_dict = {
            self.serial_number: timestamp
        }
        obs_dict = {
            "rgb": color,
            "depth": depth_img,
            #"timestamp": timestamp,
            "intrinsics": cam_intr,
            "extrinsics": cam_extr,
            #"depth_min": depth_min,
            #"depth_max": depth_max
        }
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert to AttrDict for easier access
        return obs_dict, time_stamp_dict

    def is_running(self):
        return self._started

    def set_trajectory_mode(self):
        """
        Set the camera to trajectory mode.
        This is a placeholder for any specific settings needed for trajectory mode.
        """
        # In this case, we assume trajectory mode is just the default mode.
        if not self._started:
            self.open()

    def start_recording(self, filepath):
        self.config.enable_record_to_file(filepath)
        if self._started:
            self.pipeline.stop()
            self._started = False

        self.open()

    def stop_recording(self):
        self.close()

    def get_intrinsics(self):
        return {self.serial_number: {"cameraMatrix": self.intrinsics}  }

    def get_extrinsics(self, cam_pos, cam_look_at):
        """
        Get the extrinsics matrix for the camera.
        :param cam_pos: Camera position in world coordinates.
        :param cam_look_at: Point in world coordinates that the camera is looking at.
        :return: Extrinsics matrix as a 4x4 tensor.
        """
        cam_pos = torch.tensor(cam_pos, device="cuda")
        cam_look_at = torch.tensor(cam_look_at, device="cuda")
        c2w = torch.zeros((4, 4), device="cuda")
        c2w[:3, 3] = cam_pos  # Set camera position
        z = cam_look_at - cam_pos  # Camera forward vector
        z = z / torch.norm(z)  # Normalize
        x = torch.tensor([0.0, 1.0, 0.0], device="cuda")  # Up vector
        y = torch.cross(z, x)
        y = y / torch.norm(y)  # Normalize
        c2w[:3, 0] = x  # Set camera right vector
        c2w[:3, 1] = y  # Set camera up vector
        c2w[:3, 2] = z  # Set camera forward vector
        c2w[3, 3] = 1.0  # Set homogeneous
        w2c = torch.linalg.inv(c2w)  # Inverse to get world to camera
        return w2c

    def _center_crop_and_resize(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        orig_w, orig_h = img.size
        target_ratio = target_width / target_height
        orig_ratio   = orig_w / orig_h

        if orig_ratio > target_ratio:
            new_h = orig_h
            new_w = int(target_ratio * new_h)
        else:
            new_w = orig_w
            new_h = int(new_w / target_ratio)

        # 计算中心裁剪区域
        left   = (orig_w - new_w) // 2
        top    = (orig_h - new_h) // 2
        right  = left + new_w
        bottom = top  + new_h
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped = img_cropped.resize((target_width, target_height), Image.LANCZOS)
        return np.array(img_cropped)


class MultiRealsenseWrapper:
    def __init__(self, camera_kwargs={}):
        # Open Cameras #
        rs_cameras = gather_realsense_cameras()
        self.camera_dict = {cam.serial_number: cam for cam in rs_cameras}

        # Launch Camera #
        self.set_trajectory_mode()

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        pass

    def disable_advanced_calibration(self):
        pass

    def set_calibration_mode(self, cam_id):
        pass

    def set_trajectory_mode(self):
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath):
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def read_cameras(self):
        full_obs_dict = {}
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for idx, cam_id in enumerate(all_cam_ids):
            # print(f"trying to read cam: {cam_id}")
            # if not self.camera_dict[cam_id].is_running():
            #     print(f"cam: {cam_id} not running!")
            #     continue
            data_dict, _ = self.camera_dict[cam_id].read_camera()
            #recursive_print_dic(data_dict, 0)
            #print("\n\n\n\n\n")
            data_dict = dict_apply(data_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
            full_obs_dict[f"camera{idx}"] = data_dict
        return full_obs_dict, None#full_timestamp_dict

    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.close()

    def __call__(self):
        obs_dict, _ = self.read_cameras()
        return obs_dict

def recursive_print_dic(dic, intendent = 0):
    for key, value in dic.items():
        pref = "\t" * intendent
        if isinstance(value, dict):
            print(f"{pref}{key}:")
            recursive_print_dic(value, intendent + 1)
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"{pref}{key}: {value.shape}")
        elif isinstance(value, (int, float, str)):
            print(f"{pref}{key}: {value}")
        elif hasattr(value, "__len__"):
            try:
                print(f"{pref}{key}: len = {len(value)}")
            except Exception:
                print(f"{pref}{key}: {value}")
        else:
            print(f"{pref}{key}: {value}")
