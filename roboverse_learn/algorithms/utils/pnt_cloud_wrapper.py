import gym
import numpy as np

from .pnt_cloud_getter import PntCloudGetter


class IsaaclabPointcloudWrapperDP(gym.Wrapper):
    def __init__(self, env, task_name: str, use_point_crop=True):
        super().__init__(env)
        self.task_name = task_name
        self.use_point_crop = use_point_crop
        self.num_envs = env.handler.num_envs
        # point cloud generator
        self.pc_generator = PntCloudGetter(task_name=task_name, num_envs=self.num_envs, use_point_crop=use_point_crop)

    def step(self, action):
        obs, reward, success, time_out, extras = self.env.step(action)
        try:
            rgb = obs["rgb"]
            depth = obs["depth"]
            cam_intr = obs["cam_intr"]
            cam_extr = obs["cam_extr"]
        except Exception as e:
            print("obs keys: ", obs.keys())
            raise KeyError("obs does not contain rgb, depth, cam_intr, cam_extr") from e
        pnt_cloud = self.pc_generator.get_point_cloud(rgb, depth, cam_intr, cam_extr)
        obs["point_cloud"] = pnt_cloud
        return obs, reward, success, time_out, extras

    def reset(self, states):
        obs, extras = self.env.reset(states)
        try:
            rgb = obs["rgb"]
            depth = obs["depth"]
            cam_intr = obs["cam_intr"]
            cam_extr = obs["cam_extr"]
        except Exception as e:
            print("obs keys: ", obs.keys())
            raise KeyError("obs does not contain rgb, depth, cam_intr, cam_extr") from e
        pnt_cloud = self.pc_generator.get_point_cloud(rgb, depth, cam_intr, cam_extr)
        obs["point_cloud"] = pnt_cloud
        return obs, extras
