import gym
import numpy as np

try:
    from metasim.types import EnvState
except:
    pass

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
        obs: list[EnvState]
        obs, reward, success, time_out, extras = self.env.step(action)
        for envstate in obs:
            cam_name = next(iter(envstate["cameras"].keys()))
            rgb = envstate["cameras"][cam_name]["rgb"]
            depth = envstate["cameras"][cam_name]["depth"]
            cam_intr = envstate["cameras"][cam_name]["intr"]
            cam_extr = envstate["cameras"][cam_name]["extr"]
            pnt_cloud = self.pc_generator.get_point_cloud(rgb, depth, cam_intr, cam_extr)
            envstate["cameras"][cam_name]["point_cloud"] = pnt_cloud
        return obs, reward, success, time_out, extras

    def reset(self, states):
        obs, extras = self.env.reset(states)
        for envstate in obs:
            cam_name = next(iter(envstate["cameras"].keys()))
            rgb = envstate["cameras"][cam_name]["rgb"]
            depth = envstate["cameras"][cam_name]["depth"]
            cam_intr = envstate["cameras"][cam_name]["intr"]
            cam_extr = envstate["cameras"][cam_name]["extr"]
            pnt_cloud = self.pc_generator.get_point_cloud(rgb, depth, cam_intr, cam_extr)
            envstate["cameras"][cam_name]["point_cloud"] = pnt_cloud
        return obs, extras
