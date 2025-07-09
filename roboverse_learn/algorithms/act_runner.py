import os
import pickle
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import yaml
from act.policy import ACTPolicy
from act.workspace.robotworkspace import RobotWorkspace

from metasim.cfg.policy import ACTPolicyCfg
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from roboverse_learn.eval import use_dp3_pcd, use_pcd, use_rgbd, use_sensor, use_spUnet_pcd

from .base_runner import PolicyRunner


class ACTRunner(PolicyRunner):
    def _init_policy(self, **kwargs):
        self.task_name = kwargs.get("task_name")
        payload = torch.load(open(kwargs["checkpoint_path"], "rb"), pickle_module=dill)
        chkpt = Path(kwargs["checkpoint_path"])
        output_dir = str(chkpt.parent.parent)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace: RobotWorkspace = cls(cfg, output_dir=kwargs.get("output_dir", None))
        workspace.load_payload(payload, exclude_keys=["lr_scheduler"], include_keys=None)
        policy = workspace.policy

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()
        self.policy = policy
        self.yaml_cfg = cfg
        self.policy_cfg = ACTPolicyCfg()

        stats_path = os.path.join(output_dir, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        if "policy_runner" in cfg:
            self.policy_cfg.obs_config.from_dict(cfg.policy_runner.obs)
            self.policy_cfg.action_config.from_dict(cfg.policy_runner.action)
        else:
            raise KeyError("policy_runner not found in cfg")

    def pre_process(self, s_state):
        return (s_state - torch.tensor(self.stats["state_mean"], device=self.device)) / torch.tensor(
            self.stats["state_std"], device=self.device
        )

    def post_process(self, a):
        return a * torch.tensor(self.stats["action_std"], device=self.device) + torch.tensor(
            self.stats["action_mean"], device=self.device
        )

    def predict_action(self, observation):
        # state = self.pre_process(observation["agent_pos"]).cuda()
        # curr_image = observation["head_cam"].unsqueeze(1).cuda()
        observation = dict_apply(observation, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(self.device))
        with torch.no_grad():
            action_chunk = self.policy(observation)
        action = self.post_process(action_chunk)
        action = action[:, :, : self.policy_cfg.action_config.action_dim]
        return action.transpose(0, 1)  # Expects (action_chunk_steps, n_envs, action_dim)

    def process_obs(self, obs):
        """
        Args:
            obs: dict{key: (N_env, value)}
        """
        obs = dict_apply(
            obs,
            lambda x: x.to(device=self.device) if isinstance(x, torch.Tensor) else x,
        )
        obs_dict = super().process_obs(obs)
        if use_pcd(self.yaml_cfg):
            obs_dict["point_cloud"] = obs["point_cloud"]
            print(f"Set PntCloud origin to robot root")
            from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.dataset.robot_pointcloud_dataset import (
                ROBOT_ROOT_STATES,
                transform_point_cloud,
            )

            obs_dict["point_cloud"] = (
                transform_point_cloud(obs_dict["point_cloud"], ROBOT_ROOT_STATES, self.task_name, "cuda")
                .cpu()
                .numpy()
            )

        if use_rgbd(self.yaml_cfg):
            depth = obs["depth"]  # (N_env, H, W, 1) [znear, zfar]
            if self.policy_cfg.obs_config.norm_image:
                depth = (depth - depth.min()) / (depth.max() - depth.min())

            depth = depth.permute(0, 3, 1, 2)  # (N_env, 1, H, W)
            assert depth.shape[1] == 1, f"depth should be 1 channel, but got {depth.shape}"
            obs_dict["head_cam"] = torch.cat([obs_dict["head_cam"], depth], dim=1)
            assert obs_dict["head_cam"].shape[1] == 4, (
                f"head_cam should be 4 channels, but got {obs_dict['head_cam'].shape}"
            )
        if use_spUnet_pcd(self.yaml_cfg):
            pcds = obs["point_cloud"]  # (N_env, N_points, 6)
            qpos = obs_dict["agent_pos"]  # (N_env, 9)
            new_obs_dict = dict()
            new_obs_dict["pcds"] = []
            new_obs_dict["qpos"] = qpos
            for pcd in pcds:
                coords = pcd[:, :3].astype(np.float32)
                colors = pcd[:, 3:6].astype(np.float32)
                pcd_dict = self.transform_pcd({"coord": coords, "color": colors})
                # {
                #   'coord': Tensor[M,3],
                #   'grid_coord': Tensor[M,3],
                #   'feat': Tensor[M,F],
                #   'offset': Tensor[N_env]
                # }
                new_obs_dict["pcds"].append(pcd_dict)
            # new_obs_dict["pcds"] = torch.stack(new_obs_dict["pcds"], dim=0) # (N_env, Dict)
            obs_dict = new_obs_dict

        if use_sensor(self.yaml_cfg):
            # print(f"type obs:{type(obs)}")
            # import pprint
            # pprint.pprint(obs)
            for sensor_name, sensor_state in obs["sensors"].items():
                obs_dict[sensor_name + "_pres"] = sensor_state.force.to(self.device)

        return obs_dict
