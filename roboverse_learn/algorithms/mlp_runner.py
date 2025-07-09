from __future__ import annotations

from collections import deque

import dill
import hydra
import numpy as np
import torch
from metasim.cfg.policy import MLPPolicyCfg
from omegaconf import OmegaConf

from diffusion_policy import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

from .base_runner import PolicyRunner

try:
    from metasim.types import EnvState
except:
    pass


def print_dict_keys(d: dict, prefix: str = ""):
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        print(path)
        if isinstance(v, dict):
            print_dict_keys(v, path)


class MLPRunner(PolicyRunner):
    """Runner for a diffusion policy, loads in a workspace and policy from checkpoint, and overrides some of the
    PolicyCFG attributes to match how the policy was trained
    """

    def _init_policy(self, **kwargs):
        self.task_name = kwargs.get("task_name")
        payload = torch.load(open(kwargs["checkpoint_path"], "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace: RobotWorkspace = cls(cfg, output_dir=kwargs.get("output_dir", None))
        workspace.load_payload(
            payload, exclude_keys=["lr_scheduler"], include_keys=None
        )
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(self.device)
        policy.to(device)
        policy.eval()
        self.policy = policy
        self.yaml_cfg = cfg
        self.policy_cfg = MLPPolicyCfg()

        if "policy_runner" in cfg:
            obs_cfg = cfg.shape_meta.obs

            if "agent_pos" in obs_cfg:
                dim = obs_cfg.agent_pos.shape[0]
            elif "qpos" in obs_cfg:
                from roboverse_learn.algorithms.utils.transformpcd import ComposePCD

                dim = obs_cfg.qpos.shape[0]
                transform_pcd = hydra.utils.instantiate(
                    self.yaml_cfg.task.dataset.transform_pcd
                )
                self.transform_pcd = ComposePCD(transform_pcd)
            else:
                raise KeyError("shape_meta.obs: no agent_pos or qpos！")

            self.policy_cfg.obs_config.obs_dim = dim
            self.policy_cfg.obs_config.from_dict(cfg.policy_runner.obs)
            self.policy_cfg.action_config.from_dict(cfg.policy_runner.action)

            self.policy_cfg.action_config.action_dim = cfg.shape_meta.action.shape[0]

        self.policy_cfg.action_config.action_chunk_steps = cfg.n_action_steps
        self.policy_cfg.action_config.action_dim = cfg.shape_meta.action.shape[0]

        self.obs = deque(maxlen=cfg.n_obs_steps + 1)
        self.env = None

    def reset(self):
        self.obs.clear()
        super().reset()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def predict_action(self, observation):
        with torch.no_grad():
            action_chunk = (
                self.policy.predict_action(observation)["action_pred"]
                .detach()
                .to(torch.float32)
            )
            action_chunk = action_chunk.transpose(0, 1)
        return action_chunk

    def process_obs(self, obs: list[EnvState]):
        """
        Args:
            obs: dict{key: (N_env, value)}
        """
        obs = dict_apply(
            obs,
            lambda x: x.to(device=self.device) if isinstance(x, torch.Tensor) else x,
        )
        obs_dict = super().process_obs(obs)
        if "point_cloud" in self.yaml_cfg.task.shape_meta.obs.keys():
            obs_dict["point_cloud"] = obs["point_cloud"]
            print(f"Set PntCloud origin to robot root")
            from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.dataset.robot_pointcloud_dataset import (
                ROBOT_ROOT_STATES,
                transform_point_cloud,
            )

            obs_dict["point_cloud"] = (
                transform_point_cloud(
                    obs_dict["point_cloud"],
                    ROBOT_ROOT_STATES,
                    self.task_name,
                    self.policy.device,
                )
                .cpu()
                .numpy()
            )

        if "head_cam" in self.yaml_cfg.task.shape_meta.obs.keys() and (
            self.yaml_cfg.task.shape_meta.obs.head_cam.type == "rgbd"
            or self.yaml_cfg.task.shape_meta.obs.head_cam.type == "rgbd_resnet"
        ):
            depth = obs["depth"]  # (N_env, H, W, 1) [znear, zfar]
            if self.policy_cfg.obs_config.norm_image:
                depth = (depth - depth.min()) / (depth.max() - depth.min())

            depth = depth.permute(0, 3, 1, 2)  # (N_env, 1, H, W)
            assert depth.shape[1] == 1, (
                f"depth should be 1 channel, but got {depth.shape}"
            )
            obs_dict["head_cam"] = torch.cat([obs_dict["head_cam"], depth], dim=1)
            assert obs_dict["head_cam"].shape[1] == 4, (
                f"head_cam should be 4 channels, but got {obs_dict['head_cam'].shape}"
            )
        if "pcds" in self.yaml_cfg.task.shape_meta.obs.keys():
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

        if (
            "franka_panda_leftfinger_touch_sensor_pres"
            in self.yaml_cfg.task.shape_meta.obs.keys()
        ):
            # print(f"type obs:{type(obs)}")
            # import pprint
            # pprint.pprint(obs)
            for sensor_name, sensor_state in obs["sensors"].items():
                obs_dict[sensor_name + "_pres"] = sensor_state.force.to(self.device)

        return obs_dict
