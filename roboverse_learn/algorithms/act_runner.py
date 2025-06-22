import os
import pickle
from pathlib import Path

import dill
import hydra
import torch
import yaml
from metasim.cfg.policy import ACTPolicyCfg

from act.policy import ACTPolicy
from act.workspace.robotworkspace import RobotWorkspace

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
        workspace.load_payload(
            payload, exclude_keys=["lr_scheduler"], include_keys=None
        )
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
        return (
            s_state - torch.tensor(self.stats["state_mean"], device=self.device)
        ) / torch.tensor(self.stats["state_std"], device=self.device)

    def post_process(self, a):
        return a * torch.tensor(
            self.stats["action_std"], device=self.device
        ) + torch.tensor(self.stats["action_mean"], device=self.device)

    def predict_action(self, observation):
        state = self.pre_process(observation["agent_pos"]).cuda()
        curr_image = observation["head_cam"].unsqueeze(1).cuda()
        with torch.no_grad():
            action_chunk = self.policy(state, curr_image)
        action = self.post_process(action_chunk)
        action = action[:, :, : self.policy_cfg.action_config.action_dim]
        return action.transpose(
            0, 1
        )  # Expects (action_chunk_steps, n_envs, action_dim)
