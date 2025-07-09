import copy
import math
import time
from typing import Dict

import pytorch3d.ops as torch3d_ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, reduce
from mlp_policy.common.pytorch_util import dict_apply
from mlp_policy.model.common.normalizer import LinearNormalizer

# from mlp_policy.common.model_util import print_params
from mlp_policy.model.vision.dp3_encoder import DP3Encoder
from mlp_policy.policy.base_mlp_policy import BaseMLPPolicy
from termcolor import cprint


class DP3(BaseMLPPolicy):
    def __init__(
        self,
        shape_meta: dict,
        model,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(
            f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}",
            "yellow",
        )
        cprint(
            f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}",
            "yellow",
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.kwargs = kwargs

    # ========= inference  ============

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        nobs_features = self.obs_encoder(nobs)
        naction_pred = self.model(nobs_features)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action_pred = action_pred.unsqueeze(1)  # (B, 1, action_dim)
        result = {
            "action_pred": action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        nobs_features = self.obs_encoder(nobs)
        pred = self.model(nobs_features)
        target = nactions
        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        return loss  # , loss_dict
