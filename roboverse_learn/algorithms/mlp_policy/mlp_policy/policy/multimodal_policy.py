from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, reduce
from mlp_policy.common.pytorch_util import dict_apply
from mlp_policy.model.common.normalizer import LinearNormalizer
from mlp_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from mlp_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from mlp_policy.model.vision.multimodal_encoder import MultiModalEncoder
from mlp_policy.policy.base_mlp_policy import BaseMLPPolicy


class MultiModalPolicy(BaseMLPPolicy):
    def __init__(self, shape_meta: dict, obs_encoder: MultiModalEncoder, model):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.normalizer = LinearNormalizer()
        self.obs_encoder = obs_encoder
        self.model = model
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # print("!!obs_dict", obs_dict["head_cam"].shape)
        # normalize input
        pnt_cloud_spUnet = None
        if "obs" in obs_dict and "point_cloud" in obs_dict["obs"]:
            if isinstance(obs_dict["obs"]["point_cloud"], Dict):
                pnt_cloud_spUnet = obs_dict["obs"].pop("point_cloud")
        elif "point_cloud" in obs_dict:
            if isinstance(obs_dict["point_cloud"], Dict):
                pnt_cloud_spUnet = obs_dict.pop("point_cloud")
        nobs = self.normalizer.normalize(obs_dict)

        if pnt_cloud_spUnet is not None:
            nobs["point_cloud"] = pnt_cloud_spUnet
        # print("!!To", To)
        # print(this_nobs["head_cam"].shape, this_nobs["agent_pos"].shape)
        # for key, value in this_nobs.items():
        #    print(f"After Key: {key}, Value shape: {value.shape}")
        nobs_features = self.obs_encoder(nobs)
        naction_pred = self.model(nobs_features)  # (B, action_dim)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action_pred = action_pred.unsqueeze(1)  # (B, 1, action_dim)
        result = {"action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        pnt_cloud_spUnet = None
        if isinstance(batch["obs"]["point_cloud"], Dict):
            pnt_cloud_spUnet = batch["obs"].pop("point_cloud")
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if pnt_cloud_spUnet is not None:
            nobs["point_cloud"] = pnt_cloud_spUnet
        nobs_features = self.obs_encoder(nobs)
        pred = self.model(nobs_features)
        target = nactions
        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
