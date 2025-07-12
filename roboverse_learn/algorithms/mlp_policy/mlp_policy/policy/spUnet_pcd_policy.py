"""
Reference:
- https://github.com/real-stanford/mlp_policy
"""

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from mlp_policy.common.pytorch_util import dict_apply
from mlp_policy.model.common.normalizer import LinearNormalizer
from mlp_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from mlp_policy.policy.base_mlp_policy import BaseMLPPolicy


class spUnetPcdPolicy(BaseMLPPolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        model,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.kwargs = kwargs

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        assert "obs" in obs_dict or "pcds" in obs_dict or "point_cloud" in obs_dict, (
            f"obs or pcd or point_cloud not found!, keys:{obs_dict.keys()}"
        )
        if "obs" not in obs_dict:
            pcds = None
            if "pcds" in obs_dict:
                pcds = obs_dict.pop("pcds")
            elif "point_cloud" in obs_dict:
                pcds = obs_dict.pop("point_cloud")
            nobs = self.normalizer.normalize(obs_dict)
        else:
            pcds = None
            if "pcds" in obs_dict["obs"]:
                pcds = obs_dict["obs"].pop("pcds")
            elif "point_cloud" in obs_dict["obs"]:
                pcds = obs_dict["obs"].pop("point_cloud")
            nobs = self.normalizer.normalize(obs_dict["obs"])

        if pcds is not None:
            pcds[0] = dict_apply(pcds[0], lambda x: x.to(device=self.device))
            nobs["pcds"] = pcds
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
        pcds = None
        if "pcds" in batch["obs"]:
            pcds = batch["obs"].pop("pcds")
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if pcds is not None:
            nobs["pcds"] = pcds
        nobs_features = self.obs_encoder(nobs)
        pred = self.model(nobs_features)
        target = nactions
        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
