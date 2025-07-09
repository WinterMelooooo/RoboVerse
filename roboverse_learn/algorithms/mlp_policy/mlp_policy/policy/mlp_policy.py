from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce
from mlp_policy.model.common.normalizer import LinearNormalizer
from mlp_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

from roboverse_learn.algorithms.mlp_policy.mlp_policy.policy.base_mlp_policy import BaseMLPPolicy


class MLPPolicy(BaseMLPPolicy):
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
        self.model = model
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.kwargs = kwargs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # print("!!obs_dict", obs_dict["head_cam"].shape)
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
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
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        nobs_features = self.obs_encoder(nobs)
        pred = self.model(nobs_features)
        target = nactions
        loss = F.mse_loss(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
