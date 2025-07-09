import torch
import torch.nn as nn


class DFormerFeatureExtractor(nn.Module):
    def __init__(self, dformer, device):
        super().__init__()
        self.backbone = dformer
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.device = device

    def forward(self, x, x_e=None):
        outs = self.backbone(x, x_e)
        last = outs[-1]                # [B, C, H', W']
        pooled = self.global_pool(last)     # [B, C, 1, 1]
        flat = torch.flatten(pooled,1)           # [B, C]
        assert len(flat.shape) == 2 and flat.shape[0] == x.shape[0], f"flat shape: {flat.shape}, x shape: {x.shape}"
        return flat