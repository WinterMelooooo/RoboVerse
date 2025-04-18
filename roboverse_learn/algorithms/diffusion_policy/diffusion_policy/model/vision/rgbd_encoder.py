import torch
import torch.nn as nn


class DFormerFeatureExtractor(nn.Module):
    def __init__(self, dformer):
        super().__init__()
        self.backbone = dformer
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x, x_e=None):
        outs, _ = self.backbone(x, x_e)
        last = outs[-1]                # [B, C, H', W']
        pooled = self.global_pool(last)     # [B, C, 1, 1]
        return torch.flatten(pooled,1)           # [B, C]
