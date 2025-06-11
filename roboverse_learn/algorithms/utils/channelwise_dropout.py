import torch
import torch.nn as nn


class ChannelWiseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: [B, D]
        if len(x.shape) != 2:
            raise ValueError(f"Expected input shape [B, D], got {x.shape}")
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(1, x.size(1), device=x.device) > self.p).float() / (1 - self.p)
        return x * mask  # broadcast 到 [B, D]
