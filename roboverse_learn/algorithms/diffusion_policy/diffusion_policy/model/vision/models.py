import torch
import torch.nn as nn

class AttentionPoolingHead(nn.Module):
    def __init__(self, num_tokens: int):
        super().__init__()
        self.token_weights = nn.Parameter(torch.randn(num_tokens))

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: Tensor of shape (N, B, D)
        Returns:
            pooled: Tensor of shape (B, D)
        """
        weights = torch.softmax(self.token_weights, dim=0)
        weights = weights.view(-1, 1, 1)
        pooled = (fused * weights).sum(dim=0)
        return pooled
