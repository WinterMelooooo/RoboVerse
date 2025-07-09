from typing import List

import torch
from torch import nn


def get_action_regression_head(input_dim: int, hidden_dim: int, action_dim: int) -> nn.Module:
    """
    Get a regression head for action prediction.

    Args:
        hidden_dim (int): The dimension of the hidden layer.
        action_dim (int): The dimension of the action space.
        input_dim (int): The input dimension, typically the output of an encoder.

    Returns:
        nn.Module: A mlp that maps from hidden_dim to action_dim.
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
    )
    return mlp


def get_residual_action_regression_head(
    input_dim: int, hidden_dims: List[int], action_dim: int, dropout=0.1, use_residual: bool = True
) -> nn.Module:
    """
    Get a residual regression head for action prediction.

    Args:
        hidden_dim (int): The dimension of the hidden layer.
        action_dim (int): The dimension of the action space.
        input_dim (int): The input dimension, typically the output of an encoder.

    Returns:
        nn.Module: A mlp that maps from hidden_dim to action_dim and adds it to the input.
    """
    return ResidualActionRegressionHead(input_dim, hidden_dims, action_dim, dropout=dropout, use_residual=use_residual)


class ResidualActionRegressionHead(nn.Module):
    """
    A residual regression head for action prediction.
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], action_dim: int, dropout: float = 0.1, use_residual: bool = True
    ):
        super().__init__()
        blocks = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(Block(current_dim, hidden_dim, dropout, use_residual=use_residual))
            current_dim = hidden_dim
        blocks.append(nn.Linear(current_dim, action_dim))
        self.mlp = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual regression head.

        Args:
            x (torch.Tensor): Input tensor, typically the output of an encoder.

        Returns:
            torch.Tensor: Output tensor with the same shape as input, plus the action prediction.
        """
        return self.mlp(x)


class Block(nn.Module):
    """
    A simple block with a linear layer and a ReLU activation.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if use_residual:
            self.shortcut = (
                nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))
                if input_dim != output_dim
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_x = self.dropout(self.relu(self.linear(x)))
        if hasattr(self, "shortcut"):
            new_x += self.shortcut(x)
        return new_x
