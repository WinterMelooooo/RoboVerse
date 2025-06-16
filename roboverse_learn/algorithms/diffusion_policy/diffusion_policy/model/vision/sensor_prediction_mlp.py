import torch
from torch import nn
from typing import List

class SensorPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, sensor_names: list):
        super(SensorPredictor, self).__init__()
        sensor_names = [name for name in sensor_names if name.endswith("_pred")]
        num_heads = len(sensor_names)
        self.sensor_names = sensor_names
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for head in range(num_heads)
        ])

    def forward(self, x):
        # Expect x to be of shape (B, input_dim)
        # If x is of shape (N, B, input_dim), avg pool over N
        if x.dim() == 3:
            x = x.mean(dim=0)
        x = self.relu(self.fc1(x))
        output = {}
        for i, head in enumerate(self.heads):
            output[self.sensor_names[i]] = head(x)
        return output


class TransSensorPredictor(nn.Module):
    """
    Args:
        embed_dim (int): dimension of token embeddings
        sensor_names (list): list of sensor names for output
        n_heads (int): number of attention heads
        ff_dim (int): dimension of feed-forward network
        num_layers (int): number of TransformerDecoder layers
        dropout (float): dropout rate
    """
    def __init__(self,
                 embed_dim: int,
                 output_dim: int,
                 sensor_names: list,
                 n_heads: int = 8,
                 ff_dim: int = 2048,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        sensor_names = [name for name in sensor_names if name.endswith("_pred")]
        # Trainable <SOS> token (1, 1, embed_dim)
        self.sos_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Output head to map decoder output -> tactile signal dimension
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, output_dim)
            for head in range(len(sensor_names))
        ])
        self.sensor_names = sensor_names

    def forward(self,
                fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused (Tensor): visual-tactile fused features x_t, shape (N,  B, embed_dim)

        Returns:
            Tensor: predicted tactile signal \hat h_{t+1}, shape (B, j_dim)
        """
        B = fused.shape[1]
        # Prepare target (Query): <SOS> repeated for batch
        tgt = self.sos_token.repeat(1, B, 1)
        # Decoder: masked self-attn not needed for single-step
        dec_out = self.decoder(tgt=tgt, memory=fused)
        # dec_out shape: (1, B, embed_dim) -> remove seq_dim
        dec_out = dec_out.squeeze(0) # (B, embed_dim)
        output = {}
        for i, head in enumerate(self.heads):
            output[self.sensor_names[i]] = head(dec_out)
        return output
