import torch
from torch import nn
from typing import List

class SensorPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, sensor_names: list):
        super(SensorPredictor, self).__init__()
        num_heads = len(sensor_names)
        self.sensor_names = sensor_names
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for head in range(num_heads)
        ])

    def forward(self, x):
        x = self.relu(self.fc1(x))
        output = {}
        for i, head in enumerate(self.heads):
            output[self.sensor_names[i]] = head(x)
        return output
