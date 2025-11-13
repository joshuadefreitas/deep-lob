"""
Model architectures for DeepLOB.

Here we will define:
- CNN-based models
- CNN + LSTM hybrids
- Temporal convolutional networks (TCN)
"""

import torch
import torch.nn as nn


class SimpleCNNLOB(nn.Module):
    """
    Very first baseline CNN model for LOB data.

    Expected input shape: (batch_size, channels, timesteps, levels)
    """

    def __init__(self, n_channels: int, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
