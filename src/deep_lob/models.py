"""
Minimal DeepLOB-style model:
- 1D convolutions over time
- simple Inception-style blocks
- LSTM on top
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock1D(nn.Module):
    """
    Simple Inception-like block for 1D time series.

    Parallel convs with different kernel sizes over the time dimension.

    Note: The effective out_channels is 3 * base_c, where base_c = out_channels // 3.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # base channels per branch
        base_c = out_channels // 3
        self.base_c = base_c
        self.out_channels = base_c * 3  # actual number of channels after concat

        self.conv1 = nn.Conv1d(in_channels, base_c, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, base_c, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, base_c, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        y1 = self.conv1(x)
        y3 = self.conv3(x)
        y5 = self.conv5(x)
        y = torch.cat([y1, y3, y5], dim=1)  # (B, 3*base_c, T)
        y = self.bn(y)
        return F.relu(y)


class DeepLOBModel(nn.Module):
    """
    Very small DeepLOB-style network.

    Input: (batch, T, F)
    Output: logits for 3 classes (up / flat / down)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        hidden_channels: int = 30,   # use 30 to divide cleanly into 3 branches
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()

        # First inception: in_channels = num_features
        self.inception1 = InceptionBlock1D(num_features, hidden_channels)
        # After inception1, channels = inception1.out_channels
        c1 = self.inception1.out_channels

        # Second inception: in_channels = c1
        self.inception2 = InceptionBlock1D(c1, hidden_channels)
        c2 = self.inception2.out_channels

        # LSTM takes features = c2
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)

        x = self.inception1(x)  # (B, C1, T)
        x = self.inception2(x)  # (B, C2, T)

        # Back to (B, T, C2) for LSTM: swap time and channels
        x = x.transpose(1, 2)  # (B, T, C2)

        out, _ = self.lstm(x)  # (B, T, H)
        last = out[:, -1, :]   # (B, H)

        logits = self.fc(last) # (B, num_classes)
        return logits