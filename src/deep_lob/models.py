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

class Chomp1d(nn.Module):
    """
    Remove extra padding on the right to keep the output length
    equal to the input length (causal convolution).
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Basic TCN block:
    - two dilated 1D convolutions
    - batch norm + ReLU + dropout
    - residual connection (with 1x1 conv if channels change)
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCNBackbone(nn.Module):
    """
    Stack of TemporalBlocks with increasing dilation:
    dilation = 1, 2, 4, ...
    """
    def __init__(
        self,
        num_features: int,
        num_channels=(32, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = num_features
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.out_channels = in_ch  # last channel size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, T)
        return self.network(x)


class TCNModel(nn.Module):
    """
    TCN-based model for LOB data.

    Input:  (batch, T, F)
    Output: (batch, 3) logits for classes {-1, 0, 1} (mapped to {0,1,2})
    """
    def __init__(self, num_features: int, num_classes: int = 3):
        super().__init__()
        self.tcn = TCNBackbone(num_features=num_features)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.tcn.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F) -> (batch, F, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)                  # (batch, C, T)
        x = self.global_pool(x).squeeze(-1)  # (batch, C)
        logits = self.fc(x)              # (batch, num_classes)
        return logits