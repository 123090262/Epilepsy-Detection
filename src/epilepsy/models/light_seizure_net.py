"""Compact multiscale temporal network for patient-specific EEG classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class SeparableResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 3 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=7,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class LightSeizureNet(nn.Module):
    """A small dilated CNN inspired by efficient patient-specific detectors."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.2,
        sample_rate: int = 256,
    ) -> None:
        super().__init__()
        width = max(32, min(hidden_dim, 128))
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, width, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.temporal = nn.Sequential(
            SeparableResidualBlock(width, dilation=1, dropout=dropout),
            SeparableResidualBlock(width, dilation=2, dropout=dropout),
            SeparableResidualBlock(width, dilation=4, dropout=dropout),
            SeparableResidualBlock(width, dilation=8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(width + 5 * num_channels, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        frequencies = torch.fft.rfftfreq(
            x.shape[-1], d=1.0 / self.sample_rate, device=x.device
        )
        power = torch.fft.rfft(x, dim=-1).abs().square()
        bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0))
        features = []
        for low, high in bands:
            mask = (frequencies >= low) & (frequencies < high)
            features.append(torch.log1p(power[..., mask].mean(dim=-1)))
        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.pool(self.temporal(self.stem(x))).flatten(1)
        return self.classifier(torch.cat((temporal, self.spectral_features(x)), dim=1))
