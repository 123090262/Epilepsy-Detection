"""Temporal convolutional network components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgMaxPool1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool1d(x, 1)
        mx = F.adaptive_max_pool1d(x, 1)
        return torch.cat([avg, mx], dim=1)


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.3)

        p2 = (kernel_size // 2) * dilation
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size,
            stride=1,
            padding=p2,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)

        if shortcut.shape[-1] != out.shape[-1]:
            diff = shortcut.shape[-1] - out.shape[-1]
            if diff > 0:
                shortcut = shortcut[..., :-diff]
            else:
                out = out[..., : shortcut.shape[-1]]

        return self.relu(out + shortcut)


class EnhancedResTCN(nn.Module):
    def __init__(self, fs: int = 256, feature_dim: int = 128) -> None:
        super().__init__()
        k1 = int(50 * (fs / 100))
        s1 = int(5 * (fs / 100))

        self.layer1 = TCNResidualBlock(
            1, 64, k1, stride=s1, dilation=1, padding=k1 // 2
        )
        self.layer2 = TCNResidualBlock(
            64, 128, 5, stride=2, dilation=2, padding=4
        )
        self.layer3 = TCNResidualBlock(
            128, 256, 3, stride=2, dilation=4, padding=4
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x))


class LightweightTemporalEncoder(nn.Module):
    """Small dilated depthwise CNN for per-channel EEG feature extraction."""

    def __init__(
        self,
        fs: int = 256,
        feature_dim: int = 128,
        width: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        width = max(16, width)
        kernel = max(7, int(0.06 * fs) | 1)
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.blocks = nn.Sequential(
            SeparableTemporalBlock(width, dilation=1, dropout=dropout),
            SeparableTemporalBlock(width, dilation=2, dropout=dropout),
            SeparableTemporalBlock(width, dilation=4, dropout=dropout),
        )
        self.project = nn.Sequential(
            AvgMaxPool1d(),
            nn.Flatten(),
            nn.Linear(2 * width, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.blocks(self.stem(x)))


class SeparableTemporalBlock(nn.Module):
    """Residual depthwise-separable temporal block."""

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


class FeatureExtractor(nn.Module):
    """Extract a TCN feature vector independently for each EEG channel."""

    def __init__(
        self,
        num_channels: int,
        fs: int = 256,
        feature_dim: int = 128,
        backbone: str = "lightweight",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        backbone = backbone.lower()
        if backbone in {"lightweight", "light", "depthwise"}:
            self.tcn = LightweightTemporalEncoder(fs=fs, feature_dim=feature_dim)
        elif backbone in {"enhanced_tcn", "tcn", "res_tcn"}:
            self.tcn = EnhancedResTCN(fs=fs, feature_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported temporal backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        out = self.tcn(x.reshape(batch_size * channels, 1, length))
        return out.reshape(batch_size, channels, -1)


class SpectralFeatureExtractor(nn.Module):
    """Differentiable log-bandpower features for the five canonical EEG bands."""

    def __init__(self, fs: int = 256, feature_dim: int = 128) -> None:
        super().__init__()
        self.fs = fs
        self.bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0))
        self.project = nn.Sequential(
            nn.LayerNorm(len(self.bands)),
            nn.Linear(len(self.bands), feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.abs().square() / max(x.size(-1), 1)
        freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / self.fs).to(x.device)
        bandpower = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)
            bandpower.append(power[..., mask].mean(dim=-1))
        features = torch.log1p(torch.stack(bandpower, dim=-1))
        return self.project(features)


class ClassicalFeatureExtractor(nn.Module):
    """SVM-style per-channel temporal statistics and log-bandpower features."""

    def __init__(self, fs: int = 256) -> None:
        super().__init__()
        self.fs = fs
        self.bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0))

    @property
    def num_features(self) -> int:
        return 4 + len(self.bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B, C, L), got {tuple(x.shape)}")

        temporal = [
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            torch.sqrt(x.square().mean(dim=-1).clamp_min(1e-8)),
            x.diff(dim=-1).abs().mean(dim=-1),
        ]
        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.abs().square() / max(x.size(-1), 1)
        freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / self.fs).to(x.device)
        bandpower = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)
            bandpower.append(torch.log1p(power[..., mask].mean(dim=-1)))
        return torch.stack(temporal + bandpower, dim=-1)
