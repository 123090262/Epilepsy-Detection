"""Main epilepsy GAT network."""

from __future__ import annotations

import torch
import torch.nn as nn

from epilepsy.models.gat import (
    DEFAULT_CHANNEL_NAMES,
    ChannelGating,
    GatedFusion,
    PriorMatrixBuilder,
    SCGAT,
    TAGAT,
    compute_plv_batch,
)
from epilepsy.models.tcn import (
    ClassicalFeatureExtractor,
    FeatureExtractor,
    SpectralFeatureExtractor,
)


class ChannelAttentionPool(nn.Module):
    """Learn a per-sample weighted channel summary instead of plain averaging."""

    def __init__(self, hid_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, max(hid_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(hid_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class EpilepsyGATNet(nn.Module):
    """TCN + dual graph attention network for EEG epilepsy classification."""

    def __init__(
        self,
        channel_names: list[str] | tuple[str, ...] = DEFAULT_CHANNEL_NAMES,
        fs: int = 256,
        num_classes: int = 2,
        feature_dim: int = 128,
        hid_dim: int = 256,
        dropout: float = 0.35,
        graph_dropout: float = 0.25,
        graph_heads: int = 4,
        temporal_backbone: str = "lightweight",
        spectral_fusion: bool = True,
        classical_fusion: bool = False,
        classical_hidden_dim: int = 128,
        auxiliary_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.C = len(channel_names)
        self.c_dim = 64
        self.spectral_fusion = spectral_fusion
        self.classical_fusion = classical_fusion
        self.auxiliary_weight = auxiliary_weight

        self.prior_builder = PriorMatrixBuilder(channel_names)
        self.feat_extractor = FeatureExtractor(
            self.C, fs, feature_dim, backbone=temporal_backbone
        )
        if spectral_fusion:
            self.spectral_extractor = SpectralFeatureExtractor(fs, feature_dim)
            self.feature_gate = nn.Sequential(
                nn.Linear(2 * feature_dim, feature_dim), nn.Sigmoid()
            )
            self.feature_norm = nn.LayerNorm(feature_dim)
        if classical_fusion:
            self.classical_extractor = ClassicalFeatureExtractor(fs)
            self.classical_project = nn.Sequential(
                nn.LayerNorm(self.classical_extractor.num_features),
                nn.Linear(self.classical_extractor.num_features, feature_dim),
                nn.GELU(),
            )
            self.classical_feature_gate = nn.Sequential(
                nn.Linear(2 * feature_dim, feature_dim), nn.Sigmoid()
            )
            self.classical_feature_norm = nn.LayerNorm(feature_dim)
            self.classical_global = nn.Sequential(
                nn.LayerNorm(self.C * self.classical_extractor.num_features),
                nn.Linear(
                    self.C * self.classical_extractor.num_features,
                    classical_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(classical_hidden_dim, hid_dim),
                nn.GELU(),
            )
            self.classical_fuse = GatedFusion(hid_dim)
        self.tagat = TAGAT(
            feature_dim,
            hid_dim,
            c_dim=self.c_dim,
            dropout=graph_dropout,
            num_heads=graph_heads,
        )
        self.scgat = SCGAT(
            feature_dim,
            hid_dim,
            num_classes,
            c_dim=self.c_dim,
            dropout=graph_dropout,
            num_heads=graph_heads,
        )

        self.norm_t = nn.LayerNorm(hid_dim)
        self.norm_s = nn.LayerNorm(hid_dim)
        self.pool_t = ChannelAttentionPool(hid_dim)
        self.pool_s = ChannelAttentionPool(hid_dim)
        self.gate_fuse = GatedFusion(hid_dim)
        self.ch_gate = ChannelGating(self.C, hid_dim)
        self.fused_norm = nn.LayerNorm(hid_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hid_dim // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(hid_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B, C, L), got {tuple(x.shape)}")
        if x.size(1) != self.C:
            raise ValueError(f"Expected {self.C} channels, got {x.size(1)}")

        a_plv = compute_plv_batch(x)
        a_prior = self.prior_builder(a_plv)

        feat = self.feat_extractor(x)
        if self.spectral_fusion:
            spectral = self.spectral_extractor(x)
            gate = self.feature_gate(torch.cat((feat, spectral), dim=-1))
            feat = self.feature_norm(feat + gate * spectral)
        if self.classical_fusion:
            classical = self.classical_extractor(x)
            classical_projected = self.classical_project(classical)
            classical_gate = self.classical_feature_gate(
                torch.cat((feat, classical_projected), dim=-1)
            )
            feat = self.classical_feature_norm(
                feat + classical_gate * classical_projected
            )
        t_out = self.norm_t(self.tagat(feat))
        s_out, auxiliary_logits = self.scgat(feat, a_prior)
        s_out = self.norm_s(s_out)

        fused = self.gate_fuse(self.pool_t(t_out), self.pool_s(s_out))
        ch_gated = self.ch_gate(fused, s_out)
        fused = self.fused_norm(fused + ch_gated)
        if self.classical_fusion:
            classical_global = self.classical_global(classical.flatten(1))
            fused = self.fused_norm(self.classical_fuse(fused, classical_global))
        return self.classifier(fused) + self.auxiliary_weight * auxiliary_logits
