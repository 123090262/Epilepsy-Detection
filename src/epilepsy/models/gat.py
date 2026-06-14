"""Graph attention network components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHANNEL_NAMES = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]


class PriorMatrixBuilder(nn.Module):
    """Build a learnable prior adjacency matrix from geometry and PLV."""

    _EEG_COORDS = {
        "FP1": (-0.30, 0.95, 0.10),
        "FP2": (0.30, 0.95, 0.10),
        "F7": (-0.71, 0.50, 0.50),
        "F8": (0.71, 0.50, 0.50),
        "F3": (-0.45, 0.60, 0.65),
        "F4": (0.45, 0.60, 0.65),
        "FZ": (0.00, 0.60, 0.80),
        "T3": (-0.95, 0.00, 0.30),
        "T4": (0.95, 0.00, 0.30),
        "T7": (-0.95, 0.00, 0.30),
        "T8": (0.95, 0.00, 0.30),
        "C3": (-0.71, 0.00, 0.71),
        "C4": (0.71, 0.00, 0.71),
        "CZ": (0.00, 0.00, 1.00),
        "T5": (-0.71, -0.50, 0.50),
        "T6": (0.71, -0.50, 0.50),
        "P7": (-0.71, -0.50, 0.50),
        "P8": (0.71, -0.50, 0.50),
        "P3": (-0.45, -0.60, 0.65),
        "P4": (0.45, -0.60, 0.65),
        "PZ": (0.00, -0.60, 0.80),
        "O1": (-0.30, -0.95, 0.10),
        "O2": (0.30, -0.95, 0.10),
        "OZ": (0.00, -1.00, 0.00),
        "FT9": (-1.00, 0.25, 0.10),
        "FT10": (1.00, 0.25, 0.10),
        "A1": (-1.00, -0.10, -0.10),
        "A2": (1.00, -0.10, -0.10),
    }

    def __init__(
        self,
        channel_names: list[str] | tuple[str, ...],
        sigma: float = 1.0,
        beta: float = 10.0,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.C = len(channel_names)
        self.beta = beta
        self.tau = tau

        pos = torch.tensor([self._resolve(n) for n in channel_names], dtype=torch.float32)
        dist = torch.cdist(pos, pos, p=2)
        a_geo = torch.exp(-(dist**2) / (sigma**2))
        self.register_buffer("A_geo", a_geo)

        self.weight_fusion = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.E = nn.Parameter(torch.randn(self.C, self.C) * 0.01)

    def _resolve(self, name: str) -> list[float]:
        upper = name.upper().replace(" ", "")
        if upper in self._EEG_COORDS:
            return list(self._EEG_COORDS[upper])

        clean = upper.replace("EEG", "").strip().lstrip("-")
        if "-" in clean:
            coords = []
            for part in clean.split("-")[:2]:
                coord = self._EEG_COORDS.get(part)
                if coord is None:
                    for key, value in self._EEG_COORDS.items():
                        if part.startswith(key) or key.startswith(part):
                            coord = value
                            break
                if coord is not None:
                    coords.append(coord)
            if len(coords) == 2:
                return [(coords[0][i] + coords[1][i]) / 2 for i in range(3)]
            if len(coords) == 1:
                return list(coords[0])

        return [0.0, 0.0, 0.0]

    def forward(self, A_plv: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.weight_fusion, dim=0)
        a0 = weights[0] * self.A_geo.unsqueeze(0) + weights[1] * A_plv
        a_sparse = F.softplus((self.E + self.E.T) / 2).unsqueeze(0)
        a_init = a0 * a_sparse
        a_init = (a_init + a_init.transpose(1, 2)) / 2
        mask = 1.0 - torch.eye(self.C, device=a_init.device).unsqueeze(0)
        a_init = a_init * mask
        return torch.sigmoid(self.beta * (a_init - self.tau))


class GraphAttentionLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, c_dim: int = 64, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_dim + c_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.c_dim = c_dim
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        prior_adj: torch.Tensor | None = None,
        prior_lambda: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        wh = self.W(h)
        batch_size, nodes, dim = wh.shape

        zi = wh.unsqueeze(2).expand(-1, nodes, nodes, -1)
        zj = wh.unsqueeze(1).expand(-1, nodes, nodes, -1)

        if c is None:
            c = torch.zeros(batch_size, 1, 1, self.c_dim, device=wh.device)
        else:
            if c.dim() == 2:
                c = c.unsqueeze(1).unsqueeze(1)
            c = c.expand(batch_size, nodes, nodes, self.c_dim)

        a_input = torch.cat([zi, zj, c], dim=-1)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1), 0.2)

        if prior_adj is not None:
            if prior_adj.dim() == 2:
                prior_adj = prior_adj.unsqueeze(0).expand(batch_size, -1, -1)
            e = e + prior_lambda * torch.log(prior_adj.clamp(min=eps))

        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
            e = e.masked_fill(adj == 0, -1e9)

        alpha = self.dropout(F.softmax(e, dim=-1))
        return F.elu(torch.bmm(alpha, wh))


class TAGAT(nn.Module):
    """Single-segment temporal-aware graph attention branch."""

    def __init__(
        self, feature_dim: int, hid_dim: int, c_dim: int = 64, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.c_proj = nn.Sequential(nn.Linear(feature_dim, c_dim), nn.ReLU(inplace=True))
        self.gat1 = GraphAttentionLayer(feature_dim, hid_dim, c_dim=c_dim, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hid_dim, hid_dim, c_dim=c_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c_proj(x.mean(dim=1))
        nodes = x.size(1)
        adj = torch.ones(nodes, nodes, device=x.device)
        x = self.norm1(self.gat1(x, adj=adj, c=c))
        return self.norm2(x + self.gat2(x, adj=adj, c=c))


class SCGAT(nn.Module):
    """Spatial conditional graph attention branch with prior adjacency."""

    def __init__(
        self,
        feature_dim: int,
        hid_dim: int,
        num_classes: int = 2,
        c_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gat1 = GraphAttentionLayer(feature_dim, hid_dim, c_dim=c_dim, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hid_dim, hid_dim, c_dim=c_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.aux_fc = nn.Linear(hid_dim, num_classes)
        self.cond_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, c_dim),
        )

    def forward(self, x: torch.Tensor, adj_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, nodes = x.size(0), x.size(1)
        c = self.cond_mlp(x.mean(dim=1))
        adj_full = torch.ones(nodes, nodes, device=x.device)

        if adj_prior.dim() == 2:
            adj_prior = adj_prior.unsqueeze(0).expand(batch_size, -1, -1)

        x = self.norm1(
            self.gat1(x, adj=adj_full, c=c, prior_adj=adj_prior, prior_lambda=1.0)
        )
        x = self.norm2(
            x + self.gat2(x, adj=adj_full, c=c, prior_adj=adj_prior, prior_lambda=1.0)
        )
        aux_logits = self.aux_fc(x.mean(dim=1))
        return x, aux_logits


class GatedFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * dim, dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.fc(torch.cat([x1, x2], dim=-1)))
        return z * x1 + (1 - z) * x2


class ChannelGating(nn.Module):
    def __init__(self, num_channels: int, hid_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hid_dim, num_channels), nn.Sigmoid())

    def forward(self, fused: torch.Tensor, ch_feats: torch.Tensor) -> torch.Tensor:
        weights = self.gate(fused).unsqueeze(-1)
        return (ch_feats * weights).sum(dim=1)


def compute_plv_batch(x: torch.Tensor) -> torch.Tensor:
    """Compute a batch of Phase Locking Value matrices.

    Args:
        x: EEG tensor with shape `(B, C, T)`.

    Returns:
        PLV tensor with shape `(B, C, C)` and values in `[0, 1]`.
    """

    spectrum = torch.fft.fft(x, dim=-1)
    length = x.size(-1)
    hilbert_mask = torch.zeros(length, dtype=x.dtype, device=x.device)
    hilbert_mask[0] = 1
    if length % 2 == 0:
        hilbert_mask[length // 2] = 1
        hilbert_mask[1 : length // 2] = 2
    else:
        hilbert_mask[1 : (length + 1) // 2] = 2
    analytic = torch.fft.ifft(spectrum * hilbert_mask, dim=-1)
    unit_phase = analytic / analytic.abs().clamp(min=1e-8)
    return (
        torch.bmm(unit_phase, unit_phase.conj().transpose(1, 2)).abs() / length
    ).to(dtype=x.dtype)
