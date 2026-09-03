"""Paper-faithful pairwise trajectory reachability metric head."""

from __future__ import annotations

import torch
from torch import nn


def pair_features(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """Build TRM's ordered pair feature ``[zi, zj, zi-zj, |zi-zj|]``."""
    if z_i.shape != z_j.shape:
        raise ValueError(
            f"TRM inputs must have identical shapes, got {z_i.shape} and {z_j.shape}"
        )
    if z_i.ndim < 1:
        raise ValueError("TRM inputs must have a latent dimension")
    if z_i.dtype != z_j.dtype:
        raise ValueError(
            f"TRM inputs must have identical dtypes, got {z_i.dtype} and {z_j.dtype}"
        )
    if z_i.device != z_j.device:
        raise ValueError(
            f"TRM inputs must share a device, got {z_i.device} and {z_j.device}"
        )
    delta = z_i - z_j
    return torch.cat((z_i, z_j, delta, delta.abs()), dim=-1)


class TRMHead(nn.Module):
    """Ordered pairwise scalar head from TRM v2."""

    def __init__(self, latent_dim: int = 192, hidden_dim: int = 256) -> None:
        super().__init__()
        latent_dim = int(latent_dim)
        hidden_dim = int(hidden_dim)
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.shape and z_i.shape[-1] != self.latent_dim:
            raise ValueError(
                f"TRM expected latent dim {self.latent_dim}, got {z_i.shape[-1]}"
            )
        return self.net(pair_features(z_i, z_j)).squeeze(-1)

    def architecture(self) -> dict[str, object]:
        return {
            "feature": ["z_i", "z_j", "z_i-z_j", "abs(z_i-z_j)"],
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": 2,
            "activation": "SiLU",
            "output": "Softplus scalar",
        }
