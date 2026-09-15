"""Gray-box one-step acceleration model and its learned residual."""

from __future__ import annotations

import torch
from torch import nn

from config import Config


class ResidualNet(nn.Module):
    """Map a 6W-dimensional feature vector to a three-axis residual."""

    def __init__(self, window_size: int = Config.WINDOW_SIZE) -> None:
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(6 * window_size, 128)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(128, 64)),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.utils.spectral_norm(nn.Linear(64, 3)),
        )
        self.output_scale = nn.Parameter(torch.ones(3))

    def forward(self, x_k: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x_k) * self.output_scale


class GrayBoxDynamicsModel(nn.Module):
    """Predict filtered acceleration from a nominal prior and MLP residual."""

    def __init__(self, window_size: int = Config.WINDOW_SIZE) -> None:
        super().__init__()
        self.residual_net = ResidualNet(window_size)

    def forward(
        self, x_k: torch.Tensor, cmd_delayed: torch.Tensor, dist_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual_net(x_k)
        return cmd_delayed + dist_k + residual, residual
