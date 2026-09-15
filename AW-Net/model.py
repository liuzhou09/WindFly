import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),  # 32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),  # 64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.physics_proj = nn.Linear(4, 192)
        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.act = nn.LeakyReLU(0.05)
        with torch.no_grad():
            self.v_proj.weight.mul_(0.5)
            self.fc.weight.mul_(0.01)

    def reset(self):
        """The recurrent state is passed explicitly between forward calls."""
        pass

    def forward(self, x: torch.Tensor, v, dist_pred, a_boundary_target, hx=None):
        physics_feat = torch.cat([dist_pred, a_boundary_target], dim=-1)
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v) + self.physics_proj(physics_feat))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx
