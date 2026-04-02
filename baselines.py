"""
Baseline architectures for comparison with CAWL.
All share the same interface: forward(obs, ctx=None) → action.
"""

import torch
import torch.nn as nn


class DenseMLP_BC(nn.Module):
    """Standard dense MLP BC. The averaging problem in its purest form."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs, ctx=None):
        return self.net(obs)


class ConcatMLP_BC(nn.Module):
    """Spectral context concatenated to input (v5 method)."""
    def __init__(self, obs_dim, act_dim, ctx_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + ctx_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),             nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs, ctx=None):
        x = torch.cat([obs, ctx], dim=-1) if (
            ctx is not None and ctx.shape[-1] > 0) else obs
        return self.net(x)


class FiLM_BC(nn.Module):
    """
    FiLM-conditioned MLP.
    Context modulates hidden representations via (scale, shift) per layer.
    This is proper conditioning — context changes how obs is processed,
    not what obs is.
    """
    def __init__(self, obs_dim, act_dim, ctx_dim, hidden=256):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, hidden)
        self.drop1  = nn.Dropout(0.1)
        self.layer2 = nn.Linear(hidden, hidden)
        self.drop2  = nn.Dropout(0.1)
        self.output = nn.Linear(hidden, act_dim)
        # FiLM generators: ctx → (gamma, beta) for each layer
        self.film1  = nn.Linear(ctx_dim, hidden * 2)
        self.film2  = nn.Linear(ctx_dim, hidden * 2)

    def forward(self, obs, ctx=None):
        h = self.layer1(obs)
        if ctx is not None and ctx.shape[-1] > 0:
            g1, b1 = self.film1(ctx).chunk(2, dim=-1)
            h = h * (1 + g1) + b1
        h = self.drop1(torch.relu(h))

        h = self.layer2(h)
        if ctx is not None and ctx.shape[-1] > 0:
            g2, b2 = self.film2(ctx).chunk(2, dim=-1)
            h = h * (1 + g2) + b2
        h = self.drop2(torch.relu(h))

        return self.output(h)
