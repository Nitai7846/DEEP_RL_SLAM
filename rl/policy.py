"""
rl/policy.py

Gaussian policy network for continuous action spaces.

Architecture:
    obs → Linear(obs_dim, hidden) → ReLU → Linear(hidden, hidden) → ReLU
        → mu_head(hidden, act_dim) + tanh    (mean action, clipped to [-1, 1])
        → log_std  (learnable per-dimension parameter)

The distribution is a diagonal Gaussian; ``sample_action`` draws a
reparameterized sample and returns both the clipped action and its log-prob.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    Gaussian policy: obs → (mean_action, log_std).

    Args:
        obs_dim:    dimensionality of the observation vector.
        act_dim:    dimensionality of the action vector.
        hidden_dim: number of hidden units per layer.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, obs_dim) observation tensor.

        Returns:
            mu:      (B, act_dim) mean actions in [-1, 1].
            log_std: (B, act_dim) log standard deviations.
        """
        x = self.net(obs)
        mu = torch.tanh(self.mu_head(x))
        return mu, self.log_std.expand_as(mu)

    def sample_action(
        self, obs_np: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Draw a reparameterized action sample and compute its log-probability.

        Args:
            obs_np: (obs_dim,) observation as a NumPy array.

        Returns:
            action_np: (act_dim,) action clipped to [-1, 1].
            log_prob:  scalar tensor (used by REINFORCE).
        """
        obs = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(dim=-1)

        return torch.clamp(a, -1.0, 1.0).detach().cpu().numpy()[0], log_prob.squeeze(0)
