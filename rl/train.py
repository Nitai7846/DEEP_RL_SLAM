"""
rl/train.py

REINFORCE training loop for SLAM hyperparameter optimization.

Each episode is a single environment step (1-step MDP):
    obs    = normalized segment descriptor
    action = 6-D continuous hyperparameter action in [-1, 1]
    reward = -RMSE - λ_kf × KF_density - λ_lc × LC_density

The policy gradient loss is:
    L = -log π(a|s) × G

where G is the (unnormalized) episodic return.

Artifacts written to disk:
    <log_path>  — CSV training log (one row per episode).
    <save_path> — PyTorch state dict of the best policy by reward.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from .env import SlamHyperParamEnv
from .policy import PolicyNet


def train_reinforce(
    env: SlamHyperParamEnv,
    num_episodes: int = 100,
    gamma: float = 1.0,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> PolicyNet:
    """
    Train a Gaussian policy with REINFORCE on the SLAM hyperparameter environment.

    Args:
        env:          the ``SlamHyperParamEnv`` instance.
        num_episodes: number of training episodes.
        gamma:        discount factor (use 1.0 for 1-step MDPs).
        lr:           Adam learning rate.
        device:       ``'cpu'`` or ``'cuda'``.
        save_path:    where to save the best policy state dict (.pt).
        log_path:     where to write the CSV training log.

    Returns:
        The trained ``PolicyNet``.
    """
    save_path = save_path or Path("slam_rl_policy_best.pt")
    log_path = log_path or Path("slam_rl_train_log.csv")

    policy = PolicyNet(env.obs_dim, env.act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    best_reward = -float("inf")
    best_info = None
    log_exists = False

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        log_probs, rewards, info = [], [], None

        done = False
        while not done:
            action, log_prob = policy.sample_action(obs)
            obs, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute discounted returns
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Policy gradient loss
        log_probs_t = torch.stack(log_probs).to(device)
        loss = -(log_probs_t * returns_t).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = float(np.mean(rewards))
        hp = info["hparams"]

        # Logging
        _log_episode(log_path, log_exists, ep, avg_reward, info)
        log_exists = True

        print(
            f"[Ep {ep:04d}/{num_episodes}] reward={avg_reward:+.4f}  "
            f"rmse_kf={info['rmse_kf']:.3f}  "
            f"kf={info['num_kf']}  loops={info['num_loops']}  "
            f"trans={hp['trans_thresh']:.3f}  rot={hp['rot_thresh_deg']:.1f}°  "
            f"gap={hp['min_frame_gap']}  "
            f"lc_sep={hp['lc_min_frame_separation']}  "
            f"lc_inl={hp['lc_min_inliers']}  "
            f"pnp={hp['lc_pnp_reproj_thresh']:.2f}"
        )

        # Track best
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_info = {"episode": ep, "reward": avg_reward, **info}
            torch.save(policy.state_dict(), save_path)

        if ep % 10 == 0:
            print(f"  [checkpoint] best reward so far: {best_reward:+.4f}")

    print(f"\nTraining complete. Best policy saved to: {save_path}")

    if best_info:
        best_json = save_path.with_suffix(".best.json")
        with best_json.open("w") as f:
            # Ensure JSON serializability
            serializable = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in best_info.items()
                if k != "hparams"
            }
            serializable["hparams"] = best_info["hparams"]
            json.dump(serializable, f, indent=2)
        print(f"Best episode details saved to: {best_json}")

    return policy


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "episode", "reward", "rmse_kf", "num_kf", "num_loops", "num_frames",
    "trans_thresh", "rot_thresh_deg", "min_frame_gap",
    "lc_min_frame_separation", "lc_min_inliers", "lc_pnp_reproj_thresh",
]


def _log_episode(
    path: Path,
    already_exists: bool,
    ep: int,
    reward: float,
    info: dict,
) -> None:
    hp = info["hparams"]
    row = [
        ep, reward,
        info["rmse_kf"], info["num_kf"], info["num_loops"], info["num_frames"],
        hp["trans_thresh"], hp["rot_thresh_deg"], hp["min_frame_gap"],
        hp["lc_min_frame_separation"], hp["lc_min_inliers"], hp["lc_pnp_reproj_thresh"],
    ]
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not already_exists:
            writer.writerow(_CSV_FIELDS)
        writer.writerow(row)
