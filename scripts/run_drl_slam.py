#!/usr/bin/env python3
"""
scripts/run_drl_slam.py

Train the RL agent to optimize SLAM hyperparameters on a TUM RGB-D sequence,
then evaluate the best-found configuration.

Usage:
    python scripts/run_drl_slam.py \\
        --dataset /path/to/rgbd_dataset_freiburg2_pioneer_slam2 \\
        [--episodes 100] \\
        [--lr 1e-3] \\
        [--save-dir outputs/drl]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.env import SlamHyperParamEnv
from rl.train import train_reinforce
from slam.slam_hparams import SlamHyperParams
from slam.slam_runner import run_slam_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DRL hyperparameter search for SLAM on a TUM RGB-D sequence."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the TUM sequence directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of RL training episodes (default: 100).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--lambda-kf",
        type=float,
        default=0.05,
        help="Reward penalty weight for keyframe density.",
    )
    parser.add_argument(
        "--lambda-lc",
        type=float,
        default=0.05,
        help="Reward penalty weight for loop-closure density.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/drl"),
        help="Directory for policy checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DRL SLAM hyperparameter search")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Device   : {args.device}")
    print("=" * 60)

    env = SlamHyperParamEnv(
        dataset_root=str(args.dataset),
        lambda_kf=args.lambda_kf,
        lambda_lc=args.lambda_lc,
    )

    policy = train_reinforce(
        env=env,
        num_episodes=args.episodes,
        lr=args.lr,
        device=args.device,
        save_path=args.save_dir / "slam_rl_policy_best.pt",
        log_path=args.save_dir / "slam_rl_train_log.csv",
    )

    # Load best hyperparameters and run final evaluation
    best_json = args.save_dir / "slam_rl_policy_best.best.json"
    if best_json.exists():
        import json

        best = json.loads(best_json.read_text())
        hp_dict = best.get("hparams", {})
        hparams = SlamHyperParams()
        for k, v in hp_dict.items():
            if hasattr(hparams, k):
                setattr(hparams, k, type(getattr(hparams, k))(v))

        print("\n" + "=" * 60)
        print("Final evaluation with best DRL hyperparameters")
        print(f"  {hparams}")
        print("=" * 60)

        results = run_slam_with_config(
            dataset_root=args.dataset,
            hparams=hparams,
            save_dir=args.save_dir / "best_run",
            save_plots=True,
        )
        print("\n--- Best-run results ---")
        for k, v in results.items():
            if k != "hparams":
                print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
