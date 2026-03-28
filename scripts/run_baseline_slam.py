#!/usr/bin/env python3
"""
scripts/run_baseline_slam.py

Run the SLAM pipeline once with default (or JSON-loaded) hyperparameters
and print ATE evaluation results.

Usage:
    python scripts/run_baseline_slam.py \\
        --dataset /path/to/rgbd_dataset_freiburg2_pioneer_slam2 \\
        [--hparams configs/default_hparams.json] \\
        [--save-plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slam.slam_hparams import SlamHyperParams
from slam.slam_runner import run_slam_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline SLAM on a TUM RGB-D sequence.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the TUM sequence directory (e.g. rgbd_dataset_freiburg2_pioneer_slam2).",
    )
    parser.add_argument(
        "--hparams",
        type=Path,
        default=None,
        help="Path to a JSON file with hyperparameter overrides (optional).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to write output plots (default: <dataset>/slam_runs).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save trajectory and pose-graph plots to save-dir.",
    )
    parser.add_argument("--start", type=int, default=None, help="First frame index.")
    parser.add_argument("--end", type=int, default=None, help="Last frame index (inclusive).")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hparams = SlamHyperParams()
    if args.hparams is not None:
        overrides = json.loads(args.hparams.read_text())
        for k, v in overrides.items():
            if hasattr(hparams, k):
                setattr(hparams, k, type(getattr(hparams, k))(v))

    print("=" * 60)
    print("Running baseline SLAM")
    print(f"  Dataset : {args.dataset}")
    print(f"  Hparams : {hparams}")
    print("=" * 60)

    results = run_slam_with_config(
        dataset_root=args.dataset,
        hparams=hparams,
        start_idx=args.start,
        end_idx=args.end,
        max_frames=args.max_frames,
        save_dir=args.save_dir,
        save_plots=args.save_plots,
    )

    print("\n--- Results ---")
    for k, v in results.items():
        if k != "hparams":
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
