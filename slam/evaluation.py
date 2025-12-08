#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 19:25:22 2025

@author: nitaishah
"""

"""
evaluation.py

Ground-truth evaluation utilities for TUM RGB-D SLAM:

- Associate estimated trajectory (timestamps + positions) with TUM groundtruth.
- Rigid (similarity) alignment using Umeyama.
- Compute Absolute Trajectory Error (ATE) RMSE.
"""

from typing import Dict, List, Tuple

import numpy as np


def umeyama_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama alignment: find similarity transform (s, R, t) such that:

        Y ≈ s * R @ X + t

    Args:
        X: (N, 3) estimated points.
        Y: (N, 3) ground truth points.
        with_scale: if True, estimate scale; otherwise, s = 1.

    Returns:
        s: scalar scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert X.shape == Y.shape
    n, dim = X.shape
    assert dim == 3

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    # Covariance matrix
    Sigma = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_X = (Xc**2).sum(axis=1).mean()
        s = np.trace(np.diag(D) @ S) / var_X
    else:
        s = 1.0

    t = mu_Y - s * R @ mu_X

    return float(s), R, t


def associate_to_groundtruth(
    est_timestamps: List[float],
    est_positions: List[np.ndarray],
    gt_dict: Dict[float, np.ndarray],
    max_time_diff: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate estimated trajectory to ground truth by nearest timestamp.

    Args:
        est_timestamps: list of estimated timestamps.
        est_positions: list of (3,) estimated positions (world frame).
        gt_dict: TUM groundtruth dict: timestamp -> [tx, ty, tz, qx, qy, qz, qw].
        max_time_diff: maximum allowed time difference in seconds.

    Returns:
        gt_xyz: (M, 3) matched GT positions.
        est_xyz: (M, 3) matched estimated positions, same order as gt_xyz.

        If M < 2, evaluation is not meaningful.
    """
    if len(est_timestamps) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Sort GT timestamps once
    gt_times = sorted(gt_dict.keys())
    gt_times_np = np.array(gt_times, dtype=np.float64)

    gt_xyz_list = []
    est_xyz_list = []

    for ts, pos in zip(est_timestamps, est_positions):
        # Find nearest GT time using searchsorted
        idx = np.searchsorted(gt_times_np, ts)
        candidates = []
        if idx > 0:
            candidates.append(gt_times_np[idx - 1])
        if idx < len(gt_times_np):
            candidates.append(gt_times_np[idx])

        if not candidates:
            continue

        # Pick the GT timestamp with minimum |dt|
        best_t = min(candidates, key=lambda t_c: abs(t_c - ts))
        if abs(best_t - ts) > max_time_diff:
            continue

        gt_val = gt_dict[best_t]
        gt_pos = np.array(gt_val[0:3], dtype=np.float64)

        gt_xyz_list.append(gt_pos)
        est_xyz_list.append(np.asarray(pos, dtype=np.float64))

    if len(gt_xyz_list) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    gt_xyz = np.vstack(gt_xyz_list)
    est_xyz = np.vstack(est_xyz_list)

    return gt_xyz, est_xyz


def compute_ate_rmse(
    gt_xyz: np.ndarray,
    est_xyz: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute ATE RMSE after rigid alignment (Umeyama) between GT and estimate.

    Args:
        gt_xyz: (N, 3) ground truth positions.
        est_xyz: (N, 3) estimated positions.
        with_scale: whether to estimate a scale factor.

    Returns:
        ate_rmse: root mean squared error (meters)
        ate_mean: mean absolute error (meters)
        ate_max: max absolute error (meters)
        scale: estimated scale factor
    """
    assert gt_xyz.shape == est_xyz.shape
    if gt_xyz.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), 1.0

    s, R, t = umeyama_alignment(est_xyz, gt_xyz, with_scale=with_scale)

    est_aligned = (s * (R @ est_xyz.T).T) + t[None, :]

    errors = np.linalg.norm(est_aligned - gt_xyz, axis=1)

    ate_rmse = float(np.sqrt(np.mean(errors**2)))
    ate_mean = float(np.mean(errors))
    ate_max = float(np.max(errors))

    return ate_rmse, ate_mean, ate_max, s


def evaluate_trajectory_against_gt(
    name: str,
    est_timestamps: List[float],
    est_positions: List[np.ndarray],
    gt_dict: Dict[float, np.ndarray],
    max_time_diff: float = 0.02,
    with_scale: bool = True,
) -> None:
    """
    High-level helper: associate a trajectory to GT and print ATE stats.

    Args:
        name: label for this trajectory (e.g., 'Raw VO', 'Optimized keyframes').
        est_timestamps: timestamps of estimated poses.
        est_positions: (3,) positions in world coordinates.
        gt_dict: TUM GT dict from TUMRGBDDataset.get_groundtruth_dict().
    """
    gt_xyz, est_xyz = associate_to_groundtruth(
        est_timestamps, est_positions, gt_dict, max_time_diff=max_time_diff
    )

    n = gt_xyz.shape[0]
    if n < 2:
        print(f"[Eval:{name}] Not enough matched poses for evaluation (matches={n}).")
        return

    ate_rmse, ate_mean, ate_max, scale = compute_ate_rmse(
        gt_xyz, est_xyz, with_scale=with_scale
    )

    print(f"\n[Eval:{name}] matched poses: {n}")
    print(f"[Eval:{name}] estimated scale factor: {scale:.4f}")
    print(f"[Eval:{name}] ATE RMSE: {ate_rmse:.4f} m")
    print(f"[Eval:{name}] ATE mean: {ate_mean:.4f} m")
    print(f"[Eval:{name}] ATE max:  {ate_max:.4f} m")
