"""
evaluation.py

Trajectory evaluation utilities for TUM RGB-D SLAM.

    umeyama_alignment           Similarity transform that best aligns two point clouds.
    associate_to_groundtruth    Nearest-timestamp matching of estimated → GT poses.
    compute_ate_rmse            ATE RMSE after Umeyama alignment.
    evaluate_trajectory_against_gt  High-level helper that prints ATE statistics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def umeyama_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama alignment: find (s, R, t) such that Y ≈ s * R @ X + t.

    Args:
        X:          (N, 3) estimated positions.
        Y:          (N, 3) ground-truth positions.
        with_scale: estimate a scale factor if True, otherwise s = 1.

    Returns:
        s: scalar scale factor.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]

    mu_X, mu_Y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_X, Y - mu_Y

    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_X = (Xc ** 2).sum(1).mean()
        s = float(np.trace(np.diag(D) @ S) / var_X)
    else:
        s = 1.0

    t = mu_Y - s * R @ mu_X
    return s, R, t


def associate_to_groundtruth(
    est_timestamps: List[float],
    est_positions: List[np.ndarray],
    gt_dict: Dict[float, np.ndarray],
    max_time_diff: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate an estimated trajectory to ground truth by nearest timestamp.

    Args:
        est_timestamps: timestamps of estimated poses.
        est_positions:  list of (3,) estimated camera positions.
        gt_dict:        TUM GT dict: timestamp → [tx, ty, tz, qx, qy, qz, qw].
        max_time_diff:  maximum allowed |Δt| in seconds.

    Returns:
        gt_xyz:  (M, 3) matched GT positions.
        est_xyz: (M, 3) matched estimated positions (same ordering).
    """
    if not est_timestamps:
        return np.zeros((0, 3)), np.zeros((0, 3))

    gt_times = np.array(sorted(gt_dict.keys()), dtype=np.float64)
    gt_xyz_list, est_xyz_list = [], []

    for ts, pos in zip(est_timestamps, est_positions):
        idx = np.searchsorted(gt_times, ts)
        candidates = gt_times[max(0, idx - 1) : min(len(gt_times), idx + 1)]
        if len(candidates) == 0:
            continue
        best_t = candidates[np.argmin(np.abs(candidates - ts))]
        if abs(best_t - ts) > max_time_diff:
            continue
        gt_xyz_list.append(np.asarray(gt_dict[best_t][:3], dtype=np.float64))
        est_xyz_list.append(np.asarray(pos, dtype=np.float64))

    if not gt_xyz_list:
        return np.zeros((0, 3)), np.zeros((0, 3))

    return np.vstack(gt_xyz_list), np.vstack(est_xyz_list)


def compute_ate_rmse(
    gt_xyz: np.ndarray,
    est_xyz: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute ATE RMSE after Umeyama alignment.

    Returns:
        ate_rmse, ate_mean, ate_max, scale
    """
    assert gt_xyz.shape == est_xyz.shape
    if gt_xyz.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), 1.0

    s, R, t = umeyama_alignment(est_xyz, gt_xyz, with_scale=with_scale)
    est_aligned = (s * (R @ est_xyz.T).T) + t

    errors = np.linalg.norm(est_aligned - gt_xyz, axis=1)
    return (
        float(np.sqrt(np.mean(errors ** 2))),
        float(np.mean(errors)),
        float(np.max(errors)),
        s,
    )


def evaluate_trajectory_against_gt(
    name: str,
    est_timestamps: List[float],
    est_positions: List[np.ndarray],
    gt_dict: Dict[float, np.ndarray],
    max_time_diff: float = 0.02,
    with_scale: bool = True,
) -> None:
    """
    Associate a trajectory to GT and print ATE statistics.

    Args:
        name:           label for this trajectory (e.g. 'VO', 'KF-optimized').
        est_timestamps: timestamps of estimated poses.
        est_positions:  list of (3,) positions in world coordinates.
        gt_dict:        ground-truth dict from ``TUMRGBDDataset.get_groundtruth_dict()``.
    """
    gt_xyz, est_xyz = associate_to_groundtruth(
        est_timestamps, est_positions, gt_dict, max_time_diff
    )

    if gt_xyz.shape[0] < 2:
        print(f"[Eval:{name}] Not enough matched poses ({gt_xyz.shape[0]}).")
        return

    ate_rmse, ate_mean, ate_max, scale = compute_ate_rmse(gt_xyz, est_xyz, with_scale)

    print(f"\n[Eval:{name}] matched poses : {gt_xyz.shape[0]}")
    print(f"[Eval:{name}] scale factor  : {scale:.4f}")
    print(f"[Eval:{name}] ATE RMSE      : {ate_rmse:.4f} m")
    print(f"[Eval:{name}] ATE mean      : {ate_mean:.4f} m")
    print(f"[Eval:{name}] ATE max       : {ate_max:.4f} m")
