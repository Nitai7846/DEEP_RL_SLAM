"""
slam_runner.py

Top-level SLAM pipeline: VO front-end → keyframe selection →
loop-closure detection → pose-graph optimization → ATE evaluation.

Primary entry point: ``run_slam_with_config()``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .camera import freiburg2_camera
from .dataset import TUMRGBDDataset
from .evaluation import evaluate_trajectory_against_gt
from .frame import Frame
from .loop_closure import LoopClosureDetector, LoopClosureParams
from .map_management import Map
from .slam_hparams import SlamHyperParams
from .visualization import plot_pose_graph_xz, plot_trajectory_xz, plot_trajectory_xz_with_gt
from .vo_frontend import VisualOdometry, VOParams


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def rotation_angle_deg(R_rel: np.ndarray) -> float:
    """Return the rotation angle in degrees encoded by a 3 × 3 matrix."""
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def umeyama_align(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama alignment: find (s, R, t) such that dst ≈ s * R @ src + t.

    Args:
        src:        (N, 3) source points (estimated).
        dst:        (N, 3) target points (ground truth).
        with_scale: estimate a scale factor if True.

    Returns:
        s (float), R (3 × 3), t (3,)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    src_T, dst_T = src.T, dst.T
    mu_src = src_T.mean(1, keepdims=True)
    mu_dst = dst_T.mean(1, keepdims=True)

    src_d = src_T - mu_src
    dst_d = dst_T - mu_dst

    Sigma = (dst_d @ src_d.T) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_d ** 2) / n
        s = float(np.sum(D * np.diag(S))) / float(var_src + 1e-12)
    else:
        s = 1.0

    t = (mu_dst - s * R @ mu_src).ravel()
    return s, R, t


def _build_lc_params(hp: SlamHyperParams) -> LoopClosureParams:
    return LoopClosureParams(
        min_frame_separation=hp.lc_min_frame_separation,
        min_candidate_matches=hp.lc_min_candidate_matches,
        max_candidates=hp.lc_max_candidates,
        min_inliers=hp.lc_min_inliers,
        pnp_reproj_thresh=hp.lc_pnp_reproj_thresh,
    )


def _eval_against_gt(
    name: str,
    timestamps: List[float],
    positions: List[np.ndarray],
    gt_dict: Dict[float, Any],
) -> Optional[Dict[str, Any]]:
    """
    Run Umeyama-aligned ATE evaluation and return a details dict, or None on failure.
    """
    gt_times = np.array(sorted(gt_dict.keys()), dtype=np.float64)
    gt_pos = np.array(
        [np.asarray(gt_dict[t])[:3] for t in sorted(gt_dict.keys())], dtype=np.float64
    )

    est_times = np.array(timestamps, dtype=np.float64)
    est_pos = np.array(positions, dtype=np.float64)

    matched_est, matched_gt = [], []
    for t_est, p_est in zip(est_times, est_pos):
        idx = int(np.argmin(np.abs(gt_times - t_est)))
        if abs(gt_times[idx] - t_est) <= 0.02:
            matched_est.append(p_est)
            matched_gt.append(gt_pos[idx])

    if len(matched_est) < 2:
        print(f"[Eval:{name}] Not enough matched poses.")
        return None

    matched_est_arr = np.array(matched_est)
    matched_gt_arr = np.array(matched_gt)

    s, R, t = umeyama_align(matched_est_arr, matched_gt_arr, with_scale=False)
    aligned = (s * (R @ matched_est_arr.T).T) + t
    errors = aligned - matched_gt_arr
    err_norm = np.linalg.norm(errors, axis=1)
    rmse = float(np.sqrt(np.mean(err_norm ** 2)))

    print(f"[Eval:{name}] matched={len(matched_est)}  ATE RMSE={rmse:.4f} m")

    return {
        "matched_est": aligned,
        "matched_gt": matched_gt_arr,
        "errors": errors,
        "err_norm": err_norm,
        "scale": s,
    }


# ---------------------------------------------------------------------------
# Main SLAM pipeline
# ---------------------------------------------------------------------------

def run_slam_with_config(
    dataset_root: Path,
    hparams: SlamHyperParams,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    max_frames: Optional[int] = None,
    save_dir: Optional[Path] = None,
    save_plots: bool = False,
) -> Dict[str, Any]:
    """
    Run one complete SLAM pass and return summary metrics.

    Args:
        dataset_root: path to the TUM sequence directory.
        hparams:      hyperparameter config (see ``SlamHyperParams``).
        start_idx:    first frame index (inclusive, default = 0).
        end_idx:      last frame index (inclusive, default = last frame).
        max_frames:   optional cap on the number of frames to process.
        save_dir:     directory for output plots and logs.
        save_plots:   whether to save trajectory / pose-graph plots.

    Returns:
        dict with keys:
            ``kf_rmse``, ``vo_rmse``, ``num_keyframes``,
            ``num_loops``, ``num_frames``, ``hparams``.
    """
    dataset_root = Path(dataset_root)
    save_dir = save_dir or dataset_root / "slam_runs"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TUMRGBDDataset(root_dir=dataset_root, max_time_diff=0.02, depth_scale=5000.0)
    cam = freiburg2_camera()
    gt_dict = dataset.get_groundtruth_dict()

    n_total = len(dataset)
    i0 = 0 if start_idx is None else max(0, start_idx)
    i1 = n_total if end_idx is None else min(n_total, end_idx + 1)
    if max_frames is not None:
        i1 = min(i1, i0 + max_frames)

    if i1 - i0 <= 1:
        return {
            "kf_rmse": float("nan"),
            "vo_rmse": float("nan"),
            "num_keyframes": 0,
            "num_loops": 0,
            "num_frames": i1 - i0,
            "hparams": hparams.to_dict(),
        }

    vo = VisualOdometry(cam, VOParams())
    slam_map = Map()
    lc_detector = LoopClosureDetector(_build_lc_params(hparams))

    trans_thresh = hparams.trans_thresh
    rot_thresh_deg = hparams.rot_thresh_deg
    min_frame_gap = hparams.min_frame_gap
    max_dt_gap = hparams.max_dt_gap

    frames_all: List[Frame] = []
    centers: List[np.ndarray] = []
    timestamps: List[float] = []
    vo_success_count = 0
    last_kf_frame_id = -1
    num_loops_added = 0
    prev_frame: Optional[Frame] = None

    for idx in range(i0, i1):
        data = dataset[idx]
        frame = Frame(
            id=idx,
            timestamp=data.timestamp,
            rgb=data.rgb,
            depth=data.depth,
            camera=cam,
        )

        if prev_frame is None:
            vo.process_first_frame(frame)
            slam_map.add_keyframe(frame)
            last_kf_frame_id = frame.id
        else:
            if frame.timestamp - prev_frame.timestamp > max_dt_gap:
                print(
                    f"[SLAM] Timestamp gap {frame.timestamp - prev_frame.timestamp:.3f} s "
                    f"> max_dt_gap={max_dt_gap}. Stopping at frame {idx}."
                )
                break

            success, _ = vo.process_frame(prev_frame, frame)
            if not success:
                frame.set_pose(prev_frame.get_pose())
            else:
                vo_success_count += 1

            T_w_c = frame.get_pose()

            if idx - last_kf_frame_id >= min_frame_gap:
                last_kf = slam_map.get_last_keyframe()
                if last_kf is not None:
                    T_rel = np.linalg.inv(last_kf.get_pose()) @ T_w_c
                    if (
                        np.linalg.norm(T_rel[:3, 3]) > trans_thresh
                        or rotation_angle_deg(T_rel[:3, :3]) > rot_thresh_deg
                    ):
                        slam_map.add_keyframe(frame)
                        last_kf_frame_id = frame.id

                        if len(slam_map.keyframes) > 3:
                            for kf_id, score in lc_detector.find_candidates(
                                frame, slam_map.keyframes
                            ):
                                lc_ok, T_i_j, _ = lc_detector.verify_candidate(
                                    current_kf=frame,
                                    candidate_kf=slam_map.keyframes[kf_id],
                                )
                                if lc_ok:
                                    slam_map.pose_graph.add_edge(
                                        i=kf_id,
                                        j=frame.id,
                                        T_i_j=T_i_j,
                                        edge_type="loop",
                                    )
                                    num_loops_added += 1

        frames_all.append(frame)
        centers.append(frame.camera_center())
        timestamps.append(frame.timestamp)
        prev_frame = frame

    print(
        f"[SLAM] Done: {len(centers)} frames, "
        f"VO success={vo_success_count}, "
        f"KFs={len(slam_map.keyframes)}, "
        f"loops={num_loops_added}"
    )

    if len(centers) <= 1:
        return {
            "kf_rmse": float("nan"),
            "vo_rmse": float("nan"),
            "num_keyframes": len(slam_map.keyframes),
            "num_loops": num_loops_added,
            "num_frames": len(centers),
            "hparams": hparams.to_dict(),
        }

    # Pose-graph optimization
    slam_map.pose_graph.optimize(max_iterations=5)
    slam_map.update_keyframes_from_pose_graph()

    kf_centers = [slam_map.keyframes[k].camera_center() for k in sorted(slam_map.keyframes)]
    kf_timestamps = [slam_map.keyframes[k].timestamp for k in sorted(slam_map.keyframes)]

    vo_eval = _eval_against_gt("VO", timestamps, centers, gt_dict)
    kf_eval = _eval_against_gt("KF", kf_timestamps, kf_centers, gt_dict)

    vo_rmse = float(np.sqrt(np.mean(vo_eval["err_norm"] ** 2))) if vo_eval else float("nan")
    kf_rmse = float(np.sqrt(np.mean(kf_eval["err_norm"] ** 2))) if kf_eval else float("nan")

    if save_plots:
        plot_trajectory_xz(centers, title="VO trajectory", show=False,
                           save_path=str(save_dir / "vo_xz.png"))
        plot_trajectory_xz(kf_centers, title="KF trajectory", show=False,
                           save_path=str(save_dir / "kf_xz.png"))
        if vo_eval:
            plot_trajectory_xz_with_gt(vo_eval["matched_est"], vo_eval["matched_gt"],
                                       title="VO vs GT", show=False,
                                       save_path=str(save_dir / "vo_vs_gt_xz.png"))
        if kf_eval:
            plot_trajectory_xz_with_gt(kf_eval["matched_est"], kf_eval["matched_gt"],
                                       title="KF vs GT", show=False,
                                       save_path=str(save_dir / "kf_vs_gt_xz.png"))
        plot_pose_graph_xz(slam_map.pose_graph, title="Pose graph", show=False,
                           save_path=str(save_dir / "pose_graph_xz.png"))

    return {
        "kf_rmse": kf_rmse,
        "vo_rmse": vo_rmse,
        "num_keyframes": len(slam_map.keyframes),
        "num_loops": num_loops_added,
        "num_frames": len(centers),
        "hparams": hparams.to_dict(),
    }
