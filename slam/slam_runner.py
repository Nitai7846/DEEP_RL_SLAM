#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 01:17:38 2025

@author: nitaishah
"""

# slam/slam_runner.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import math
import numpy as np

from slam.dataset import TUMRGBDDataset
from slam.camera import freiburg2_camera
from slam.frame import Frame
from slam.vo_frontend import VisualOdometry, VOParams
from slam.map_management import Map
from slam.loop_closure import LoopClosureDetector, LoopClosureParams
from slam.visualization import (
    plot_trajectory_xz,
    plot_pose_graph_xz,
    plot_trajectory_xz_with_gt,
)

from slam.slam_hparams import SlamHyperParams


# -------------------------------------------------------------
# Small helpers (copied from your previous script)
# -------------------------------------------------------------
def rotation_angle_deg(R_rel: np.ndarray) -> float:
    """
    Compute rotation angle in degrees from a relative rotation matrix R_rel.
    """
    trace = float(np.trace(R_rel))
    cos_theta = max(min((trace - 1.0) / 2.0, 1.0), -1.0)
    theta = math.acos(cos_theta)
    return math.degrees(theta)


def umeyama_align(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama alignment: find similarity transform (s, R, t) that best aligns

        dst ≈ s * R @ src + t

    where src, dst are (N, 3).

    Returns:
        s (float), R (3x3), t (3,)
    """
    assert src.shape == dst.shape
    assert src.shape[1] == 3

    src = src.T  # (3, N)
    dst = dst.T  # (3, N)
    n = src.shape[1]

    mu_src = np.mean(src, axis=1, keepdims=True)
    mu_dst = np.mean(dst, axis=1, keepdims=True)

    src_demean = src - mu_src
    dst_demean = dst - mu_dst

    # Covariance
    Sigma = (dst_demean @ src_demean.T) / float(n)

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_demean ** 2) / float(n)
        s = float(np.sum(D * np.diag(S))) / float(var_src + 1e-12)
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src  # (3,1)

    return s, R, t.reshape(3)


def evaluate_trajectory_against_gt(
    name: str,
    est_timestamps: List[float],
    est_positions: List[np.ndarray],
    gt_dict: Dict[float, np.ndarray],
    max_time_diff: float = 0.02,
    with_scale: bool = False,
    est_indices=None,
    return_details: bool = False,
    use_se3_align: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Simple ATE evaluation:

    - est_timestamps: list of floats
    - est_positions:  list of (3,) np.array (camera centers in world frame)
    - gt_dict: dataset.get_groundtruth_dict() (ts -> pose or vector)
        * If value is 4x4, we take pose[0:3, 3]
        * If value is 1D (e.g. [tx, ty, tz, qx, qy, qz, qw]), we take value[0:3]
    - max_time_diff: max allowed |t_est - t_gt|
    - with_scale: if True and use_se3_align=True, allow a global scale
                  in the Umeyama alignment; otherwise only rigid SE(3).
    - est_indices: optional list/array, same length as est_timestamps,
                   indicating external indices (e.g. frame index).
    - return_details: if True, return dict with matched arrays and errors.
    - use_se3_align: if True, perform full SE(3) (or Sim(3)) alignment via Umeyama.
    """

    # ---- Build GT time / position arrays ----
    gt_times = np.array(sorted(gt_dict.keys()), dtype=np.float64)

    gt_positions_list = []
    for t in gt_times:
        pose = gt_dict[t]
        pose = np.asarray(pose)

        if pose.ndim == 2 and pose.shape == (4, 4):
            gt_positions_list.append(pose[0:3, 3])
        elif pose.ndim == 1 and pose.shape[0] >= 3:
            gt_positions_list.append(pose[0:3])
        else:
            raise ValueError(
                f"Unexpected GT pose format for timestamp {t}: shape {pose.shape}"
            )

    gt_positions = np.array(gt_positions_list, dtype=np.float64)

    # ---- Build estimated arrays ----
    est_times = np.array(est_timestamps, dtype=np.float64)
    est_positions = np.array(est_positions, dtype=np.float64)

    if est_indices is None:
        est_indices = np.arange(len(est_times), dtype=int)
    else:
        est_indices = np.array(est_indices, dtype=int)

    matched_est = []
    matched_gt = []
    matched_est_times = []
    matched_gt_times = []
    matched_indices = []

    for idx_local, (t_est, p_est) in enumerate(zip(est_times, est_positions)):
        # find nearest gt timestamp
        idx = int(np.argmin(np.abs(gt_times - t_est)))
        t_gt = gt_times[idx]
        if abs(t_gt - t_est) <= max_time_diff:
            matched_est.append(p_est)
            matched_gt.append(gt_positions[idx])
            matched_est_times.append(t_est)
            matched_gt_times.append(t_gt)
            matched_indices.append(est_indices[idx_local])

    if len(matched_est) == 0:
        print(f"[Eval:{name}] No matched poses within time diff {max_time_diff}s.")
        return None

    matched_est = np.array(matched_est, dtype=np.float64)
    matched_gt = np.array(matched_gt, dtype=np.float64)
    matched_est_times = np.array(matched_est_times, dtype=np.float64)
    matched_gt_times = np.array(matched_gt_times, dtype=np.float64)
    matched_indices = np.array(matched_indices, dtype=int)

    # ---- Alignment: either simple scale or full SE(3) ----
    if use_se3_align:
        s, R, t = umeyama_align(matched_est, matched_gt, with_scale=with_scale)
        matched_est_aligned = (s * (R @ matched_est.T)).T + t
        print(
            f"[Eval:{name}] SE(3) alignment used (with_scale={with_scale}); "
            f"scale={s:.4f}"
        )
    else:
        if with_scale:
            num = float(np.sum(matched_est * matched_gt))
            den = float(np.sum(matched_est * matched_est) + 1e-12)
            s = num / den
            matched_est_aligned = s * matched_est
            print(f"[Eval:{name}] scalar scale factor: {s:.4f}")
        else:
            s = 1.0
            matched_est_aligned = matched_est
            print(f"[Eval:{name}] estimated scale factor: 1.0000")

    errors = matched_est_aligned - matched_gt
    err_norm = np.linalg.norm(errors, axis=1)

    rmse = float(np.sqrt(np.mean(err_norm ** 2)))
    mean_err = float(np.mean(err_norm))
    max_err = float(np.max(err_norm))

    print(f"[Eval:{name}] matched poses: {len(matched_est)}")
    print(f"[Eval:{name}] ATE RMSE: {rmse:.4f} m")
    print(f"[Eval:{name}] ATE mean: {mean_err:.4f} m")
    print(f"[Eval:{name}] ATE max:  {max_err:.4f} m")
    print("")

    if not return_details:
        return None

    return {
        "matched_est": matched_est_aligned,
        "matched_gt": matched_gt,
        "matched_est_times": matched_est_times,
        "matched_gt_times": matched_gt_times,
        "matched_indices": matched_indices,
        "errors": errors,
        "err_norm": err_norm,
        "scale": s,
    }


def build_lc_params_from_hparams(hp: SlamHyperParams) -> LoopClosureParams:
    return LoopClosureParams(
        min_frame_separation=hp.lc_min_frame_separation,
        min_candidate_matches=hp.lc_min_candidate_matches,
        max_candidates=hp.lc_max_candidates,
        min_inliers=hp.lc_min_inliers,
        pnp_reproj_thresh=hp.lc_pnp_reproj_thresh,
    )


# -------------------------------------------------------------
# Main callable: single SLAM run with a given config
# -------------------------------------------------------------
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
    Run SLAM once with a given hyperparameter config and return summary metrics.

    Returns dict with at least:
        - 'kf_rmse'
        - 'vo_rmse'
        - 'num_keyframes'
        - 'num_loops'
        - 'num_frames'
        - 'hparams'
    """
    dataset_root = Path(dataset_root)

    if save_dir is None:
        save_dir = dataset_root / "slam_runs"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TUMRGBDDataset(
        root_dir=dataset_root,
        max_time_diff=0.02,   # GT association tolerance
        depth_scale=5000.0,
    )
    cam = freiburg2_camera()

    n_total = len(dataset)
    i0 = 0 if start_idx is None else max(0, start_idx)
    i1 = n_total if end_idx is None else min(n_total, end_idx + 1)
    num_frames = i1 - i0
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
        i1 = i0 + num_frames

    if num_frames <= 1:
        print("[SLAM] Not enough frames to run (num_frames <= 1).")
        return {
            "kf_rmse": float("nan"),
            "vo_rmse": float("nan"),
            "num_keyframes": 0,
            "num_loops": 0,
            "num_frames": num_frames,
            "hparams": hparams.to_dict(),
        }

    vo = VisualOdometry(cam, VOParams())
    slam_map = Map()

    lc_params = build_lc_params_from_hparams(hparams)
    lc_detector = LoopClosureDetector(lc_params)

    frames_all: List[Frame] = []
    centers: List[np.ndarray] = []
    timestamps: List[float] = []
    vo_success_count = 0
    vo_stats: List[Dict[str, Any]] = []
    vo_debug_all: List[Optional[Dict[str, Any]]] = []

    last_kf_frame_id = -1
    num_loops_added = 0

    # thresholds from hyperparams
    trans_thresh = hparams.trans_thresh
    rot_thresh_deg = hparams.rot_thresh_deg
    min_frame_gap = hparams.min_frame_gap
    max_dt_gap = hparams.max_dt_gap

    prev_frame_obj: Optional[Frame] = None

    # ---------------------------------------------------------
    # Main SLAM loop (single segment until first big gap)
    # ---------------------------------------------------------
    for idx in range(i0, i1):
        data = dataset[idx]
        frame_id = idx

        frame_obj = Frame(
            id=frame_id,
            timestamp=data.timestamp,
            rgb=data.rgb,
            depth=data.depth,
            camera=cam,
        )

        if prev_frame_obj is None:
            # first frame
            frames_all.append(frame_obj)
            vo.process_first_frame(frame_obj)
            slam_map.add_keyframe(frame_obj)
            last_kf_frame_id = frame_obj.id

            center = frame_obj.camera_center()
            centers.append(center)
            timestamps.append(frame_obj.timestamp)

            vo_debug_all.append(None)
            vo_stats.append({
                "frame_id": frame_id,
                "timestamp": frame_obj.timestamp,
                "success": True,
                "num_matches": 0,
                "num_inliers": 0,
                "trans_mag": 0.0,
                "rot_deg": 0.0,
            })
            prev_frame_obj = frame_obj
            continue

        # gap check (terminate run if huge dt)
        dt = frame_obj.timestamp - prev_frame_obj.timestamp
        if dt > max_dt_gap:
            print(
                "[SLAM] Time gap exceeded (dt=%.3f > max_dt_gap=%.3f). "
                "Stopping segment here at frame %d."
                % (dt, max_dt_gap, frame_id)
            )
            break

        frames_all.append(frame_obj)

        success, inliers = vo.process_frame(prev_frame_obj, frame_obj)
        if not success:
            frame_obj.set_pose(prev_frame_obj.get_pose())
        else:
            vo_success_count += 1

        T_w_c = frame_obj.get_pose()
        center = frame_obj.camera_center()
        centers.append(center)
        timestamps.append(frame_obj.timestamp)

        # relative motion (for diagnostics)
        T_w_prev = prev_frame_obj.get_pose()
        T_prev_inv = np.linalg.inv(T_w_prev)
        T_prev_cur = T_prev_inv @ T_w_c
        R_rel = T_prev_cur[:3, :3]
        t_rel = T_prev_cur[:3, 3]
        trans_mag_frame = float(np.linalg.norm(t_rel))
        rot_deg_frame = rotation_angle_deg(R_rel)

        if success and vo.last_debug is not None:
            dbg = vo.last_debug
            vo_debug_all.append({
                "num_matches": dbg["num_matches"],
                "num_inliers": dbg["num_inliers"],
            })
            num_matches_dbg = dbg["num_matches"]
            num_inliers_dbg = dbg["num_inliers"]
        else:
            vo_debug_all.append(None)
            num_matches_dbg = 0
            num_inliers_dbg = 0

        vo_stats.append({
            "frame_id": frame_id,
            "timestamp": frame_obj.timestamp,
            "success": bool(success),
            "num_matches": int(num_matches_dbg),
            "num_inliers": int(num_inliers_dbg),
            "trans_mag": trans_mag_frame,
            "rot_deg": rot_deg_frame,
        })

        # Keyframe decision
        if frame_id - last_kf_frame_id >= min_frame_gap:
            last_kf = slam_map.get_last_keyframe()
            if last_kf is not None:
                T_w_last_kf = last_kf.get_pose()
                T_last_kf_inv = np.linalg.inv(T_w_last_kf)
                T_lastkf_cur = T_last_kf_inv @ T_w_c

                R_rel_kf = T_lastkf_cur[:3, :3]
                t_rel_kf = T_lastkf_cur[:3, 3]
                trans_mag = float(np.linalg.norm(t_rel_kf))
                rot_deg = rotation_angle_deg(R_rel_kf)

                if trans_mag > trans_thresh or rot_deg > rot_thresh_deg:
                    slam_map.add_keyframe(frame_obj)
                    last_kf_frame_id = frame_obj.id

                    # loop closure on keyframes
                    if len(slam_map.keyframes) > 3:
                        current_kf = frame_obj
                        candidates = lc_detector.find_candidates(
                            current_kf, slam_map.keyframes
                        )
                        for kf_id, score in candidates:
                            candidate_kf = slam_map.keyframes[kf_id]
                            lc_success, T_i_j, lc_inliers = lc_detector.verify_candidate(
                                current_kf=current_kf,
                                candidate_kf=candidate_kf,
                            )
                            if lc_success:
                                slam_map.pose_graph.add_edge(
                                    i=candidate_kf.id,
                                    j=current_kf.id,
                                    T_i_j=T_i_j,
                                    edge_type="loop",
                                )
                                num_loops_added += 1

        prev_frame_obj = frame_obj

    num_processed = len(centers)
    print(
        f"[SLAM] Finished run: frames processed = {num_processed}, "
        f"VO succ = {vo_success_count}, KFs = {len(slam_map.keyframes)}, "
        f"Loops = {num_loops_added}"
    )

    if num_processed <= 1:
        return {
            "kf_rmse": float("nan"),
            "vo_rmse": float("nan"),
            "num_keyframes": len(slam_map.keyframes),
            "num_loops": num_loops_added,
            "num_frames": num_processed,
            "hparams": hparams.to_dict(),
        }

    # ---------------------------------------------------------
    # Pose-graph optimization and evaluation
    # ---------------------------------------------------------
    slam_map.pose_graph.optimize(max_iterations=5)
    slam_map.update_keyframes_from_pose_graph()

    # keyframe trajectory
    kf_centers: List[np.ndarray] = []
    kf_timestamps: List[float] = []
    for kf_id in sorted(slam_map.keyframes.keys()):
        kf = slam_map.keyframes[kf_id]
        kf_centers.append(kf.camera_center())
        kf_timestamps.append(kf.timestamp)

    gt_dict = dataset.get_groundtruth_dict()

    # VO evaluation
    vo_eval = evaluate_trajectory_against_gt(
        name="VO",
        est_timestamps=timestamps,
        est_positions=centers,
        gt_dict=gt_dict,
        max_time_diff=0.02,
        with_scale=False,
        est_indices=list(range(len(timestamps))),
        return_details=True,
        use_se3_align=True,
    )

    # KF evaluation
    kf_eval = evaluate_trajectory_against_gt(
        name="KF",
        est_timestamps=kf_timestamps,
        est_positions=kf_centers,
        gt_dict=gt_dict,
        max_time_diff=0.02,
        with_scale=False,
        est_indices=list(range(len(kf_timestamps))),
        return_details=True,
        use_se3_align=True,
    )

    if vo_eval is not None:
        vo_rmse = float(np.sqrt(np.mean(vo_eval["err_norm"] ** 2)))
    else:
        vo_rmse = float("nan")

    if kf_eval is not None:
        kf_rmse = float(np.sqrt(np.mean(kf_eval["err_norm"] ** 2)))
    else:
        kf_rmse = float("nan")

    # Optional plotting (disabled for RL)
    if save_plots:
        plot_trajectory_xz(
            centers,
            title="VO trajectory",
            show=False,
            save_path=str(save_dir / "vo_xz.png"),
        )

        plot_trajectory_xz(
            kf_centers,
            title="KF trajectory",
            show=False,
            save_path=str(save_dir / "kf_xz.png"),
        )

        if vo_eval is not None:
            plot_trajectory_xz_with_gt(
                est_centers=vo_eval["matched_est"],
                gt_centers=vo_eval["matched_gt"],
                title="VO vs GT",
                show=False,
                save_path=str(save_dir / "vo_vs_gt_xz.png"),
            )

        if kf_eval is not None:
            plot_trajectory_xz_with_gt(
                est_centers=kf_eval["matched_est"],
                gt_centers=kf_eval["matched_gt"],
                title="KF vs GT",
                show=False,
                save_path=str(save_dir / "kf_vs_gt_xz.png"),
            )

        plot_pose_graph_xz(
            slam_map.pose_graph,
            title="Pose graph",
            show=False,
            save_path=str(save_dir / "pose_graph_xz.png"),
            gt_centers=kf_eval["matched_gt"] if kf_eval is not None else None,
        )

    results = {
        "kf_rmse": kf_rmse,
        "vo_rmse": vo_rmse,
        "num_keyframes": len(slam_map.keyframes),
        "num_loops": num_loops_added,
        "num_frames": num_processed,
        "hparams": hparams.to_dict(),
    }
    return results


if __name__ == "__main__":
    # Quick manual test
    root = Path("/Volumes/One Touch/Deep_RL_Project/rgbd_dataset_freiburg2_pioneer_slam2")
    hp = SlamHyperParams()
    out = run_slam_with_config(root, hp, save_plots=True)
    print(out)
