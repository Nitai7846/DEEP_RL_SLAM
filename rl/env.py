"""
rl/env.py

Gym-compatible 1-step episodic environment for SLAM hyperparameter tuning.

Each episode:
    1. ``reset()``  — randomly selects a contiguous dataset segment.
    2. ``step(a)``  — maps a 6-D action ∈ [-1, 1]⁶ to SLAM hyperparameters,
                      runs the SLAM pipeline on the segment, and returns:

                          reward = -RMSE_seg
                                   - λ_kf  × (num_keyframes / num_frames)
                                   - λ_lc  × (num_loops    / num_frames)

The action dimensions map to:
    a[0]  trans_thresh          ∈ [0.05, 0.50] m
    a[1]  rot_thresh_deg        ∈ [2,    20  ] °
    a[2]  min_frame_gap         ∈ [5,    30  ] frames
    a[3]  lc_min_frame_sep      ∈ [30,   200 ] frames
    a[4]  lc_min_inliers        ∈ [20,   200 ] inliers
    a[5]  lc_pnp_reproj_thresh  ∈ [0.5,  5.0 ] pixels
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from slam.camera import freiburg2_camera
from slam.dataset import TUMRGBDDataset
from slam.frame import Frame
from slam.loop_closure import LoopClosureDetector, LoopClosureParams
from slam.map_management import Map
from slam.slam_runner import rotation_angle_deg, _eval_against_gt
from slam.vo_frontend import VisualOdometry, VOParams


def _compute_segments(
    dataset: TUMRGBDDataset,
    max_dt_gap: float = 0.20,
) -> List[Tuple[int, int]]:
    """
    Split a dataset into contiguous segments based on timestamp gaps.

    Returns:
        List of (start_idx, end_idx) inclusive pairs.
    """
    N = len(dataset)
    if N == 0:
        return []

    segments = []
    start = 0
    prev_ts = dataset[0].timestamp

    for i in range(1, N):
        ts = dataset[i].timestamp
        if ts - prev_ts > max_dt_gap:
            segments.append((start, i - 1))
            start = i
        prev_ts = ts

    segments.append((start, N - 1))
    return segments


class SlamHyperParamEnv:
    """
    1-step episodic RL environment for per-segment SLAM hyperparameter tuning.

    Follows a simplified Gym interface (``reset`` / ``step``).

    Args:
        dataset_root:        path to the TUM sequence directory.
        max_time_diff:       max timestamp diff (s) for RGB–depth association.
        depth_scale:         depth divisor (TUM: 5000).
        max_dt_gap:          max timestamp gap (s) used to split segments.
        lambda_kf:           reward penalty weight for keyframe density.
        lambda_lc:           reward penalty weight for loop-closure density.
        min_segment_frames:  discard segments shorter than this.
    """

    obs_dim: int = 4   # [start/N, end/N, len/N, num_segs/10]
    act_dim: int = 6   # 6 continuous actions in [-1, 1]

    def __init__(
        self,
        dataset_root: str,
        max_time_diff: float = 0.02,
        depth_scale: float = 5000.0,
        max_dt_gap: float = 0.20,
        lambda_kf: float = 0.05,
        lambda_lc: float = 0.05,
        min_segment_frames: int = 200,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.lambda_kf = lambda_kf
        self.lambda_lc = lambda_lc

        self.dataset = TUMRGBDDataset(
            root_dir=self.dataset_root,
            max_time_diff=max_time_diff,
            depth_scale=depth_scale,
        )
        self.cam = freiburg2_camera()
        self.gt_dict = self.dataset.get_groundtruth_dict()

        all_segs = _compute_segments(self.dataset, max_dt_gap=max_dt_gap)
        self.segments = [(s, e) for s, e in all_segs if e - s + 1 >= min_segment_frames]

        if not self.segments:
            print(
                f"[Env] WARNING: no segments ≥ {min_segment_frames} frames. "
                "Falling back to all segments."
            )
            self.segments = all_segs

        if not self.segments:
            raise RuntimeError("No segments available for RL training.")

        print(
            f"[Env] {len(self.segments)} usable segments out of {len(all_segs)} total."
        )

        self._current_seg_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Pick a random segment and return a normalized observation."""
        self._current_seg_idx = int(np.random.randint(len(self.segments)))
        start, end = self.segments[self._current_seg_idx]
        N = len(self.dataset)
        return np.array(
            [
                start / max(1, N),
                end / max(1, N),
                (end - start + 1) / max(1, N),
                len(self.segments) / 10.0,
            ],
            dtype=np.float32,
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply *action* → run SLAM on the current segment → return (obs, reward, done, info).

        Args:
            action: (6,) array in [-1, 1].
        """
        if self._current_seg_idx is None:
            raise RuntimeError("Call reset() before step().")

        params = self._action_to_hparams(np.asarray(action, dtype=np.float32))
        start, end = self.segments[self._current_seg_idx]
        metrics = self._run_slam_segment(start, end, params)

        num_frames = metrics["num_frames"]
        if num_frames == 0:
            reward = -10.0
        else:
            reward = (
                -metrics["rmse_kf"]
                - self.lambda_kf * (metrics["num_kf"] / num_frames)
                - self.lambda_lc * (metrics["num_loops"] / num_frames)
            )

        info: Dict[str, Any] = {
            "segment_index": self._current_seg_idx,
            "start_idx": start,
            "end_idx": end,
            **metrics,
            "hparams": params,
        }

        return np.zeros(self.obs_dim, dtype=np.float32), reward, True, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _action_to_hparams(self, a: np.ndarray) -> Dict[str, Any]:
        """Map action ∈ [-1, 1]⁶ to concrete hyperparameter values."""
        a = np.clip(a, -1.0, 1.0)

        def scale(val: float, lo: float, hi: float) -> float:
            return lo + 0.5 * (val + 1.0) * (hi - lo)

        return {
            "trans_thresh":           scale(a[0], 0.05, 0.50),
            "rot_thresh_deg":         scale(a[1], 2.0,  20.0),
            "min_frame_gap":          int(round(scale(a[2], 5,   30))),
            "lc_min_frame_separation":int(round(scale(a[3], 30,  200))),
            "lc_min_inliers":         int(round(scale(a[4], 20,  200))),
            "lc_pnp_reproj_thresh":   scale(a[5], 0.5,  5.0),
        }

    def _run_slam_segment(
        self,
        start: int,
        end: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run VO + keyframe selection + loop closure + PGO on one segment."""
        vo = VisualOdometry(self.cam, VOParams())
        slam_map = Map()
        lc_detector = LoopClosureDetector(
            LoopClosureParams(
                min_frame_separation=int(params["lc_min_frame_separation"]),
                min_candidate_matches=80,
                max_candidates=5,
                min_inliers=int(params["lc_min_inliers"]),
                pnp_reproj_thresh=float(params["lc_pnp_reproj_thresh"]),
            )
        )

        trans_thresh = float(params["trans_thresh"])
        rot_thresh = float(params["rot_thresh_deg"])
        min_gap = int(params["min_frame_gap"])

        last_kf_id = -1
        num_loops = 0
        prev_frame: Optional[Frame] = None

        for i in range(start, end + 1):
            data = self.dataset[i]
            frame = Frame(
                id=i,
                timestamp=data.timestamp,
                rgb=data.rgb,
                depth=data.depth,
                camera=self.cam,
            )

            if prev_frame is None:
                vo.process_first_frame(frame)
                slam_map.add_keyframe(frame)
                last_kf_id = frame.id
            else:
                ok, _ = vo.process_frame(prev_frame, frame)
                if not ok:
                    frame.set_pose(prev_frame.get_pose())

                if i - last_kf_id >= min_gap:
                    last_kf = slam_map.get_last_keyframe()
                    if last_kf is not None:
                        T_rel = np.linalg.inv(last_kf.get_pose()) @ frame.get_pose()
                        if (
                            np.linalg.norm(T_rel[:3, 3]) > trans_thresh
                            or rotation_angle_deg(T_rel[:3, :3]) > rot_thresh
                        ):
                            slam_map.add_keyframe(frame)
                            last_kf_id = frame.id

                            if len(slam_map.keyframes) > 3:
                                for kf_id, _ in lc_detector.find_candidates(
                                    frame, slam_map.keyframes
                                ):
                                    lc_ok, T_i_j, _ = lc_detector.verify_candidate(
                                        current_kf=frame,
                                        candidate_kf=slam_map.keyframes[kf_id],
                                    )
                                    if lc_ok:
                                        slam_map.pose_graph.add_edge(
                                            i=kf_id, j=frame.id,
                                            T_i_j=T_i_j, edge_type="loop",
                                        )
                                        num_loops += 1

            prev_frame = frame

        slam_map.pose_graph.optimize(max_iterations=5)
        slam_map.update_keyframes_from_pose_graph()

        kf_centers = [slam_map.keyframes[k].camera_center() for k in sorted(slam_map.keyframes)]
        kf_times = [slam_map.keyframes[k].timestamp for k in sorted(slam_map.keyframes)]

        rmse_kf = 10.0
        if kf_centers:
            result = _eval_against_gt("KF", kf_times, kf_centers, self.gt_dict)
            if result is not None:
                rmse_kf = float(np.sqrt(np.mean(result["err_norm"] ** 2)))

        return {
            "rmse_kf": rmse_kf,
            "num_kf": len(slam_map.keyframes),
            "num_loops": num_loops,
            "num_frames": end - start + 1,
        }
