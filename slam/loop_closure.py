"""
loop_closure.py

Appearance-based loop-closure detection for RGB-D SLAM.

Two-stage approach:
    1. **Candidate search** — ORB descriptor similarity (match count) proposes
       past keyframes that may be the same place.
    2. **Geometric verification** — 3D–2D PnP RANSAC confirms the loop and
       recovers the relative pose T_i_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .frame import Frame


@dataclass
class LoopClosureParams:
    """Parameters controlling loop-closure detection."""

    min_frame_separation: int = 30      # min frame-id gap to consider a loop
    min_candidate_matches: int = 80     # descriptor matches needed to propose a candidate
    max_candidates: int = 5             # max candidates to verify per keyframe
    min_inliers: int = 50               # min PnP inliers to accept the loop
    pnp_reproj_thresh: float = 3.0      # PnP RANSAC reprojection error (pixels)


class LoopClosureDetector:
    """Detects and geometrically verifies loop closures between keyframes."""

    def __init__(self, params: LoopClosureParams = LoopClosureParams()) -> None:
        self.params = params
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_candidates(
        self,
        current_kf: Frame,
        keyframes: Dict[int, Frame],
    ) -> List[Tuple[int, int]]:
        """
        Propose loop candidates for *current_kf* by descriptor match count.

        Args:
            current_kf: The new keyframe being queried.
            keyframes:  All existing keyframes {id → Frame}.

        Returns:
            List of (kf_id, score) sorted by descending score, capped at
            ``max_candidates`` and filtered by ``min_candidate_matches``.
        """
        if current_kf.descriptors is None:
            return []

        cur_id = current_kf.id
        candidates: List[Tuple[int, int]] = []

        for kf_id, kf in keyframes.items():
            if kf_id == cur_id:
                continue
            if abs(kf_id - cur_id) < self.params.min_frame_separation:
                continue
            if kf.descriptors is None:
                continue

            good = sum(
                1
                for m, n in self.matcher.knnMatch(kf.descriptors, current_kf.descriptors, k=2)
                if m.distance < 0.75 * n.distance
            )
            if good >= self.params.min_candidate_matches:
                candidates.append((kf_id, good))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[: self.params.max_candidates]

    def verify_candidate(
        self,
        current_kf: Frame,
        candidate_kf: Frame,
    ) -> Tuple[bool, np.ndarray, int]:
        """
        Geometrically verify a loop candidate via 3D–2D PnP RANSAC.

        Uses 3-D points from *candidate_kf* and 2-D observations from *current_kf*.

        Returns:
            (success, T_i_j, num_inliers)
            where T_i_j maps points from candidate frame coords to current frame
            coords (X_j = R * X_i + t) when *success* is True; identity otherwise.
        """
        for f in (candidate_kf, current_kf):
            if any(x is None for x in (f.descriptors, f.keypoints, f.pts3d)):
                return False, np.eye(4), 0

        matches = [
            m
            for m, n in self.matcher.knnMatch(
                candidate_kf.descriptors, current_kf.descriptors, k=2
            )
            if m.distance < 0.75 * n.distance
        ]

        if len(matches) < self.params.min_inliers:
            return False, np.eye(4), len(matches)

        pts3d, pts2d = [], []
        for m in matches:
            X_i = candidate_kf.pts3d[m.queryIdx]
            if not np.isfinite(X_i).all():
                continue
            pts3d.append(X_i)
            pts2d.append(current_kf.keypoints[m.trainIdx])

        if len(pts3d) < self.params.min_inliers:
            return False, np.eye(4), len(pts3d)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(pts3d, dtype=np.float32),
            np.asarray(pts2d, dtype=np.float32),
            current_kf.camera.K,
            np.zeros(5, dtype=np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.params.pnp_reproj_thresh,
            confidence=0.999,
            iterationsCount=100,
        )

        if not success or inliers is None:
            return False, np.eye(4), 0

        num_inliers = int(len(inliers))
        if num_inliers < self.params.min_inliers:
            return False, np.eye(4), num_inliers

        R, _ = cv2.Rodrigues(rvec)
        T_i_j = np.eye(4, dtype=np.float64)
        T_i_j[:3, :3] = R
        T_i_j[:3, 3] = tvec.ravel()

        print(
            f"[LC] Loop verified: KF {candidate_kf.id} → KF {current_kf.id}, "
            f"inliers = {num_inliers}"
        )
        return True, T_i_j, num_inliers
