#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 19:05:07 2025

@author: nitaishah
"""

"""
loop_closure.py

Naive loop closure detection for RGB-D SLAM using:
- ORB descriptor similarity (match count) to propose candidates
- 3D-2D PnP RANSAC for geometric verification

Assumes:
- Each keyframe is a Frame with ORB descriptors, keypoints, and 3D points.
- Camera intrinsics are known and similar across frames.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import cv2

from .frame import Frame


@dataclass
class LoopClosureParams:
    # Minimum frame-id gap between keyframes to even consider a loop
    min_frame_separation: int = 30

    # Descriptor-based candidate selection
    min_candidate_matches: int = 80
    max_candidates: int = 5

    # PnP geometric verification
    min_inliers: int = 50
    pnp_reproj_thresh: float = 3.0


class LoopClosureDetector:
    def __init__(self, params: LoopClosureParams = LoopClosureParams()) -> None:
        self.params = params
        # Matcher for binary descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------
    # Candidate search
    # ------------------------------------------------------------------
    def find_candidates(
        self,
        current_kf: Frame,
        keyframes: Dict[int, Frame],
    ) -> List[Tuple[int, int]]:
        """
        Propose loop candidates for 'current_kf' based on descriptor match counts.

        Args:
            current_kf: current keyframe (Frame) with descriptors, keypoints, pts3d.
            keyframes: dict of all existing keyframes {id -> Frame}.

        Returns:
            List of (kf_id, score) sorted by descending score, where score is
            the number of good descriptor matches. Limited to max_candidates and
            only includes scores >= min_candidate_matches.
        """
        if current_kf.descriptors is None:
            print("[LC] Current keyframe has no descriptors; cannot find candidates.")
            return []

        cur_id = current_kf.id
        desc_cur = current_kf.descriptors

        candidates: List[Tuple[int, int]] = []

        for kf_id, kf in keyframes.items():
            # Skip self
            if kf_id == cur_id:
                continue

            # Skip keyframes that are too close in time
            if abs(kf_id - cur_id) < self.params.min_frame_separation:
                continue

            if kf.descriptors is None:
                continue

            desc_kf = kf.descriptors

            # KNN match (k=2) + ratio test to estimate similarity
            knn = self.matcher.knnMatch(desc_kf, desc_cur, k=2)
            good = 0
            for m, n in knn:
                if m.distance < 0.75 * n.distance:
                    good += 1

            if good >= self.params.min_candidate_matches:
                candidates.append((kf_id, good))

        # Sort by descending match count
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Truncate to max_candidates
        candidates = candidates[: self.params.max_candidates]

        if candidates:
            print("[LC] Candidates for keyframe %d:" % cur_id, candidates)
        else:
            print("[LC] No loop candidates for keyframe %d." % cur_id)

        return candidates

    # ------------------------------------------------------------------
    # Geometric verification
    # ------------------------------------------------------------------
    def verify_candidate(
        self,
        current_kf: Frame,
        candidate_kf: Frame,
    ) -> Tuple[bool, np.ndarray, int]:
        """
        Geometric verification of a loop candidate via 3D-2D PnP RANSAC.

        We use:
          - 3D points from candidate_kf (RGB-D backprojected)
          - 2D points from current_kf

        Args:
            current_kf: Frame (keyframe) for which we are detecting a loop.
            candidate_kf: past keyframe hypothesized to be the same place.

        Returns:
            (success, T_i_j, num_inliers)
            where:
                success: True if PnP produced a good solution.
                T_i_j: 4x4 transform from candidate frame coords (i)
                       to current frame coords (j) if success, else identity.
                num_inliers: number of PnP inliers.
        """
        if (
            candidate_kf.descriptors is None
            or candidate_kf.keypoints is None
            or candidate_kf.pts3d is None
        ):
            print("[LC] Candidate keyframe %d has no valid features." % candidate_kf.id)
            return False, np.eye(4), 0

        if (
            current_kf.descriptors is None
            or current_kf.keypoints is None
            or current_kf.pts3d is None
        ):
            print("[LC] Current keyframe %d has no valid features." % current_kf.id)
            return False, np.eye(4), 0

        # Match descriptors: candidate -> current
        knn_matches = self.matcher.knnMatch(
            candidate_kf.descriptors, current_kf.descriptors, k=2
        )

        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.params.min_inliers:
            print(
                f"[LC] Not enough matches between KF {candidate_kf.id} and "
                f"{current_kf.id}: {len(good_matches)}"
            )
            return False, np.eye(4), len(good_matches)

        # Build 3D-2D correspondences:
        #   3D from candidate_kf (camera coords of candidate)
        #   2D from current_kf
        pts3d = []
        pts2d = []

        kp_i = candidate_kf.keypoints
        pts3d_i = candidate_kf.pts3d
        kp_j = current_kf.keypoints

        for m in good_matches:
            idx_i = m.queryIdx
            idx_j = m.trainIdx

            X_i = pts3d_i[idx_i]  # (3,)
            if not np.isfinite(X_i).all():
                continue
            u_j, v_j = kp_j[idx_j]
            pts3d.append(X_i)
            pts2d.append([u_j, v_j])

        if len(pts3d) < self.params.min_inliers:
            print(
                f"[LC] Not enough valid 3D-2D pairs after filtering: {len(pts3d)}"
            )
            return False, np.eye(4), len(pts3d)

        pts3d = np.asarray(pts3d, dtype=np.float32)
        pts2d = np.asarray(pts2d, dtype=np.float32)

        K = current_kf.camera.K
        dist_coeffs = np.zeros(5, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.params.pnp_reproj_thresh,
            confidence=0.999,
            iterationsCount=100,
        )

        if not success or inliers is None:
            print(
                f"[LC] PnPRansac failed for loop between KF {candidate_kf.id} "
                f"and {current_kf.id}."
            )
            return False, np.eye(4), 0

        num_inliers = int(len(inliers))
        if num_inliers < self.params.min_inliers:
            print(
                f"[LC] PnP inliers too few for loop between KF {candidate_kf.id} "
                f"and {current_kf.id}: {num_inliers}"
            )
            return False, np.eye(4), num_inliers

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        # This transform maps points from candidate camera coords to current camera coords:
        # X_j = R * X_i + t
        T_i_j = np.eye(4, dtype=np.float64)
        T_i_j[0:3, 0:3] = R
        T_i_j[0:3, 3] = t

        print(
            f"[LC] Loop verified between KF {candidate_kf.id} and {current_kf.id}, "
            f"inliers = {num_inliers}"
        )

        return True, T_i_j, num_inliers
