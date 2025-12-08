#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 16:19:17 2025

@author: nitaishah
"""

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import cv2

from .frame import Frame
from .camera import PinholeCamera


@dataclass
class VOParams:
    n_features: int = 2000
    # Lowe's ratio test threshold
    match_ratio: float = 0.75
    # Minimum number of inliers for accepting PnP
    min_inliers: int = 30
    # RANSAC reprojection threshold in pixels
    pnp_reproj_thresh: float = 3.0


class VisualOdometry:
    def __init__(self, camera: PinholeCamera, params: VOParams = VOParams()) -> None:
        self.cam = camera
        self.params = params

        # ORB feature detector/descriptor
        self.orb = cv2.ORB_create(nfeatures=self.params.n_features)

        # Brute-force matcher for binary descriptors (ORB)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # NEW: debug info for visualization
        self.last_debug = None

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def process_first_frame(self, frame: Frame) -> None:
        """Initialize VO on the first frame (pose already identity)."""
        self._extract_features(frame)
        frame.set_pose(np.eye(4, dtype=np.float64))

    def process_frame(self, prev_frame: Frame, frame: Frame) -> Tuple[bool, int]:
        """
        Estimate the pose of 'frame' relative to 'prev_frame', and update
        frame.T_w_c accordingly.

        Returns:
            (success, num_inliers)
        """
        # reset debug info
        self.last_debug = None

        # Extract features for current frame
        self._extract_features(frame)

        if (
            prev_frame.keypoints is None
            or prev_frame.descriptors is None
            or prev_frame.pts3d is None
        ):
            print("[VO] Previous frame has no valid features.")
            return False, 0

        if (
            frame.keypoints is None
            or frame.descriptors is None
            or frame.pts3d is None
        ):
            print("[VO] Current frame has no valid features.")
            return False, 0

        # Match descriptors between previous and current
        matches = self._match_descriptors(prev_frame.descriptors, frame.descriptors)
        if len(matches) < self.params.min_inliers:
            print(f"[VO] Not enough matches: {len(matches)}")
            return False, 0

        # Build 3D-2D correspondences for PnP:
        #  - 3D points from prev_frame (camera coord of prev)
        #  - 2D points from current frame
        pts3d = []
        pts2d = []

        # NEW: keep the 2D points in prev and current used for PnP
        prev_pts_pnp = []
        cur_pts_pnp = []

        kp_prev = prev_frame.keypoints
        kp_cur = frame.keypoints
        pts3d_prev = prev_frame.pts3d

        for m in matches:
            idx_prev = m.queryIdx
            idx_cur = m.trainIdx

            X_prev = pts3d_prev[idx_prev]  # (3,)
            if not np.isfinite(X_prev).all():
                continue

            u_prev, v_prev = kp_prev[idx_prev]
            u_cur, v_cur = kp_cur[idx_cur]

            pts3d.append(X_prev)
            pts2d.append([u_cur, v_cur])

            prev_pts_pnp.append([u_prev, v_prev])
            cur_pts_pnp.append([u_cur, v_cur])

        if len(pts3d) < self.params.min_inliers:
            print(f"[VO] Not enough valid 3D-2D pairs: {len(pts3d)}")
            return False, len(pts3d)

        pts3d = np.asarray(pts3d, dtype=np.float32)
        pts2d = np.asarray(pts2d, dtype=np.float32)
        prev_pts_pnp = np.asarray(prev_pts_pnp, dtype=np.float32)
        cur_pts_pnp = np.asarray(cur_pts_pnp, dtype=np.float32)

        # Camera matrix and zero distortion
        K = self.cam.K
        dist_coeffs = np.zeros(5, dtype=np.float32)

        # PnP with RANSAC: solves for transform from prev-frame coords to current camera coords
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.params.pnp_reproj_thresh,
            confidence=0.999,
            iterationsCount=100
        )

        if not success or inliers is None:
            print("[VO] PnPRansac failed.")
            return False, 0

        num_inliers = int(len(inliers))
        if num_inliers < self.params.min_inliers:
            print(f"[VO] Too few inliers after PnP: {num_inliers}")
            return False, num_inliers

        # Convert rvec, tvec to a 4x4 pose transform
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        # This transform maps 3D points from prev_frame camera coords to current camera coords:
        # X_cur = R * X_prev + t
        T_prev_cur = np.eye(4, dtype=np.float64)
        T_prev_cur[0:3, 0:3] = R
        T_prev_cur[0:3, 3] = t

        # Compose with previous world pose:
        # world_T_cur = world_T_prev @ prev_T_cur
        T_w_prev = prev_frame.get_pose()
        T_w_cur = T_w_prev @ T_prev_cur
        frame.set_pose(T_w_cur)

        # NEW: store debug info for visualization
        inlier_idx = inliers.ravel()
        prev_inliers = prev_pts_pnp[inlier_idx]
        cur_inliers = cur_pts_pnp[inlier_idx]

        self.last_debug = {
            "prev_id": prev_frame.id,
            "cur_id": frame.id,
            "num_matches": len(matches),
            "num_inliers": num_inliers,
            "prev_inliers": prev_inliers,
            "cur_inliers": cur_inliers,
        }

        return True, num_inliers

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _extract_features(self, frame: Frame) -> None:
        """Detect ORB keypoints, compute descriptors, and 3D points from depth."""
        # Convert RGB to grayscale for ORB
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)

        keypoints_cv, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints_cv) == 0:
            frame.keypoints = None
            frame.descriptors = None
            frame.pts3d = None
            print(f"[VO] No ORB features detected in frame {frame.id}.")
            return

        # Convert keypoints to (N, 2) float32 [u, v]
        pts2d = np.array(
            [[kp.pt[0], kp.pt[1]] for kp in keypoints_cv],
            dtype=np.float32
        )

        # Back-project to 3D using depth
        h, w = frame.depth.shape
        pts3d = []
        valid_2d = []
        valid_desc = []

        for i, (u, v) in enumerate(pts2d):
            x = int(round(u))
            y = int(round(v))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            d = frame.depth[y, x]  # meters
            if d <= 0 or not np.isfinite(d):
                continue
            X = frame.camera.depth_to_3d(x, y, float(d))
            pts3d.append(X)
            valid_2d.append([u, v])
            valid_desc.append(descriptors[i])

        if len(pts3d) == 0:
            frame.keypoints = None
            frame.descriptors = None
            frame.pts3d = None
            print(f"[VO] No valid depth-backed features in frame {frame.id}.")
            return

        frame.keypoints = np.asarray(valid_2d, dtype=np.float32)
        frame.descriptors = np.asarray(valid_desc, dtype=np.uint8)
        frame.pts3d = np.asarray(pts3d, dtype=np.float32)

    def _match_descriptors(
        self, desc_prev: np.ndarray, desc_cur: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Match ORB descriptors with Lowe's ratio test.
        Returns a list of good matches.
        """
        # KNN match with k=2
        knn_matches = self.matcher.knnMatch(desc_prev, desc_cur, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.params.match_ratio * n.distance:
                good_matches.append(m)

        return good_matches
