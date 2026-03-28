"""
vo_frontend.py

ORB-based visual odometry (VO) front-end for RGB-D sequences.

Pipeline per frame:
    1. Detect ORB keypoints and compute descriptors.
    2. Back-project keypoints to 3-D using the depth map.
    3. Match descriptors from the previous frame with Lowe's ratio test.
    4. Solve for the relative pose via PnP + RANSAC (solvePnPRansac).
    5. Compose with the previous world pose to obtain T_w_c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera import PinholeCamera
from .frame import Frame


@dataclass
class VOParams:
    """Tunable parameters for the visual odometry front-end."""

    n_features: int = 2000           # ORB feature budget
    match_ratio: float = 0.75        # Lowe's ratio-test threshold
    min_inliers: int = 30            # minimum PnP inliers to accept a pose
    pnp_reproj_thresh: float = 3.0   # RANSAC reprojection error (pixels)


class VisualOdometry:
    """Frame-to-frame visual odometry using ORB features and PnP RANSAC."""

    def __init__(self, camera: PinholeCamera, params: VOParams = VOParams()) -> None:
        self.cam = camera
        self.params = params

        self.orb = cv2.ORB_create(nfeatures=params.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Debug payload from the most recent successful frame (or None)
        self.last_debug: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_first_frame(self, frame: Frame) -> None:
        """Initialise VO on the first frame (pose set to identity)."""
        self._extract_features(frame)
        frame.set_pose(np.eye(4, dtype=np.float64))

    def process_frame(self, prev_frame: Frame, frame: Frame) -> Tuple[bool, int]:
        """
        Estimate the pose of *frame* relative to *prev_frame* and update
        ``frame.T_w_c`` in-place.

        Returns:
            (success, num_inliers)
        """
        self.last_debug = None
        self._extract_features(frame)

        for f in (prev_frame, frame):
            if any(x is None for x in (f.keypoints, f.descriptors, f.pts3d)):
                print(f"[VO] Frame {f.id} has no valid features.")
                return False, 0

        matches = self._match_descriptors(prev_frame.descriptors, frame.descriptors)
        if len(matches) < self.params.min_inliers:
            print(f"[VO] Not enough matches: {len(matches)}")
            return False, 0

        pts3d, pts2d, prev_pts, cur_pts = [], [], [], []
        for m in matches:
            X_prev = prev_frame.pts3d[m.queryIdx]
            if not np.isfinite(X_prev).all():
                continue
            pts3d.append(X_prev)
            pts2d.append(frame.keypoints[m.trainIdx])
            prev_pts.append(prev_frame.keypoints[m.queryIdx])
            cur_pts.append(frame.keypoints[m.trainIdx])

        if len(pts3d) < self.params.min_inliers:
            print(f"[VO] Not enough valid 3D–2D pairs: {len(pts3d)}")
            return False, len(pts3d)

        pts3d_arr = np.asarray(pts3d, dtype=np.float32)
        pts2d_arr = np.asarray(pts2d, dtype=np.float32)
        prev_pts_arr = np.asarray(prev_pts, dtype=np.float32)
        cur_pts_arr = np.asarray(cur_pts, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_arr,
            pts2d_arr,
            self.cam.K,
            np.zeros(5, dtype=np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.params.pnp_reproj_thresh,
            confidence=0.999,
            iterationsCount=100,
        )

        if not success or inliers is None:
            print("[VO] PnP RANSAC failed.")
            return False, 0

        num_inliers = int(len(inliers))
        if num_inliers < self.params.min_inliers:
            print(f"[VO] Too few PnP inliers: {num_inliers}")
            return False, num_inliers

        R, _ = cv2.Rodrigues(rvec)
        T_prev_cur = np.eye(4, dtype=np.float64)
        T_prev_cur[:3, :3] = R
        T_prev_cur[:3, 3] = tvec.ravel()

        frame.set_pose(prev_frame.get_pose() @ T_prev_cur)

        idx = inliers.ravel()
        self.last_debug = {
            "prev_id": prev_frame.id,
            "cur_id": frame.id,
            "num_matches": len(matches),
            "num_inliers": num_inliers,
            "prev_inliers": prev_pts_arr[idx],
            "cur_inliers": cur_pts_arr[idx],
        }

        return True, num_inliers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, frame: Frame) -> None:
        """Detect ORB keypoints, compute descriptors, and back-project to 3-D."""
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        kps_cv, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(kps_cv) == 0:
            frame.keypoints = frame.descriptors = frame.pts3d = None
            return

        pts2d_raw = np.array([[kp.pt[0], kp.pt[1]] for kp in kps_cv], dtype=np.float32)
        h, w = frame.depth.shape

        pts3d, valid_2d, valid_desc = [], [], []
        for i, (u, v) in enumerate(pts2d_raw):
            xi, yi = int(round(u)), int(round(v))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            d = float(frame.depth[yi, xi])
            if d <= 0 or not np.isfinite(d):
                continue
            pts3d.append(frame.camera.depth_to_3d(xi, yi, d))
            valid_2d.append([u, v])
            valid_desc.append(descriptors[i])

        if not pts3d:
            frame.keypoints = frame.descriptors = frame.pts3d = None
            return

        frame.keypoints = np.asarray(valid_2d, dtype=np.float32)
        frame.descriptors = np.asarray(valid_desc, dtype=np.uint8)
        frame.pts3d = np.asarray(pts3d, dtype=np.float32)

    def _match_descriptors(
        self, desc_prev: np.ndarray, desc_cur: np.ndarray
    ) -> List[cv2.DMatch]:
        """KNN (k=2) match with Lowe's ratio test."""
        good = []
        for m, n in self.matcher.knnMatch(desc_prev, desc_cur, k=2):
            if m.distance < self.params.match_ratio * n.distance:
                good.append(m)
        return good
