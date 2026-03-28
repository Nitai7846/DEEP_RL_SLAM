"""
frame.py

Frame dataclass for the RGB-D SLAM pipeline.

Holds:
    - RGB image and depth map
    - Camera intrinsics
    - Current SE(3) pose  T_w_c  (world_T_camera, 4 × 4)
    - ORB keypoints, descriptors, and back-projected 3-D points
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .camera import PinholeCamera


@dataclass
class Frame:
    id: int
    timestamp: float
    rgb: np.ndarray           # (H, W, 3), uint8, RGB order
    depth: np.ndarray         # (H, W), float32, meters
    camera: PinholeCamera

    # SE(3) pose: world_T_camera (4 × 4, float64)
    T_w_c: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    # Feature fields — populated by the VO front-end
    keypoints: Optional[np.ndarray] = None    # (N, 2) float32 [u, v]
    descriptors: Optional[np.ndarray] = None  # (N, D) uint8 (ORB)
    pts3d: Optional[np.ndarray] = None        # (N, 3) float32, camera frame

    def set_pose(self, T_w_c: np.ndarray) -> None:
        """Set the world_T_camera pose (must be 4 × 4)."""
        T_w_c = np.asarray(T_w_c, dtype=np.float64)
        assert T_w_c.shape == (4, 4)
        self.T_w_c = T_w_c

    def get_pose(self) -> np.ndarray:
        """Return the world_T_camera pose (4 × 4, float64)."""
        return self.T_w_c

    def camera_center(self) -> np.ndarray:
        """
        Return the camera center in world coordinates.

        For T_w_c = [R | t; 0 | 1], the camera center is C_w = -R^T t.
        """
        R = self.T_w_c[:3, :3]
        t = self.T_w_c[:3, 3]
        return -np.dot(R.T, t)
