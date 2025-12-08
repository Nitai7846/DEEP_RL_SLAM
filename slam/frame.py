#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:30:12 2025

@author: nitaishah
"""

"""
frame.py

Basic Frame class for the RGB-D SLAM pipeline.

Holds:
- RGB image
- Depth image
- Camera intrinsics
- Current pose T_w_c (world_T_camera)
- Placeholders for features (keypoints, descriptors, 3D points)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .camera import PinholeCamera


@dataclass
class Frame:
    id: int
    timestamp: float
    rgb: np.ndarray          # (H, W, 3), uint8, RGB
    depth: np.ndarray        # (H, W), float32, meters
    camera: PinholeCamera

    # Pose of this frame in world coordinates: 4x4 SE(3) matrix (world_T_camera)
    T_w_c: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    # Feature-related fields (to be filled by VO front-end later)
    keypoints: Optional[np.ndarray] = None   # e.g., (N, 2) float32 [u, v]
    descriptors: Optional[np.ndarray] = None # e.g., (N, D) uint8 (ORB)
    pts3d: Optional[np.ndarray] = None       # e.g., (N, 3) float32 in camera frame

    def set_pose(self, T_w_c: np.ndarray) -> None:
        """Set the world_T_camera pose."""
        T_w_c = np.asarray(T_w_c, dtype=np.float64)
        assert T_w_c.shape == (4, 4)
        self.T_w_c = T_w_c

    def get_pose(self) -> np.ndarray:
        """Return the world_T_camera pose."""
        return self.T_w_c

    def camera_center(self) -> np.ndarray:
        """
        Return the camera center in world coordinates.

        If T_w_c = [R t; 0 1], camera center C_w = -R^T t.
        """
        R = self.T_w_c[0:3, 0:3]
        t = self.T_w_c[0:3, 3]
        return -np.dot(R.T, t)
