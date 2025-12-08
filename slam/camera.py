#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:15:25 2025

@author: nitaishah
"""

"""
camera.py

Pinhole camera model + basic projection utilities.

We also provide a helper to construct the default
freiburg2_desk camera intrinsics.
"""

#from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 5000.0  # TUM default

    @property
    def K(self) -> np.ndarray:
        """Intrinsic matrix as (3,3) float32."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel coords (u, v) to a normalized ray in camera frame.

        Returns:
            np.ndarray (3,), direction vector [x, y, 1].
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return np.array([x, y, 1.0], dtype=np.float32)

    def depth_to_3d(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """
        Back-project a pixel with depth (in meters) to 3D in camera frame.

        Returns:
            np.ndarray (3,) = [X, Y, Z] in meters.
        """
        z = depth_m
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float32)

    def project_points(self, pts_cam: np.ndarray) -> np.ndarray:
        """
        Project 3D camera-frame points to pixel coordinates.

        Args:
            pts_cam: (N, 3) array of [X, Y, Z] in camera frame.

        Returns:
            (N, 2) array of pixel coordinates [u, v].
        """
        pts_cam = np.asarray(pts_cam, dtype=np.float32)
        assert pts_cam.shape[-1] == 3

        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]

        # Avoid division by zero
        Z = np.where(Z == 0, 1e-6, Z)

        x = X / Z
        y = Y / Z

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        return np.stack([u, v], axis=-1)


def freiburg2_camera() -> PinholeCamera:
    """
    Factory for the TUM freiburg2_desk camera.

    Official intrinsics (approx):
        fx = fy = 520.9
        cx = 325.1
        cy = 249.7
        depth_scale = 5000

    Resolution: 640x480
    """
    return PinholeCamera(
        width=640,
        height=480,
        fx=520.9,
        fy=520.9,
        cx=325.1,
        cy=249.7,
        depth_scale=5000.0,
    )
