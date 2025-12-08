#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:22:56 2025

@author: nitaishah
"""

"""
dataset.py

Simple loader for the TUM RGB-D dataset (e.g., rgbd_dataset_freiburg2_desk).

- Reads rgb.txt, depth.txt, groundtruth.txt
- Associates RGB and depth frames by timestamp (nearest neighbor)
- Exposes TUMRGBDDataset with __len__ and __getitem__
"""

#from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import cv2


@dataclass
class RGBDFrame:
    """A single synchronized RGB-D frame."""
    index: int
    timestamp: float
    rgb: np.ndarray   # (H, W, 3), uint8, RGB order
    depth: np.ndarray # (H, W), float32, meters


class TUMRGBDDataset:
    """
    Loader for a TUM RGB-D sequence directory.

    Expected structure inside root_dir:
      - rgb.txt
      - depth.txt
      - groundtruth.txt
      - associate.txt (optional, preferred if present)
      - accelerometer.txt (optional)
      - rgb images under some subfolder
      - depth images under some subfolder
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        max_time_diff: float = 0.02,
        depth_scale: float = 5000.0,
    ) -> None:
        """
        Args:
            root_dir: Path to e.g. 'rgbd_dataset_freiburg2_desk' or
                      'rgbd_dataset_freiburg2_pioneer_slam2'.
            max_time_diff: Max allowable time difference (seconds) when associating
                           RGB and depth frames (used only if associate.txt is absent).
            depth_scale: Depth values are usually in integer units; divide by this
                         to get meters (TUM uses 5000.0).
        """
        self.root = Path(root_dir)
        self.max_time_diff = max_time_diff
        self.depth_scale = depth_scale

        # Parse file lists
        self.rgb_files = self._read_image_list(self.root / "rgb.txt")
        self.depth_files = self._read_image_list(self.root / "depth.txt")
        self.groundtruth = self._read_groundtruth(self.root / "groundtruth.txt")

        # Prefer explicit associations if available
        assoc_path = self.root / "associate.txt"
        if assoc_path.exists():
            print(f"[TUMRGBDDataset] Using associate.txt at {assoc_path}")
            self.associations: List[Tuple[float, str, float, str]] = (
                self._read_associations(assoc_path)
            )
        else:
            print("[TUMRGBDDataset] No associate.txt, using timestamp-based association.")
            self.associations = self._associate(
                self.rgb_files, self.depth_files, max_time_diff
            )

        if len(self.associations) == 0:
            raise RuntimeError(
                f"No RGB–Depth pairs found within {max_time_diff} s in {self.root}"
            )

        # Optional accelerometer data
        accel_path = self.root / "accelerometer.txt"
        self.accel_times: Optional[np.ndarray] = None
        self.accel_values: Optional[np.ndarray] = None
        if accel_path.exists():
            print(f"[TUMRGBDDataset] Found accelerometer.txt at {accel_path}")
            self._read_accelerometer(accel_path)
        else:
            print("[TUMRGBDDataset] No accelerometer.txt found; IMU will be unavailable.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _read_image_list(path: Path) -> Dict[float, str]:
        """
        Read a TUM-style image list file.

        Each valid line:
            <timestamp> <relative_path>

        Returns:
            dict: timestamp -> relative_path
        """
        files: Dict[float, str] = {}

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                t = float(parts[0])
                rel_path = parts[1]
                files[t] = rel_path

        return files

    @staticmethod
    def _read_groundtruth(path: Path) -> Dict[float, np.ndarray]:
        """
        Read TUM groundtruth file.

        Each valid line:
            <timestamp> tx ty tz qx qy qz qw
        """
        gt: Dict[float, np.ndarray] = {}

        if not path.exists():
            # Ground truth is not strictly required for loading RGB-D frames.
            return gt

        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                t = float(parts[0])
                vals = np.array([float(x) for x in parts[1:]], dtype=np.float64)
                gt[t] = vals

        return gt

    @staticmethod
    def _associate(
        rgb_files: Dict[float, str],
        depth_files: Dict[float, str],
        max_time_diff: float,
    ) -> List[Tuple[float, str, float, str]]:
        """
        Associate RGB and depth by nearest timestamp, within a time window.

        Returns:
            List of tuples:
                (rgb_ts, rgb_rel_path, depth_ts, depth_rel_path)
        """
        rgb_times = sorted(rgb_files.keys())
        depth_times = sorted(depth_files.keys())

        i, j = 0, 0
        pairs: List[Tuple[float, str, float, str]] = []

        while i < len(rgb_times) and j < len(depth_times):
            t_rgb = rgb_times[i]
            t_depth = depth_times[j]
            diff = t_rgb - t_depth

            if abs(diff) <= max_time_diff:
                pairs.append(
                    (
                        t_rgb,
                        rgb_files[t_rgb],
                        t_depth,
                        depth_files[t_depth],
                    )
                )
                i += 1
                j += 1
            elif diff > 0:
                # depth is behind rgb, move depth forward
                j += 1
            else:
                # rgb is behind depth, move rgb forward
                i += 1

        return pairs

    def _read_associations(
        self,
        path: Path,
    ) -> List[Tuple[float, str, float, str]]:
        """
        Read TUM-style associate.txt:

            <t_rgb> <rgb_rel_path> <t_depth> <depth_rel_path>

        Returns:
            List of (t_rgb, rgb_rel_path, t_depth, depth_rel_path)
        """
        pairs: List[Tuple[float, str, float, str]] = []

        if not path.exists():
            return pairs

        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                t_rgb = float(parts[0])
                rgb_rel = parts[1]
                t_depth = float(parts[2])
                depth_rel = parts[3]
                pairs.append((t_rgb, rgb_rel, t_depth, depth_rel))

        return pairs

    def _read_accelerometer(self, path: Path) -> None:
        """
        Read accelerometer.txt.

        Expected format (TUM pioneer-style):
            <timestamp> ax ay az ...

        We store:
          accel_times: (N,) float64 timestamps
          accel_values: (N, 3) float64 accelerations [ax, ay, az]
        """
        times: List[float] = []
        values: List[List[float]] = []

        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                t = float(parts[0])
                ax, ay, az = map(float, parts[1:4])
                times.append(t)
                values.append([ax, ay, az])

        if len(times) == 0:
            print("[TUMRGBDDataset] accelerometer.txt was empty or malformed.")
            return

        self.accel_times = np.asarray(times, dtype=np.float64)
        self.accel_values = np.asarray(values, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.associations)

    def __getitem__(self, idx: int) -> RGBDFrame:
        """
        Load and return the idx-th RGB-D frame.

        Lazy loads images each time. For small experiments this is fine.
        """
        if idx < 0 or idx >= len(self.associations):
            raise IndexError(idx)

        rgb_ts, rgb_rel, depth_ts, depth_rel = self.associations[idx]

        rgb_path = self.root / rgb_rel
        depth_path = self.root / depth_rel

        # Load RGB: OpenCV reads as BGR, convert to RGB.
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth as uint16, convert to float meters.
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)

        depth = depth_raw.astype(np.float32) / float(self.depth_scale)

        # Use the RGB timestamp as the frame timestamp.
        return RGBDFrame(
            index=idx,
            timestamp=rgb_ts,
            rgb=rgb,
            depth=depth,
        )

    def get_associations(self) -> List[Tuple[float, str, float, str]]:
        """Return the list of (rgb_ts, rgb_rel, depth_ts, depth_rel)."""
        return self.associations

    def get_groundtruth_dict(self) -> Dict[float, np.ndarray]:
        """Return raw ground truth dict, timestamp -> [tx ty tz qx qy qz qw]."""
        return self.groundtruth

    def get_accel_between(
        self,
        t0: float,
        t1: float,
    ) -> Optional[np.ndarray]:
        """
        Return accelerometer samples between times [min(t0, t1), max(t0, t1)].

        Returns:
            None if no accelerometer data is available, otherwise
            an (N, 3) array of [ax, ay, az] in the interval (possibly N=0).
        """
        if self.accel_times is None or self.accel_values is None:
            return None

        t_start = min(t0, t1)
        t_end = max(t0, t1)

        mask = (self.accel_times >= t_start) & (self.accel_times <= t_end)
        if not np.any(mask):
            # Return an empty (0,3) array if no samples in range
            return self.accel_values[0:0, :]

        return self.accel_values[mask]
