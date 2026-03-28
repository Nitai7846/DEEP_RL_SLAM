"""
dataset.py

Loader for the TUM RGB-D benchmark dataset.

Reads rgb.txt, depth.txt, and groundtruth.txt; associates RGB and depth
frames by nearest timestamp; and exposes a simple indexable dataset.

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class RGBDFrame:
    """A single synchronized RGB-D frame."""

    index: int
    timestamp: float
    rgb: np.ndarray    # (H, W, 3), uint8, RGB order
    depth: np.ndarray  # (H, W), float32, meters


class TUMRGBDDataset:
    """
    Loader for a TUM RGB-D sequence directory.

    Expected directory structure::

        <root_dir>/
            rgb.txt
            depth.txt
            groundtruth.txt
            associate.txt          (optional — preferred when present)
            accelerometer.txt      (optional)
            rgb/   ...             (image files referenced in rgb.txt)
            depth/ ...             (image files referenced in depth.txt)
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        max_time_diff: float = 0.02,
        depth_scale: float = 5000.0,
    ) -> None:
        """
        Args:
            root_dir:      Path to the TUM sequence directory.
            max_time_diff: Maximum timestamp difference (s) when associating RGB
                           and depth frames (used only if associate.txt is absent).
            depth_scale:   Divisor to convert raw uint16 depth to meters (TUM: 5000).
        """
        self.root = Path(root_dir)
        self.max_time_diff = max_time_diff
        self.depth_scale = depth_scale

        self.rgb_files = self._read_image_list(self.root / "rgb.txt")
        self.depth_files = self._read_image_list(self.root / "depth.txt")
        self.groundtruth = self._read_groundtruth(self.root / "groundtruth.txt")

        assoc_path = self.root / "associate.txt"
        if assoc_path.exists():
            print(f"[TUMRGBDDataset] Using associate.txt at {assoc_path}")
            self.associations: List[Tuple[float, str, float, str]] = (
                self._read_associations(assoc_path)
            )
        else:
            print("[TUMRGBDDataset] No associate.txt — using timestamp-based association.")
            self.associations = self._associate(
                self.rgb_files, self.depth_files, max_time_diff
            )

        if len(self.associations) == 0:
            raise RuntimeError(
                f"No RGB–depth pairs found within {max_time_diff} s in {self.root}"
            )

        accel_path = self.root / "accelerometer.txt"
        self.accel_times: Optional[np.ndarray] = None
        self.accel_values: Optional[np.ndarray] = None
        if accel_path.exists():
            self._read_accelerometer(accel_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.associations)

    def __getitem__(self, idx: int) -> RGBDFrame:
        """Load and return the idx-th synchronized RGB-D frame (lazy)."""
        if idx < 0 or idx >= len(self.associations):
            raise IndexError(idx)

        rgb_ts, rgb_rel, depth_ts, depth_rel = self.associations[idx]

        rgb = cv2.imread(str(self.root / rgb_rel), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(self.root / rgb_rel)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(str(self.root / depth_rel), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(self.root / depth_rel)
        depth = depth_raw.astype(np.float32) / float(self.depth_scale)

        return RGBDFrame(index=idx, timestamp=rgb_ts, rgb=rgb, depth=depth)

    def get_associations(self) -> List[Tuple[float, str, float, str]]:
        """Return the list of (rgb_ts, rgb_path, depth_ts, depth_path) tuples."""
        return self.associations

    def get_groundtruth_dict(self) -> Dict[float, np.ndarray]:
        """Return the raw ground-truth dict: timestamp → [tx, ty, tz, qx, qy, qz, qw]."""
        return self.groundtruth

    def get_accel_between(self, t0: float, t1: float) -> Optional[np.ndarray]:
        """
        Return accelerometer samples in the interval [min(t0,t1), max(t0,t1)].

        Returns None if no accelerometer data is available, otherwise (N, 3).
        """
        if self.accel_times is None or self.accel_values is None:
            return None

        t_start, t_end = min(t0, t1), max(t0, t1)
        mask = (self.accel_times >= t_start) & (self.accel_times <= t_end)
        return self.accel_values[mask]

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_image_list(path: Path) -> Dict[float, str]:
        """Parse a TUM-style image list file: <timestamp> <relative_path>."""
        if not path.exists():
            raise FileNotFoundError(path)
        files: Dict[float, str] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    files[float(parts[0])] = parts[1]
        return files

    @staticmethod
    def _read_groundtruth(path: Path) -> Dict[float, np.ndarray]:
        """Parse a TUM groundtruth file: <timestamp> tx ty tz qx qy qz qw."""
        gt: Dict[float, np.ndarray] = {}
        if not path.exists():
            return gt
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    gt[float(parts[0])] = np.array(
                        [float(x) for x in parts[1:]], dtype=np.float64
                    )
        return gt

    @staticmethod
    def _associate(
        rgb_files: Dict[float, str],
        depth_files: Dict[float, str],
        max_time_diff: float,
    ) -> List[Tuple[float, str, float, str]]:
        """Associate RGB and depth frames by nearest timestamp within a time window."""
        rgb_times = sorted(rgb_files.keys())
        depth_times = sorted(depth_files.keys())

        i, j = 0, 0
        pairs: List[Tuple[float, str, float, str]] = []

        while i < len(rgb_times) and j < len(depth_times):
            t_rgb, t_depth = rgb_times[i], depth_times[j]
            diff = t_rgb - t_depth

            if abs(diff) <= max_time_diff:
                pairs.append((t_rgb, rgb_files[t_rgb], t_depth, depth_files[t_depth]))
                i += 1
                j += 1
            elif diff > 0:
                j += 1
            else:
                i += 1

        return pairs

    def _read_associations(self, path: Path) -> List[Tuple[float, str, float, str]]:
        """Parse a TUM associate.txt file."""
        pairs: List[Tuple[float, str, float, str]] = []
        if not path.exists():
            return pairs
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    pairs.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
        return pairs

    def _read_accelerometer(self, path: Path) -> None:
        """Parse accelerometer.txt: <timestamp> ax ay az ..."""
        times, values = [], []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    times.append(float(parts[0]))
                    values.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if times:
            self.accel_times = np.asarray(times, dtype=np.float64)
            self.accel_values = np.asarray(values, dtype=np.float64)
