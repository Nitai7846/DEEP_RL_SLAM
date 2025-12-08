#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 01:17:02 2025

@author: nitaishah
"""

# slam/slam_hparams.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SlamHyperParams:
    """
    Hyperparameters that control SLAM behavior.
    RL will tune these per run.
    """

    # --- Keyframe selection ---
    trans_thresh: float = 0.10       # meters
    rot_thresh_deg: float = 10.0     # degrees
    min_frame_gap: int = 10          # frames between keyframes

    # --- Loop-closure heuristics ---
    lc_min_frame_separation: int = 30
    lc_min_candidate_matches: int = 80
    lc_max_candidates: int = 5
    lc_min_inliers: int = 50
    lc_pnp_reproj_thresh: float = 3.0

    # --- Dataset / segmentation ---
    # Max allowed dt between consecutive associated RGB-D frames.
    # If exceeded, we stop the run there (first segment only).
    max_dt_gap: float = 0.20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
