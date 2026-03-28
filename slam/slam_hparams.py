"""
slam_hparams.py

Hyperparameter dataclass for the SLAM pipeline.

All tunable values live here so the RL agent has a single, well-typed
interface for proposing and applying hyperparameter configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class SlamHyperParams:
    """Hyperparameters that control SLAM behavior. Tuned per-run by the RL agent."""

    # --- Keyframe selection ---
    trans_thresh: float = 0.10      # translation threshold (meters)
    rot_thresh_deg: float = 10.0    # rotation threshold (degrees)
    min_frame_gap: int = 10         # minimum frames between keyframe candidates

    # --- Loop-closure heuristics ---
    lc_min_frame_separation: int = 30    # min frame-id gap to consider a loop
    lc_min_candidate_matches: int = 80   # descriptor matches needed to propose a candidate
    lc_max_candidates: int = 5           # max candidates to geometrically verify per keyframe
    lc_min_inliers: int = 50             # min PnP inliers to accept a loop
    lc_pnp_reproj_thresh: float = 3.0    # PnP RANSAC reprojection error (pixels)

    # --- Dataset / segmentation ---
    max_dt_gap: float = 0.20  # max timestamp gap (seconds); larger gaps split segments

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (JSON-serializable)."""
        return asdict(self)
