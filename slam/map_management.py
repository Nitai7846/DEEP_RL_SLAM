"""
map_management.py

Keyframe map for RGB-D SLAM.

Responsibilities:
    - Store Frame objects selected as keyframes.
    - Maintain a PoseGraph with nodes and sequential odometry edges.
    - Sync keyframe poses back from the pose graph after optimization.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .frame import Frame
from .pose_graph import PoseGraph


class Map:
    """Manages keyframes and their associated pose graph."""

    def __init__(self) -> None:
        self.keyframes: Dict[int, Frame] = {}
        self.pose_graph = PoseGraph()
        self.last_keyframe_id: Optional[int] = None

    def add_keyframe(self, frame: Frame) -> None:
        """
        Register *frame* as a new keyframe.

        - Adds a pose-graph node at frame.id.
        - If a previous keyframe exists, adds an odometry edge to it.
        """
        kf_id = frame.id
        T_w_kf = frame.get_pose()

        if self.pose_graph.has_node(kf_id):
            raise ValueError(f"Keyframe {kf_id} already exists in the pose graph.")

        self.pose_graph.add_node(kf_id, T_w_kf)
        self.keyframes[kf_id] = frame

        if self.last_keyframe_id is not None:
            T_w_last = self.pose_graph.get_node_pose(self.last_keyframe_id)
            T_last_cur = np.linalg.inv(T_w_last) @ T_w_kf
            self.pose_graph.add_edge(
                i=self.last_keyframe_id,
                j=kf_id,
                T_i_j=T_last_cur,
                edge_type="odometry",
            )

        self.last_keyframe_id = kf_id

    def get_last_keyframe(self) -> Optional[Frame]:
        """Return the most recently added keyframe, or None."""
        if self.last_keyframe_id is None:
            return None
        return self.keyframes[self.last_keyframe_id]

    def update_keyframes_from_pose_graph(self) -> None:
        """
        After pose-graph optimization, copy optimized poses back into each
        keyframe's ``T_w_c`` field.
        """
        for kf_id, kf in self.keyframes.items():
            if self.pose_graph.has_node(kf_id):
                kf.set_pose(self.pose_graph.get_node_pose(kf_id))
