#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:53:05 2025

@author: nitaishah
"""

"""
map_management.py

Minimal map / keyframe manager for RGB-D SLAM.

Responsibilities:
- Store keyframes (Frame objects)
- Maintain a PoseGraph with nodes and odometry edges
"""

from typing import Dict, Optional

import numpy as np

from .frame import Frame
from .pose_graph import PoseGraph


class Map:
    def __init__(self) -> None:
        self.keyframes: Dict[int, Frame] = {}
        self.pose_graph = PoseGraph()
        self.last_keyframe_id: Optional[int] = None

    def add_keyframe(self, frame: Frame) -> None:
        """
        Add a Frame as a keyframe.

        - Stores the frame
        - Adds a pose graph node at frame.id
        - If there is a previous keyframe, adds an odometry edge between them.
        """
        kf_id = frame.id
        T_w_kf = frame.get_pose()

        if self.pose_graph.has_node(kf_id):
            raise ValueError("Keyframe id %d already exists in pose graph" % kf_id)

        # Add node
        self.pose_graph.add_node(kf_id, T_w_kf)
        self.keyframes[kf_id] = frame

        # Add odometry edge to previous keyframe if it exists
        if self.last_keyframe_id is not None:
            last_id = self.last_keyframe_id
            T_w_last = self.pose_graph.get_node_pose(last_id)

            # Relative pose from last -> current:
            # T_last_cur = T_last^-1 @ T_cur
            T_last_inv = np.linalg.inv(T_w_last)
            T_last_cur = np.dot(T_last_inv, T_w_kf)

            self.pose_graph.add_edge(
                i=last_id,
                j=kf_id,
                T_i_j=T_last_cur,
                edge_type="odometry",
            )

        # Update last keyframe id
        self.last_keyframe_id = kf_id

    def get_last_keyframe(self) -> Optional[Frame]:
        if self.last_keyframe_id is None:
            return None
        return self.keyframes[self.last_keyframe_id]
    
    def update_keyframes_from_pose_graph(self) -> None:
       """
       After pose graph optimization, update each keyframe's pose
       from the corresponding PoseGraph node pose.
       """
       for kf_id, kf in self.keyframes.items():
           if not self.pose_graph.has_node(kf_id):
               continue
           T_w_i = self.pose_graph.get_node_pose(kf_id)
           kf.set_pose(T_w_i)
   
