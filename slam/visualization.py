#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:44:46 2025

@author: nitaishah
"""

"""
visualization.py

Simple plotting utilities for SLAM:
- 2D top-down trajectory (X-Z plane).
"""

"""
visualization.py

Simple plotting utilities for SLAM:
- 2D top-down trajectory (X-Z plane).
"""

from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt

from .pose_graph import PoseGraph  # add this import

def plot_trajectory_xz(
    centers: Sequence[np.ndarray],
    title: str = "Camera trajectory (top-down X-Z)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot camera centers in X-Z plane (top-down view).

    Args:
        centers: sequence of (3,) arrays [X, Y, Z] in world coordinates.
        title: plot title.
        show: whether to call plt.show().
        save_path: optional path to save the figure (e.g. 'traj.png').
    """
    if len(centers) == 0:
        print("[viz] No centers to plot.")
        return

    centers = np.asarray(centers, dtype=float)
    assert centers.shape[1] == 3, "Each center must be (3,)"

    xs = centers[:, 0]
    zs = centers[:, 2]

    plt.figure()
    plt.plot(xs, zs, marker=".", linewidth=1)
    plt.scatter(xs[0], zs[0], c="g", label="start")
    plt.scatter(xs[-1], zs[-1], c="r", label="end")

    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved trajectory plot to: {save_path}")

    if show:
        plt.show()
        
def plot_trajectory_xz_with_gt(
    est_centers: np.ndarray,
    gt_centers: np.ndarray,
    title: str = "Trajectory vs ground truth (top-down X-Z)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot estimated and ground truth camera centers in X-Z plane.

    Args:
        est_centers: (N,3) estimated positions.
        gt_centers:  (N,3) ground truth positions (aligned with est_centers).
    """
    if est_centers.size == 0 or gt_centers.size == 0:
        print("[viz] No centers to plot for trajectory-with-GT.")
        return

    est_centers = np.asarray(est_centers, dtype=float)
    gt_centers = np.asarray(gt_centers, dtype=float)

    xs_est = est_centers[:, 0]
    zs_est = est_centers[:, 2]

    xs_gt = gt_centers[:, 0]
    zs_gt = gt_centers[:, 2]

    plt.figure()
    plt.plot(xs_gt, zs_gt, "b-", linewidth=1.5, label="ground truth")
    plt.plot(xs_est, zs_est, "r--", linewidth=1.0, label="estimate")

    plt.scatter(xs_gt[0], zs_gt[0], c="g", label="start")
    plt.scatter(xs_gt[-1], zs_gt[-1], c="k", label="end (GT)")

    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved trajectory-with-GT plot to: {save_path}")

    if show:
        plt.show()



        
def plot_pose_graph_xz(
    pose_graph: PoseGraph,
    title: str = "Pose graph (top-down X-Z)",
    show: bool = True,
    save_path: Optional[str] = None,
    gt_centers: Optional[np.ndarray] = None,
) -> None:
    """
    Plot pose graph nodes and edges in X-Z plane.

    - Nodes: keyframe camera centers.
    - Odometry edges: thin gray lines.
    - Loop edges: thin red lines.
    - Optional ground truth trajectory: blue curve.
    """
    if len(pose_graph.nodes) == 0:
        print("[viz] Pose graph is empty.")
        return

    # Collect node centers
    node_ids = sorted(pose_graph.nodes.keys())
    centers = []
    for nid in node_ids:
        T_w_i = pose_graph.nodes[nid].T_w_i
        R = T_w_i[0:3, 0:3]
        t = T_w_i[0:3, 3]
        # Camera center: C_w = -R^T t  (consistent with your pose graph convention)
        C_w = -R.T @ t
        centers.append(C_w)

    centers = np.asarray(centers, dtype=float)
    xs = centers[:, 0]
    zs = centers[:, 2]

    plt.figure()
    # Nodes
    plt.scatter(xs, zs, c="k", s=10, label="nodes")

    # Edges
    for e in pose_graph.edges:
        T_w_i = pose_graph.nodes[e.i].T_w_i
        T_w_j = pose_graph.nodes[e.j].T_w_i

        R_i = T_w_i[0:3, 0:3]
        t_i = T_w_i[0:3, 3]
        R_j = T_w_j[0:3, 0:3]
        t_j = T_w_j[0:3, 3]

        C_i = -R_i.T @ t_i
        C_j = -R_j.T @ t_j

        x_pair = [C_i[0], C_j[0]]
        z_pair = [C_i[2], C_j[2]]

        if e.edge_type == "odometry":
            plt.plot(x_pair, z_pair, linestyle="-", linewidth=0.5, alpha=0.5, color="0.6")
        else:  # loop
            plt.plot(x_pair, z_pair, linestyle="-", linewidth=0.8, alpha=0.8, color="r")

    # Optional GT overlay
    #if gt_centers is not None and len(gt_centers) > 0:
        #gt_centers = np.asarray(gt_centers, dtype=float)
        #xs_gt = gt_centers[:, 0]
        #zs_gt = gt_centers[:, 2]
        #plt.plot(xs_gt, zs_gt, "b-", linewidth=1.0, label="ground truth")

    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved pose graph plot to: {save_path}")

    if show:
        plt.show()
