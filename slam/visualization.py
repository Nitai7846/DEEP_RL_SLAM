"""
visualization.py

Matplotlib plotting utilities for SLAM trajectories and pose graphs.

    plot_trajectory_xz          2-D top-down trajectory (X-Z plane).
    plot_trajectory_xz_with_gt  Estimated trajectory overlaid with ground truth.
    plot_pose_graph_xz          Pose-graph nodes and edges (X-Z plane).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .pose_graph import PoseGraph


def plot_trajectory_xz(
    centers: Sequence[np.ndarray],
    title: str = "Camera trajectory (top-down X-Z)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot camera centers in the X-Z plane (top-down view).

    Args:
        centers:   sequence of (3,) world positions [X, Y, Z].
        title:     plot title.
        show:      whether to call ``plt.show()``.
        save_path: optional file path to save the figure.
    """
    if not centers:
        return

    pts = np.asarray(centers, dtype=float)
    xs, zs = pts[:, 0], pts[:, 2]

    fig, ax = plt.subplots()
    ax.plot(xs, zs, marker=".", linewidth=1)
    ax.scatter(xs[0], zs[0], c="g", label="start", zorder=5)
    ax.scatter(xs[-1], zs[-1], c="r", label="end", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_xz_with_gt(
    est_centers: np.ndarray,
    gt_centers: np.ndarray,
    title: str = "Estimated trajectory vs ground truth (X-Z)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Overlay estimated and ground-truth trajectories in the X-Z plane.

    Args:
        est_centers: (N, 3) estimated camera positions.
        gt_centers:  (N, 3) ground-truth positions (aligned with est_centers).
    """
    if est_centers.size == 0 or gt_centers.size == 0:
        return

    est = np.asarray(est_centers, dtype=float)
    gt = np.asarray(gt_centers, dtype=float)

    fig, ax = plt.subplots()
    ax.plot(gt[:, 0], gt[:, 2], "b-", linewidth=1.5, label="ground truth")
    ax.plot(est[:, 0], est[:, 2], "r--", linewidth=1.0, label="estimate")
    ax.scatter(gt[0, 0], gt[0, 2], c="g", label="start", zorder=5)
    ax.scatter(gt[-1, 0], gt[-1, 2], c="k", label="end (GT)", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_pose_graph_xz(
    pose_graph: PoseGraph,
    title: str = "Pose graph (top-down X-Z)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Draw pose-graph nodes and edges in the X-Z plane.

    - Nodes: black dots (keyframe camera centers).
    - Odometry edges: thin gray lines.
    - Loop-closure edges: thin red lines.
    """
    if not pose_graph.nodes:
        return

    node_ids = sorted(pose_graph.nodes.keys())

    def center(T: np.ndarray) -> np.ndarray:
        R, t = T[:3, :3], T[:3, 3]
        return -R.T @ t

    centers = np.array([center(pose_graph.nodes[nid].T_w_i) for nid in node_ids])

    fig, ax = plt.subplots()
    ax.scatter(centers[:, 0], centers[:, 2], c="k", s=10, label="nodes", zorder=3)

    for e in pose_graph.edges:
        ci = center(pose_graph.nodes[e.i].T_w_i)
        cj = center(pose_graph.nodes[e.j].T_w_i)
        color = "0.6" if e.edge_type == "odometry" else "r"
        lw = 0.5 if e.edge_type == "odometry" else 0.8
        ax.plot([ci[0], cj[0]], [ci[2], cj[2]], color=color, linewidth=lw, alpha=0.7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
