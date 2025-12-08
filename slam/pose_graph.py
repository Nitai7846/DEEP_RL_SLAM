#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 18:52:45 2025

@author: nitaishah
"""

"""
pose_graph.py

Minimal pose graph data structures for SLAM:
- PoseGraphNode: holds a pose T_w_i
- PoseGraphEdge: holds a relative pose constraint T_i_j
- PoseGraph: container for nodes and edges + optimize()
"""
"""
pose_graph.py

Pose graph data structures for SLAM:
- PoseGraphNode: holds a pose T_w_i
- PoseGraphEdge: holds a relative pose constraint T_i_j
- PoseGraph: container for nodes and edges + SE(3) optimizer
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .se3 import se3_exp, se3_log


@dataclass
class PoseGraphNode:
    id: int
    T_w_i: np.ndarray  # 4x4 world_T_node


@dataclass
class PoseGraphEdge:
    i: int  # from node id
    j: int  # to node id
    T_i_j: np.ndarray  # 4x4 relative pose from i to j (T_j = T_i @ T_i_j)
    edge_type: str = "odometry"  # 'odometry' or 'loop'
    information: np.ndarray = field(
        default_factory=lambda: np.eye(6, dtype=np.float64)
    )  # 6x6 information (not used explicitly yet)


class PoseGraph:
    def __init__(self) -> None:
        self.nodes: Dict[int, PoseGraphNode] = {}
        self.edges: List[PoseGraphEdge] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------
    def add_node(self, node_id: int, T_w_i: np.ndarray) -> None:
        T_w_i = np.asarray(T_w_i, dtype=np.float64)
        assert T_w_i.shape == (4, 4)
        if node_id in self.nodes:
            raise ValueError("Node with id %d already exists" % node_id)
        self.nodes[node_id] = PoseGraphNode(id=node_id, T_w_i=T_w_i)

    def has_node(self, node_id: int) -> bool:
        return node_id in self.nodes

    def get_node_pose(self, node_id: int) -> np.ndarray:
        if node_id not in self.nodes:
            raise KeyError("Node id %d not found" % node_id)
        return self.nodes[node_id].T_w_i

    def set_node_pose(self, node_id: int, T_w_i: np.ndarray) -> None:
        if node_id not in self.nodes:
            raise KeyError("Node id %d not found" % node_id)
        T_w_i = np.asarray(T_w_i, dtype=np.float64)
        assert T_w_i.shape == (4, 4)
        self.nodes[node_id].T_w_i = T_w_i

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------
    def add_edge(
        self,
        i: int,
        j: int,
        T_i_j: np.ndarray,
        edge_type: str = "odometry",
        information: Optional[np.ndarray] = None,
    ) -> None:
        T_i_j = np.asarray(T_i_j, dtype=np.float64)
        assert T_i_j.shape == (4, 4)
        if information is None:
            information = np.eye(6, dtype=np.float64)
        else:
            information = np.asarray(information, dtype=np.float64)
            assert information.shape == (6, 6)

        edge = PoseGraphEdge(
            i=i,
            j=j,
            T_i_j=T_i_j,
            edge_type=edge_type,
            information=information,
        )
        self.edges.append(edge)

    # ------------------------------------------------------------------
    # Optimization (full SE(3) with Gauss-Newton, numeric Jacobians)
    # ------------------------------------------------------------------
    def optimize(self, max_iterations: int = 5, damping: float = 1e-6) -> None:
        """
        Full SE(3) pose graph optimization using Gauss-Newton with numeric Jacobians.

        State: SE(3) pose for each node.
        Gauge freedom is fixed by keeping the first node (smallest id) fixed.

        Error for each edge (i -> j) with measured T_i_j:

            T_ij_pred   = inv(T_w_i) @ T_w_j
            T_error     = inv(T_i_j) @ T_ij_pred
            r_ij (6x1)  = se3_log(T_error)

        We linearize r_ij around current estimates and solve for increments
        in the tangent space (6 DoF per node, except the fixed anchor).
        """
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)

        if num_nodes < 2 or num_edges == 0:
            print(
                "[PoseGraph] optimize() called but not enough nodes/edges "
                f"({num_nodes} nodes, {num_edges} edges)"
            )
            return

        # Anchor node (fixed) is the first node id in sorted order
        node_ids = sorted(self.nodes.keys())
        anchor_id = node_ids[0]

        # Unknown node ids and index mapping: 6 parameters per unknown node
        unknown_ids = [nid for nid in node_ids if nid != anchor_id]
        if not unknown_ids:
            print("[PoseGraph] Only anchor node present; nothing to optimize.")
            return

        idx_map = {nid: 6 * k for k, nid in enumerate(unknown_ids)}
        num_unknowns = 6 * len(unknown_ids)

        print(
            f"[PoseGraph] Starting SE(3) optimization: {num_nodes} nodes, "
            f"{num_edges} edges, {len(unknown_ids)} unknown nodes"
        )

        eps = 1e-6  # finite-difference step for numeric Jacobians

        for it in range(max_iterations):
            # Normal equations H x = -b
            H = np.zeros((num_unknowns, num_unknowns), dtype=np.float64)
            b = np.zeros((num_unknowns,), dtype=np.float64)

            total_error_sq = 0.0

            for e in self.edges:
                i = e.i
                j = e.j
                T_ij_meas = e.T_i_j

                T_w_i = self.nodes[i].T_w_i
                T_w_j = self.nodes[j].T_w_i

                # Predicted relative pose
                T_i_w = np.linalg.inv(T_w_i)
                T_ij_pred = T_i_w @ T_w_j

                # Error transform: meas^{-1} * pred
                T_err = np.linalg.inv(T_ij_meas) @ T_ij_pred

                # Residual (6x1)
                r = se3_log(T_err)
                total_error_sq += float(r @ r)

                # Numeric Jacobians w.r.t node i and j (each 6x6)
                J_i = None
                J_j = None

                # Jacobian w.r.t node i (if not anchor)
                if i != anchor_id:
                    J_i = np.zeros((6, 6), dtype=np.float64)
                    base_T_w_i = T_w_i.copy()

                    for k in range(6):
                        dxi = np.zeros(6, dtype=np.float64)
                        dxi[k] = eps
                        dT = se3_exp(dxi)

                        # Left-multiply increment: T_w_i_pert = dT * T_w_i
                        T_w_i_pert = dT @ base_T_w_i
                        T_i_w_pert = np.linalg.inv(T_w_i_pert)
                        T_ij_pred_pert = T_i_w_pert @ T_w_j
                        T_err_pert = np.linalg.inv(T_ij_meas) @ T_ij_pred_pert
                        r_pert = se3_log(T_err_pert)

                        J_i[:, k] = (r_pert - r) / eps

                # Jacobian w.r.t node j (if not anchor)
                if j != anchor_id:
                    J_j = np.zeros((6, 6), dtype=np.float64)
                    base_T_w_j = T_w_j.copy()

                    for k in range(6):
                        dxj = np.zeros(6, dtype=np.float64)
                        dxj[k] = eps
                        dT = se3_exp(dxj)

                        # Left-multiply increment: T_w_j_pert = dT * T_w_j
                        T_w_j_pert = dT @ base_T_w_j
                        T_i_w = np.linalg.inv(T_w_i)
                        T_ij_pred_pert = T_i_w @ T_w_j_pert
                        T_err_pert = np.linalg.inv(T_ij_meas) @ T_ij_pred_pert
                        r_pert = se3_log(T_err_pert)

                        J_j[:, k] = (r_pert - r) / eps

                # Assemble into H and b
                if i != anchor_id and J_i is not None:
                    idx_i = idx_map[i]
                    H[idx_i : idx_i + 6, idx_i : idx_i + 6] += J_i.T @ J_i
                    b[idx_i : idx_i + 6] += J_i.T @ r

                if j != anchor_id and J_j is not None:
                    idx_j = idx_map[j]
                    H[idx_j : idx_j + 6, idx_j : idx_j + 6] += J_j.T @ J_j
                    b[idx_j : idx_j + 6] += J_j.T @ r

                if i != anchor_id and j != anchor_id and J_i is not None and J_j is not None:
                    idx_i = idx_map[i]
                    idx_j = idx_map[j]
                    H[idx_i : idx_i + 6, idx_j : idx_j + 6] += J_i.T @ J_j
                    H[idx_j : idx_j + 6, idx_i : idx_i + 6] += J_j.T @ J_i

            # Damping for numerical stability
            H += damping * np.eye(num_unknowns, dtype=np.float64)

            # Solve for increments: H * dx = -b
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                print("[PoseGraph] Linear solve failed (singular H). Stopping.")
                break

            # Apply increments to unknown node poses
            max_step = 0.0
            for k, nid in enumerate(unknown_ids):
                dxi = dx[6 * k : 6 * k + 6]
                max_step = max(max_step, float(np.linalg.norm(dxi)))
                dT = se3_exp(dxi)
                T_w_i = self.nodes[nid].T_w_i
                self.nodes[nid].T_w_i = dT @ T_w_i

            print(
                f"[PoseGraph] iter {it+1}/{max_iterations}, "
                f"total_error_sq={total_error_sq:.6f}, max_step={max_step:.6e}"
            )

            # Simple convergence check
            if max_step < 1e-6:
                print("[PoseGraph] Converged.")
                break

        print(
            "[PoseGraph] optimize(): SE(3) GN completed with "
            f"{num_nodes} nodes, {num_edges} edges"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """
        Print a short summary of the pose graph.
        """
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        num_odom = sum(1 for e in self.edges if e.edge_type == "odometry")
        num_loop = sum(1 for e in self.edges if e.edge_type == "loop")

        print(f"[PoseGraph] Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"  - Odometry edges: {num_odom}")
        print(f"  - Loop edges:     {num_loop}")
