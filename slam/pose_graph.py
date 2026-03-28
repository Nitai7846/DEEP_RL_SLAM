"""
pose_graph.py

Pose-graph data structures and a Gauss-Newton SE(3) optimizer for SLAM.

    PoseGraphNode  — holds a world pose T_w_i (4 × 4).
    PoseGraphEdge  — holds a relative pose constraint T_i_j with an edge type
                     ('odometry' or 'loop') and a 6 × 6 information matrix.
    PoseGraph      — container + Gauss-Newton optimizer with numeric Jacobians.

Optimization details:
    - Residual per edge: r = se3_log(inv(T_i_j_meas) @ inv(T_w_i) @ T_w_j)
    - Left-perturbation increments applied in the tangent space.
    - Anchor: the node with the smallest id is held fixed (gauge freedom).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .se3 import se3_exp, se3_log


@dataclass
class PoseGraphNode:
    id: int
    T_w_i: np.ndarray  # (4, 4) world_T_node


@dataclass
class PoseGraphEdge:
    i: int                  # from-node id
    j: int                  # to-node id
    T_i_j: np.ndarray       # (4, 4) relative pose  T_j = T_i @ T_i_j
    edge_type: str = "odometry"  # 'odometry' | 'loop'
    information: np.ndarray = field(
        default_factory=lambda: np.eye(6, dtype=np.float64)
    )  # (6, 6) information matrix


class PoseGraph:
    """Keyframe pose graph with SE(3) Gauss-Newton optimizer."""

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
            raise ValueError(f"Node {node_id} already exists.")
        self.nodes[node_id] = PoseGraphNode(id=node_id, T_w_i=T_w_i)

    def has_node(self, node_id: int) -> bool:
        return node_id in self.nodes

    def get_node_pose(self, node_id: int) -> np.ndarray:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found.")
        return self.nodes[node_id].T_w_i

    def set_node_pose(self, node_id: int, T_w_i: np.ndarray) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found.")
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
        self.edges.append(PoseGraphEdge(i=i, j=j, T_i_j=T_i_j, edge_type=edge_type, information=information))

    # ------------------------------------------------------------------
    # Gauss-Newton SE(3) optimizer
    # ------------------------------------------------------------------

    def optimize(self, max_iterations: int = 5, damping: float = 1e-6) -> None:
        """
        Run Gauss-Newton pose-graph optimization in SE(3).

        For each edge (i → j) with measured T_i_j:

            T_ij_pred  = inv(T_w_i) @ T_w_j
            T_error    = inv(T_i_j_meas) @ T_ij_pred
            r (6×1)    = se3_log(T_error)

        Jacobians are computed numerically via finite differences.
        The node with the smallest id is anchored (gauge freedom).
        """
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)

        if num_nodes < 2 or num_edges == 0:
            print(f"[PoseGraph] Nothing to optimize ({num_nodes} nodes, {num_edges} edges).")
            return

        node_ids = sorted(self.nodes.keys())
        anchor_id = node_ids[0]
        unknown_ids = [nid for nid in node_ids if nid != anchor_id]

        if not unknown_ids:
            return

        idx_map = {nid: 6 * k for k, nid in enumerate(unknown_ids)}
        num_unknowns = 6 * len(unknown_ids)
        eps = 1e-6

        for it in range(max_iterations):
            H = np.zeros((num_unknowns, num_unknowns))
            b = np.zeros(num_unknowns)
            total_error_sq = 0.0

            for e in self.edges:
                T_w_i = self.nodes[e.i].T_w_i
                T_w_j = self.nodes[e.j].T_w_i
                T_ij_pred = np.linalg.inv(T_w_i) @ T_w_j
                T_err = np.linalg.inv(e.T_i_j) @ T_ij_pred
                r = se3_log(T_err)
                total_error_sq += float(r @ r)

                J_i = J_j = None

                if e.i != anchor_id:
                    J_i = np.zeros((6, 6))
                    for k in range(6):
                        dxi = np.zeros(6)
                        dxi[k] = eps
                        T_w_i_p = se3_exp(dxi) @ T_w_i
                        r_p = se3_log(np.linalg.inv(e.T_i_j) @ np.linalg.inv(T_w_i_p) @ T_w_j)
                        J_i[:, k] = (r_p - r) / eps

                if e.j != anchor_id:
                    J_j = np.zeros((6, 6))
                    for k in range(6):
                        dxj = np.zeros(6)
                        dxj[k] = eps
                        T_w_j_p = se3_exp(dxj) @ T_w_j
                        r_p = se3_log(np.linalg.inv(e.T_i_j) @ np.linalg.inv(T_w_i) @ T_w_j_p)
                        J_j[:, k] = (r_p - r) / eps

                if e.i != anchor_id and J_i is not None:
                    s = idx_map[e.i]
                    H[s:s+6, s:s+6] += J_i.T @ J_i
                    b[s:s+6] += J_i.T @ r

                if e.j != anchor_id and J_j is not None:
                    s = idx_map[e.j]
                    H[s:s+6, s:s+6] += J_j.T @ J_j
                    b[s:s+6] += J_j.T @ r

                if e.i != anchor_id and e.j != anchor_id and J_i is not None and J_j is not None:
                    si, sj = idx_map[e.i], idx_map[e.j]
                    H[si:si+6, sj:sj+6] += J_i.T @ J_j
                    H[sj:sj+6, si:si+6] += J_j.T @ J_i

            H += damping * np.eye(num_unknowns)

            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                print("[PoseGraph] Linear solve failed (singular H). Stopping.")
                break

            max_step = 0.0
            for k, nid in enumerate(unknown_ids):
                dxi = dx[6*k : 6*k+6]
                max_step = max(max_step, float(np.linalg.norm(dxi)))
                self.nodes[nid].T_w_i = se3_exp(dxi) @ self.nodes[nid].T_w_i

            print(
                f"[PoseGraph] iter {it+1}/{max_iterations} "
                f"total_err²={total_error_sq:.6f}  max_step={max_step:.2e}"
            )

            if max_step < 1e-6:
                print("[PoseGraph] Converged.")
                break

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a short summary of node and edge counts."""
        n_odom = sum(1 for e in self.edges if e.edge_type == "odometry")
        n_loop = sum(1 for e in self.edges if e.edge_type == "loop")
        print(
            f"[PoseGraph] {len(self.nodes)} nodes, {len(self.edges)} edges "
            f"({n_odom} odometry, {n_loop} loop)"
        )
