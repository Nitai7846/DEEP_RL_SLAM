"""
se3.py

Minimal SE(3) / SO(3) Lie-group utilities implemented from scratch with NumPy.

Public API:
    hat3(omega)     3-vector → skew-symmetric matrix (so(3) hat operator)
    vee3(Omega)     skew-symmetric matrix → 3-vector (inverse of hat3)
    so3_exp(omega)  so(3) → SO(3) exponential map (Rodrigues formula)
    so3_log(R)      SO(3) → so(3) logarithm map
    se3_exp(xi)     se(3) → SE(3) exponential map
    se3_log(T)      SE(3) → se(3) logarithm map
"""

from __future__ import annotations

import numpy as np


def hat3(omega: np.ndarray) -> np.ndarray:
    """Convert a 3-vector to its skew-symmetric matrix (so(3) hat operator)."""
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz,  0.0, -wx],
            [-wy, wx,  0.0],
        ],
        dtype=np.float64,
    )


def vee3(Omega: np.ndarray) -> np.ndarray:
    """Inverse of hat3: skew-symmetric 3 × 3 matrix → 3-vector."""
    return np.array(
        [Omega[2, 1], Omega[0, 2], Omega[1, 0]],
        dtype=np.float64,
    )


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3) (Rodrigues formula).

    Args:
        omega: (3,) rotation vector (axis × angle).

    Returns:
        R: (3, 3) rotation matrix.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(omega)

    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) + hat3(omega)

    axis_hat = hat3(omega / theta)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * axis_hat
        + (1.0 - np.cos(theta)) * (axis_hat @ axis_hat)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SO(3) to so(3).

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        omega: (3,) rotation vector.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)

    return vee3((R - R.T) * (0.5 * theta / np.sin(theta)))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se(3) to SE(3).

    Args:
        xi: (6,) twist vector [omega (3), v (3)].

    Returns:
        T: (4, 4) transformation matrix.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega, v = xi[:3], xi[3:]

    theta = np.linalg.norm(omega)
    Omega = hat3(omega)

    if theta < 1e-8:
        R = np.eye(3, dtype=np.float64) + Omega
        V = np.eye(3, dtype=np.float64) + 0.5 * Omega
    else:
        R = so3_exp(omega)
        Omega2 = Omega @ Omega
        B = (1.0 - np.cos(theta)) / (theta * theta)
        C = (theta - np.sin(theta)) / (theta ** 3)
        V = np.eye(3, dtype=np.float64) + B * Omega + C * Omega2

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SE(3) to se(3).

    Args:
        T: (4, 4) transformation matrix.

    Returns:
        xi: (6,) twist vector [omega (3), v (3)].
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]

    omega = so3_log(R)
    theta = np.linalg.norm(omega)
    Omega = hat3(omega)

    if theta < 1e-8:
        V_inv = np.eye(3, dtype=np.float64) - 0.5 * Omega
    else:
        Omega2 = Omega @ Omega
        V_inv = (
            np.eye(3, dtype=np.float64)
            - 0.5 * Omega
            + (1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
            * Omega2
        )

    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = omega
    xi[3:] = V_inv @ t
    return xi
