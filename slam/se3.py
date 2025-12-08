#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 19:40:49 2025

@author: nitaishah
"""

"""
se3.py

Minimal SE(3) utilities:
- hat / vee for so(3)
- so3_exp, so3_log
- se3_exp, se3_log
"""

from typing import Tuple

import numpy as np


def hat3(omega: np.ndarray) -> np.ndarray:
    """
    Convert a 3-vector to skew-symmetric matrix (so(3) hat operator).
    """
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=np.float64,
    )


def vee3(omega_hat: np.ndarray) -> np.ndarray:
    """
    Inverse of hat3: skew-symmetric matrix -> 3-vector.
    """
    return np.array(
        [
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0],
        ],
        dtype=np.float64,
    )


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3).

    Args:
        omega: (3,) rotation vector.

    Returns:
        R: (3, 3) rotation matrix.
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(omega)

    if theta < 1e-8:
        # First-order approximation
        return np.eye(3, dtype=np.float64) + hat3(omega)

    axis = omega / theta
    axis_hat = hat3(axis)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    R = (
        np.eye(3, dtype=np.float64)
        + sin_theta * axis_hat
        + (1.0 - cos_theta) * (axis_hat @ axis_hat)
    )
    return R


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SO(3) to so(3) (rotation vector).

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        omega: (3,) rotation vector.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)

    omega_hat = (R - R.T) * (0.5 * theta / np.sin(theta))
    omega = vee3(omega_hat)
    return omega


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se(3) to SE(3).

    Args:
        xi: (6,) twist vector [omega, v].

    Returns:
        T: (4, 4) transformation matrix.
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega = xi[0:3]
    v = xi[3:6]

    theta = np.linalg.norm(omega)
    Omega = hat3(omega)

    if theta < 1e-8:
        # For very small rotations, use series expansion
        R = np.eye(3, dtype=np.float64) + Omega
        V = np.eye(3, dtype=np.float64) + 0.5 * Omega
    else:
        R = so3_exp(omega)
        Omega2 = Omega @ Omega
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        A = sin_theta / theta
        B = (1.0 - cos_theta) / (theta * theta)
        C = (theta - sin_theta) / (theta * theta * theta)

        V = np.eye(3, dtype=np.float64) + B * Omega + C * Omega2

    t = V @ v

    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SE(3) to se(3).

    Args:
        T: (4, 4) transformation matrix.

    Returns:
        xi: (6,) twist vector [omega, v].
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[0:3, 0:3]
    t = T[0:3, 3]

    omega = so3_log(R)
    theta = np.linalg.norm(omega)

    if theta < 1e-8:
        # For small rotation, V ~ I + 0.5 * Omega, so V^{-1} ~ I - 0.5 * Omega
        Omega = hat3(omega)
        V_inv = np.eye(3, dtype=np.float64) - 0.5 * Omega
    else:
        Omega = hat3(omega)
        Omega2 = Omega @ Omega
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        A = sin_theta / theta
        B = (1.0 - cos_theta) / (theta * theta)
        C = (1.0 - A) / (theta * theta)

        # V = I + B * Omega + C * Omega^2
        # V^{-1} formula for SE(3) (see Barfoot, etc.)
        V_inv = (
            np.eye(3, dtype=np.float64)
            - 0.5 * Omega
            + (1.0 / (theta * theta) - (1.0 + cos_theta) / (2.0 * theta * sin_theta))
            * (Omega2)
        )

    v = V_inv @ t

    xi = np.zeros(6, dtype=np.float64)
    xi[0:3] = omega
    xi[3:6] = v
    return xi
