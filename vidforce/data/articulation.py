"""One-degree-of-freedom joint coordinate of a trajectory about its articulation axis: the signed swept angle
of a revolute joint [rad] or the displacement along a prismatic joint [m].
"""
from __future__ import annotations

import numpy as np

JOINT_UNITS = {"revolute": "rad", "prismatic": "m"}


def joint_coordinates(traj: np.ndarray, axis: np.ndarray, origin: np.ndarray,
                      joint_type: str) -> np.ndarray:
    p = np.asarray(traj, np.float64)
    a = np.asarray(axis, np.float64)
    o = np.asarray(origin, np.float64)
    n = p.shape[0]
    if n == 0:
        return np.zeros(0)

    if joint_type == "prismatic":
        return (p - p[0]) @ a

    if joint_type != "revolute":
        raise ValueError(f"unknown joint_type {joint_type!r}")


    u = (p - o) - np.outer((p - o) @ a, a)
    norm = np.linalg.norm(u, axis=1)
    if (norm < 1e-9).any():


        first = int(np.argmax(norm > 1e-9))
        if norm[first] < 1e-9:
            return np.zeros(n)
    else:
        first = 0
    u = u / np.maximum(norm, 1e-9)[:, None]

    q = np.zeros(n)
    for i in range(first + 1, n):
        if norm[i] < 1e-9:
            q[i] = q[i - 1]
            continue
        step = np.arctan2(a @ np.cross(u[i - 1], u[i]), u[i - 1] @ u[i])
        q[i] = q[i - 1] + step
    return q
