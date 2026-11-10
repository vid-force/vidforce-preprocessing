"""Geometry helpers: resampling of a pose track, world-to-camera transforms, pinhole projection, contact mask
and work from a force series.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample_pose_track(
    times: np.ndarray,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    num_waypoints: int,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)
    quats = np.asarray(quats_xyzw, dtype=np.float64)
    n = times.shape[0]
    if n < 2:
        raise ValueError(f"need >=2 source poses, got {n}")
    if positions.shape != (n, 3):
        raise ValueError(f"positions must be ({n},3), got {positions.shape}")
    if quats.shape != (n, 4):
        raise ValueError(f"quats must be ({n},4), got {quats.shape}")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing")
    if num_waypoints < 2:
        raise ValueError(f"num_waypoints must be >=2, got {num_waypoints}")

    t0 = times[0] if t_start is None else max(float(t_start), times[0])
    t1 = times[-1] if t_end is None else min(float(t_end), times[-1])
    if not t1 > t0:
        raise ValueError(f"empty resample window [{t0}, {t1}]")
    query = np.linspace(t0, t1, num_waypoints)

    pos_out = np.stack([np.interp(query, times, positions[:, k]) for k in range(3)], axis=1)

    rot = Rotation.from_quat(quats)
    slerp = Slerp(times, rot)
    quat_out = slerp(query).as_quat()

    return pos_out.astype(np.float32), quat_out.astype(np.float32)


def pose_world_to_camera(
    positions_w: np.ndarray,
    quats_w_xyzw: np.ndarray,
    T_world_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions_w = np.asarray(positions_w, dtype=np.float64)
    quats_w = np.asarray(quats_w_xyzw, dtype=np.float64)
    T = np.asarray(T_world_cam, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_world_cam must be (4,4), got {T.shape}")
    if positions_w.ndim != 2 or positions_w.shape[1] != 3:
        raise ValueError(f"positions_w must be (N,3), got {positions_w.shape}")
    if quats_w.shape != (positions_w.shape[0], 4):
        raise ValueError(f"quats_w must match positions ({positions_w.shape[0]},4)")

    R_wc = T[:3, :3]
    t_wc = T[:3, 3]

    if not np.allclose(R_wc @ R_wc.T, np.eye(3), atol=1e-3):
        raise ValueError("T_world_cam rotation block is not orthonormal")
    R_cw = R_wc.T

    pos_cam = (R_cw @ (positions_w.T - t_wc[:, None])).T
    rot_cw = Rotation.from_matrix(R_cw)
    quat_cam = (rot_cw * Rotation.from_quat(quats_w)).as_quat()
    return pos_cam.astype(np.float32), quat_cam.astype(np.float32)


def direction_world_to_camera(axis_w: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    axis_w = np.asarray(axis_w, dtype=np.float64).reshape(3)
    T = np.asarray(T_world_cam, dtype=np.float64)
    R_cw = T[:3, :3].T
    d = R_cw @ axis_w
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("axis maps to a near-zero vector")
    return (d / n).astype(np.float32)


def project_to_pixels(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_cam must be (N,3), got {pts.shape}")
    K = np.asarray(K, dtype=np.float64)
    z = pts[:, 2]
    valid = z > 1e-6
    zc = np.clip(z, 1e-6, None)
    u = K[0, 0] * pts[:, 0] / zc + K[0, 2]
    v = K[1, 1] * pts[:, 1] / zc + K[1, 2]
    return np.stack([u, v], axis=1), valid


def max_force_n(force_xyz: np.ndarray) -> float:
    F = np.asarray(force_xyz, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"force_xyz must be (N,3), got {F.shape}")
    if F.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(F, axis=1).max())


def contact_mask_from_force(force_xyz: np.ndarray, threshold_n: float) -> np.ndarray:
    F = np.asarray(force_xyz, dtype=np.float64)
    return np.linalg.norm(F, axis=1) >= float(threshold_n)


def contact_gated_work_j(
    force_xyz: np.ndarray,
    positions: np.ndarray,
    *,
    contact_mask: np.ndarray | None = None,
    threshold_n: float = 1.0,
) -> float:
    F = np.asarray(force_xyz, dtype=np.float64)
    X = np.asarray(positions, dtype=np.float64)
    if F.shape != X.shape or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"force {F.shape} and positions {X.shape} must both be (N,3)")
    n = F.shape[0]
    if n < 2:
        return 0.0
    if contact_mask is None:
        contact_mask = contact_mask_from_force(F, threshold_n)
    contact_mask = np.asarray(contact_mask, dtype=bool).reshape(-1)
    if contact_mask.shape[0] != n:
        raise ValueError(f"contact_mask length {contact_mask.shape[0]} != N={n}")
    dx = np.diff(X, axis=0)
    f_step = F[:-1]
    gate = contact_mask[:-1]
    dwork = np.sum(f_step * dx, axis=1)
    return float(np.sum(dwork[gate]))
