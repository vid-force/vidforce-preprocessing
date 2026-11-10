"""Calibrated labels of one interaction window: tool-point poses in the reference camera, the wrench about the
tool point, the contact phase, and waypoint times that keep the sample of the force peak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from vidforce.data.hoi.calibrated_streams import (
    GripperCalibration,
    interpolate_pose_matrices,
    interpolate_vectors,
    nearest_boolean_samples,
    poses_in_reference_camera,
)
from vidforce.data.hoi.clip_sample import (
    DenseMechanicsLabel,
    ForceTrajectoryLabel,
    WaypointTrajectoryLabel,
)
from vidforce.data.hoi.confirmed_interactions import (
    ConfirmedInteractionWindow,
    contact_gated_work_j_calibrated,
)


@dataclass(frozen=True)
class CalibratedInteractionLabels:

    waypoints: WaypointTrajectoryLabel
    force: ForceTrajectoryLabel
    dense_mechanics: DenseMechanicsLabel
    waypoint_times_ns: np.ndarray
    dense_times_ns: np.ndarray
    dense_tool_poses_world: np.ndarray
    dense_force_world_n: np.ndarray
    dense_contact: np.ndarray
    waypoint_start_ns: float
    waypoint_end_ns: float


def peak_preserving_waypoint_times(
    start: float,
    end: float,
    peak: float,
    *,
    num_waypoints: int,
) -> np.ndarray:

    start, end, peak = float(start), float(end), float(peak)
    if not np.isfinite([start, end, peak]).all() or end <= start:
        raise ValueError("start/end/peak must be finite with end > start")
    if peak < start or peak > end:
        raise ValueError(f"peak={peak} is outside [{start}, {end}]")
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least two")
    query = np.linspace(start, end, int(num_waypoints), dtype=np.float64)
    tolerance = max(1e-12, 1e-12 * max(abs(start), abs(end), 1.0))
    if np.min(np.abs(query - peak)) <= tolerance:
        return query
    if num_waypoints == 2:

        raise ValueError("num_waypoints=2 cannot preserve an interior force peak")
    replace = 1 + int(np.argmin(np.abs(query[1:-1] - peak)))
    query[replace] = peak
    query.sort()
    if not np.all(np.diff(query) > 0.0):
        raise RuntimeError("peak-preserving waypoint construction produced duplicate times")
    return query


def _require_window_coverage(
    window: ConfirmedInteractionWindow,
    streams: dict[str, np.ndarray],
) -> None:
    for name, raw_times in streams.items():
        times = np.asarray(raw_times, dtype=np.float64).reshape(-1)
        if times.size < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
            raise ValueError(f"{name} timestamps must contain >=2 finite increasing values")
        if window.start_ns < times[0] or window.end_ns > times[-1]:
            raise ValueError(
                f"confirmed window {window.name} is outside {name} range "
                f"[{times[0]}, {times[-1]}]"
            )


def _calibrated_poses(
    times_ns: np.ndarray,
    *,
    device_times_ns: np.ndarray,
    device_positions_world: np.ndarray,
    device_quats_xyzw_world: np.ndarray,
    calibration: GripperCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    device = interpolate_pose_matrices(
        device_times_ns,
        device_positions_world,
        device_quats_xyzw_world,
        times_ns,
    )
    return device @ calibration.T_device_tool, device @ calibration.T_device_force_sensor


def _wrench_about_tool_in_reference(
    force_sensor_n: np.ndarray,
    torque_sensor_nm: np.ndarray,
    poses_world_sensor: np.ndarray,
    poses_world_tool: np.ndarray,
    T_world_reference_camera: np.ndarray,
    *,
    force_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_world_sensor = poses_world_sensor[:, :3, :3]
    force_world = float(force_sign) * np.einsum("nij,nj->ni", R_world_sensor, force_sensor_n)
    torque_world_sensor = float(force_sign) * np.einsum(
        "nij,nj->ni", R_world_sensor, torque_sensor_nm
    )
    sensor_from_tool_world = poses_world_sensor[:, :3, 3] - poses_world_tool[:, :3, 3]
    torque_world_tool = torque_world_sensor + np.cross(sensor_from_tool_world, force_world)
    R_reference_world = np.asarray(T_world_reference_camera, dtype=np.float64)[:3, :3].T
    force_reference = np.einsum("ij,nj->ni", R_reference_world, force_world)
    torque_reference = np.einsum("ij,nj->ni", R_reference_world, torque_world_tool)
    return force_reference, torque_reference, force_world


def build_calibrated_interaction_labels(
    window: ConfirmedInteractionWindow,
    *,
    device_times_ns: np.ndarray,
    device_positions_world: np.ndarray,
    device_quats_xyzw_world: np.ndarray,
    force_times_ns: np.ndarray,
    force_sensor_n: np.ndarray,
    torque_sensor_nm: np.ndarray,
    trigger_times_ns: np.ndarray,
    trigger_values: np.ndarray,
    calibration: GripperCalibration,
    T_world_reference_camera: np.ndarray,
    num_waypoints: int = 32,
    contact_threshold: float = 0.5,
    force_sign: float = -1.0,
) -> CalibratedInteractionLabels:

    _require_window_coverage(
        window,
        {
            "device pose": device_times_ns,
            "force": force_times_ns,
            "contact trigger": trigger_times_ns,
        },
    )
    force_times = np.asarray(force_times_ns, dtype=np.float64).reshape(-1)
    force = np.asarray(force_sensor_n, dtype=np.float64)
    torque = np.asarray(torque_sensor_nm, dtype=np.float64)
    if force.shape != (force_times.size, 3) or torque.shape != force.shape:
        raise ValueError(
            f"force_sensor_n and torque_sensor_nm must be ({force_times.size},3), "
            f"got {force.shape} and {torque.shape}"
        )
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise ValueError("force/torque streams contain non-finite values")
    dense_mask = (force_times >= window.start_ns) & (force_times <= window.end_ns)
    dense_times = force_times[dense_mask]
    dense_force_sensor = force[dense_mask]
    dense_torque_sensor = torque[dense_mask]
    if dense_times.size < 2:
        raise ValueError(f"confirmed window {window.name} contains fewer than two force samples")

    contact_dense = nearest_boolean_samples(
        trigger_times_ns,
        trigger_values,
        dense_times,
        threshold=contact_threshold,
    )
    contact_indices = np.flatnonzero(contact_dense)
    if contact_indices.size < 2:
        raise ValueError(f"confirmed window {window.name} contains fewer than two contact samples")
    waypoint_start = float(dense_times[contact_indices[0]])
    waypoint_end = float(dense_times[contact_indices[-1]])
    if waypoint_end <= waypoint_start:
        raise ValueError(f"confirmed window {window.name} has a degenerate contact phase")
    dense_magnitude = np.linalg.norm(dense_force_sensor, axis=1)
    contact_magnitude = np.where(contact_dense, dense_magnitude, -np.inf)
    peak_index = int(np.argmax(contact_magnitude))
    peak_time = float(dense_times[peak_index])
    waypoint_times = peak_preserving_waypoint_times(
        waypoint_start,
        waypoint_end,
        peak_time,
        num_waypoints=num_waypoints,
    )
    force_waypoint_sensor = interpolate_vectors(force_times, force, waypoint_times)
    torque_waypoint_sensor = interpolate_vectors(force_times, torque, waypoint_times)

    dense_tool, dense_sensor = _calibrated_poses(
        dense_times,
        device_times_ns=device_times_ns,
        device_positions_world=device_positions_world,
        device_quats_xyzw_world=device_quats_xyzw_world,
        calibration=calibration,
    )
    waypoint_tool, waypoint_sensor = _calibrated_poses(
        waypoint_times,
        device_times_ns=device_times_ns,
        device_positions_world=device_positions_world,
        device_quats_xyzw_world=device_quats_xyzw_world,
        calibration=calibration,
    )
    contact_waypoint = nearest_boolean_samples(
        trigger_times_ns,
        trigger_values,
        waypoint_times,
        threshold=contact_threshold,
    )
    force_reference, torque_reference, _ = _wrench_about_tool_in_reference(
        force_waypoint_sensor,
        torque_waypoint_sensor,
        waypoint_sensor,
        waypoint_tool,
        T_world_reference_camera,
        force_sign=force_sign,
    )
    dense_force_reference, dense_torque_reference, dense_force_world = (
        _wrench_about_tool_in_reference(
            dense_force_sensor,
            dense_torque_sensor,
            dense_sensor,
            dense_tool,
            T_world_reference_camera,
            force_sign=force_sign,
        )
    )
    work_j = contact_gated_work_j_calibrated(
        dense_force_sensor,
        dense_sensor[:, :3, :3],
        dense_tool[:, :3, 3],
        contact_dense,
        force_sign=force_sign,
    )
    poses_reference_tool = poses_in_reference_camera(waypoint_tool, T_world_reference_camera)
    dense_poses_reference_tool = poses_in_reference_camera(
        dense_tool, T_world_reference_camera
    )
    dense_rotation_reference_tool = dense_poses_reference_tool[:, :3, :3]
    dense_force_tool = np.einsum(
        "nji,nj->ni", dense_rotation_reference_tool, dense_force_reference
    )
    dense_torque_tool = np.einsum(
        "nji,nj->ni", dense_rotation_reference_tool, dense_torque_reference
    )
    waypoint_label = WaypointTrajectoryLabel(
        time_s=((waypoint_times - waypoint_start) / 1_000_000_000.0).astype(np.float32),
        position_m=poses_reference_tool[:, :3, 3].astype(np.float32),
        quat_xyzw=Rotation.from_matrix(poses_reference_tool[:, :3, :3]).as_quat().astype(np.float32),
    )
    force_label = ForceTrajectoryLabel(
        vector_n=force_reference.astype(np.float32),
        torque_nm=torque_reference.astype(np.float32),
        contact=contact_waypoint,
        max_force_n=float(contact_magnitude[peak_index]),
        work_j=work_j,
    )
    dense_mechanics = DenseMechanicsLabel(
        time_s=(dense_times - dense_times[0]) / 1_000_000_000.0,
        tool_position_m=dense_poses_reference_tool[:, :3, 3],
        tool_quat_xyzw=Rotation.from_matrix(
            dense_poses_reference_tool[:, :3, :3]
        ).as_quat(),
        force_reference_n=dense_force_reference,
        torque_reference_nm=dense_torque_reference,
        force_tool_n=dense_force_tool,
        torque_tool_nm=dense_torque_tool,
        contact=contact_dense,
        contact_start_s=(waypoint_start - dense_times[0]) / 1_000_000_000.0,
        contact_end_s=(waypoint_end - dense_times[0]) / 1_000_000_000.0,
        max_force_n=float(contact_magnitude[peak_index]),
        work_j=work_j,
    )
    return CalibratedInteractionLabels(
        waypoints=waypoint_label,
        force=force_label,
        dense_mechanics=dense_mechanics,
        waypoint_times_ns=waypoint_times,
        dense_times_ns=dense_times,
        dense_tool_poses_world=dense_tool,
        dense_force_world_n=dense_force_world,
        dense_contact=contact_dense,
        waypoint_start_ns=waypoint_start,
        waypoint_end_ns=waypoint_end,
    )
