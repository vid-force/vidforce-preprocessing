"""Calibrated time series of a Hoi! session: interpolation of poses and vectors, nearest contact samples, poses
expressed in a reference camera, the rectified camera transform of the tracking device, and the gripper calibration
(device, sensor and tool frames).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _as_rigid_transform(value, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be (4,4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} rotation is not a proper orthonormal matrix")
    return transform


def _validate_time_series(times: np.ndarray, values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64)
    if t.size < 2:
        raise ValueError(f"{name} needs at least two timestamped samples")
    if v.shape[0] != t.size:
        raise ValueError(f"{name} value count {v.shape[0]} != timestamp count {t.size}")
    if not np.isfinite(t).all() or not np.isfinite(v).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError(f"{name} timestamps must be strictly increasing")
    return t, v


def _validate_queries(times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    query = np.asarray(query_times, dtype=np.float64).reshape(-1)
    if not np.isfinite(query).all():
        raise ValueError("query_times contains non-finite values")
    tolerance = max(1e-9, 1e-12 * max(abs(times[0]), abs(times[-1]), 1.0))
    if query.size and (query.min() < times[0] - tolerance or query.max() > times[-1] + tolerance):
        raise ValueError(
            f"query time outside pose range [{times[0]}, {times[-1]}]: "
            f"[{query.min()}, {query.max()}]"
        )
    return np.clip(query, times[0], times[-1])


def interpolate_pose_matrices(
    times: np.ndarray,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:

    times, positions = _validate_time_series(times, positions, "pose track")
    quats = np.asarray(quats_xyzw, dtype=np.float64)
    if positions.shape != (times.size, 3):
        raise ValueError(f"positions must be ({times.size},3), got {positions.shape}")
    if quats.shape != (times.size, 4):
        raise ValueError(f"quats_xyzw must be ({times.size},4), got {quats.shape}")
    if not np.isfinite(quats).all() or np.any(np.linalg.norm(quats, axis=1) < 1e-9):
        raise ValueError("quats_xyzw contains invalid quaternions")
    query = _validate_queries(times, query_times)

    pos = np.stack([np.interp(query, times, positions[:, dim]) for dim in range(3)], axis=1)
    rot = Slerp(times, Rotation.from_quat(quats))(query).as_matrix()
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], query.size, axis=0)
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = pos
    return poses


def interpolate_vectors(times: np.ndarray, vectors: np.ndarray, query_times: np.ndarray) -> np.ndarray:

    times, vectors = _validate_time_series(times, vectors, "vector stream")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be (N,D), got {vectors.shape}")
    query = _validate_queries(times, query_times)
    return np.stack(
        [np.interp(query, times, vectors[:, dim]) for dim in range(vectors.shape[1])], axis=1
    )


def nearest_boolean_samples(
    times: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:

    times, values_2d = _validate_time_series(times, np.asarray(values).reshape(-1, 1), "contact stream")
    query = _validate_queries(times, query_times)
    after = np.searchsorted(times, query, side="left")
    after = np.clip(after, 0, times.size - 1)
    before = np.maximum(after - 1, 0)
    choose_after = np.abs(times[after] - query) < np.abs(query - times[before])
    indices = np.where(choose_after, after, before)
    return values_2d[indices, 0] >= float(threshold)


def poses_in_reference_camera(
    poses_world_target: np.ndarray,
    T_world_reference_camera: np.ndarray,
) -> np.ndarray:

    poses = np.asarray(poses_world_target, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses_world_target must be (N,4,4), got {poses.shape}")
    reference = _as_rigid_transform(T_world_reference_camera, "T_world_reference_camera")
    return np.linalg.inv(reference)[None] @ poses


def aria_rectified_camera_transform(aria_calibration: Mapping) -> np.ndarray:

    try:
        pinhole = aria_calibration["PINHOLE"]
        T_device_raw = _as_rigid_transform(pinhole["T_device_camera"], "T_device_camRaw")
        T_raw_rectified = _as_rigid_transform(
            pinhole.get("pinhole_T_device_camera", np.eye(4)),
            "T_camRaw_camRect",
        )
    except KeyError as exc:
        raise ValueError(f"missing Aria camera calibration field: {exc}") from exc
    return _as_rigid_transform(T_device_raw @ T_raw_rectified, "T_device_camRect")


@dataclass(frozen=True)
class GripperCalibration:

    T_device_tool: np.ndarray
    T_device_force_sensor: np.ndarray

    @classmethod
    def from_release_metadata(
        cls,
        aria_calibration: Mapping,
        gripper_kalibr: Mapping,
    ) -> "GripperCalibration":

        def _first(mapping: Mapping, *keys: str):
            for key in keys:
                if key in mapping:
                    return mapping[key]
            raise KeyError("|".join(keys))

        try:
            T_device_cam_raw = _as_rigid_transform(
                aria_calibration["PINHOLE"]["T_device_camera"], "T_device_cam_raw"
            )
            T_cam1_cam0 = _as_rigid_transform(
                gripper_kalibr["cam1"]["T_cn_cnm1"], "T_cam1_cam0"
            )
            T_cam2_cam1 = _as_rigid_transform(
                gripper_kalibr["cam2"]["T_cn_cnm1"], "T_cam2_cam1"
            )
            T_cam0_imu = _as_rigid_transform(
                gripper_kalibr["cam0"]["T_cam_imu"], "T_cam0_imu"
            )
            T_imu_tool = _as_rigid_transform(
                _first(gripper_kalibr["imu0"], "T_imu_tool", "T_i_tool"),
                "T_imu_tool",
            )
            T_imu_sensor = _as_rigid_transform(
                _first(gripper_kalibr["imu0"], "T_imu_sensor", "T_i_s"),
                "T_imu_sensor",
            )
        except KeyError as exc:
            raise ValueError(f"missing gripper calibration field: {exc}") from exc

        T_cam_raw_cam0 = T_cam2_cam1 @ T_cam1_cam0
        T_device_cam0 = T_device_cam_raw @ T_cam_raw_cam0
        T_device_tool = T_device_cam0 @ T_cam0_imu @ T_imu_tool
        T_device_sensor = T_device_cam0 @ T_cam0_imu @ T_imu_sensor
        return cls(
            T_device_tool=_as_rigid_transform(T_device_tool, "T_device_tool"),
            T_device_force_sensor=_as_rigid_transform(
                T_device_sensor, "T_device_force_sensor"
            ),
        )
