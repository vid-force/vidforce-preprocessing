"""Validated containers of one interaction clip: waypoint trajectory, articulation (screw) label, force
trajectory, dense mechanics (force, torque, contact, work) and the clip sample that bundles frames and labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from vidforce.data.hoi.sample import JointType


class ClipTask(str, Enum):
    ARTICULATION = "articulation"
    FORCE = "force"


def _finite_array(value, *, dtype, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class WaypointTrajectoryLabel:
    time_s: np.ndarray
    position_m: np.ndarray
    quat_xyzw: np.ndarray

    def __post_init__(self) -> None:
        time = _finite_array(self.time_s, dtype=np.float32, name="time_s").reshape(-1)
        if time.size < 2:
            raise ValueError("waypoints need at least two timestamps")
        if not np.isclose(time[0], 0.0, atol=1e-6):
            raise ValueError(f"waypoint time_s must start at zero, got {time[0]}")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("waypoint time_s must be strictly increasing")
        position = _finite_array(self.position_m, dtype=np.float32, name="position_m")
        quat = _finite_array(self.quat_xyzw, dtype=np.float32, name="quat_xyzw")
        if position.shape != (time.size, 3):
            raise ValueError(f"position_m must be ({time.size},3), got {position.shape}")
        if quat.shape != (time.size, 4):
            raise ValueError(f"quat_xyzw must be ({time.size},4), got {quat.shape}")
        norms = np.linalg.norm(quat, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("quat_xyzw rows must be unit quaternions")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quat_xyzw", quat / norms[:, None])

    @property
    def num_waypoints(self) -> int:
        return int(self.time_s.size)


@dataclass(frozen=True)
class ScrewLabel:
    joint_type: JointType
    axis: np.ndarray
    origin_m: np.ndarray
    confidence: float

    def __post_init__(self) -> None:
        joint_type = JointType(int(self.joint_type))
        if joint_type == JointType.UNKNOWN:
            raise ValueError("ScrewLabel joint_type cannot be UNKNOWN")
        axis = _finite_array(self.axis, dtype=np.float32, name="axis")
        origin = _finite_array(self.origin_m, dtype=np.float32, name="origin_m")
        if axis.shape != (3,) or origin.shape != (3,):
            raise ValueError(f"axis and origin_m must both be (3,), got {axis.shape}, {origin.shape}")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            raise ValueError("axis is degenerate")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0,1]")
        object.__setattr__(self, "joint_type", joint_type)
        object.__setattr__(self, "axis", (axis / norm).astype(np.float32))
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class ForceTrajectoryLabel:
    vector_n: np.ndarray
    torque_nm: np.ndarray
    contact: np.ndarray
    max_force_n: float
    work_j: float

    def __post_init__(self) -> None:
        force = _finite_array(self.vector_n, dtype=np.float32, name="vector_n")
        torque = _finite_array(self.torque_nm, dtype=np.float32, name="torque_nm")
        contact = np.asarray(self.contact, dtype=bool).reshape(-1)
        if force.ndim != 2 or force.shape[1] != 3:
            raise ValueError(f"vector_n must be (T,3), got {force.shape}")
        if torque.shape != force.shape:
            raise ValueError(f"torque_nm must have shape {force.shape}, got {torque.shape}")
        if contact.shape != (force.shape[0],):
            raise ValueError(f"contact must have length {force.shape[0]}, got {contact.shape}")
        max_force = float(self.max_force_n)
        work = float(self.work_j)
        if not np.isfinite(max_force) or not np.isfinite(work):
            raise ValueError("max_force_n and work_j must be finite")
        derived_max = float(np.linalg.norm(force, axis=1).max(initial=0.0))
        if not np.isclose(max_force, derived_max, rtol=1e-4, atol=1e-4):
            raise ValueError(
                f"max_force_n={max_force} is inconsistent with vector profile max={derived_max}"
            )
        object.__setattr__(self, "vector_n", force)
        object.__setattr__(self, "torque_nm", torque)
        object.__setattr__(self, "contact", contact)
        object.__setattr__(self, "max_force_n", max_force)
        object.__setattr__(self, "work_j", work)


def _readonly_copy(value: np.ndarray, *, dtype, name: str) -> np.ndarray:
    array = _finite_array(value, dtype=dtype, name=name).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DenseMechanicsLabel:

    time_s: np.ndarray
    tool_position_m: np.ndarray
    tool_quat_xyzw: np.ndarray
    force_reference_n: np.ndarray
    torque_reference_nm: np.ndarray
    force_tool_n: np.ndarray
    torque_tool_nm: np.ndarray
    contact: np.ndarray
    contact_start_s: float
    contact_end_s: float
    max_force_n: float
    work_j: float

    def __post_init__(self) -> None:
        time = _readonly_copy(self.time_s, dtype=np.float64, name="time_s").reshape(-1)
        if time.size < 2:
            raise ValueError("dense mechanics needs at least two timestamps")
        if not np.isclose(time[0], 0.0, atol=1e-9):
            raise ValueError(f"dense time_s must start at zero, got {time[0]}")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("dense time_s must be strictly increasing")
        count = int(time.size)

        position = _readonly_copy(
            self.tool_position_m, dtype=np.float32, name="tool_position_m"
        )
        quaternion = _readonly_copy(
            self.tool_quat_xyzw, dtype=np.float32, name="tool_quat_xyzw"
        )
        if position.shape != (count, 3):
            raise ValueError(f"tool_position_m must be ({count},3), got {position.shape}")
        if quaternion.shape != (count, 4):
            raise ValueError(f"tool_quat_xyzw must be ({count},4), got {quaternion.shape}")
        quaternion_norm = np.linalg.norm(quaternion, axis=1)
        if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
            raise ValueError("tool_quat_xyzw rows must be unit quaternions")
        quaternion = (quaternion / quaternion_norm[:, None]).astype(np.float32)
        quaternion.setflags(write=False)

        names = (
            "force_reference_n",
            "torque_reference_nm",
            "force_tool_n",
            "torque_tool_nm",
        )
        vectors = {
            name: _readonly_copy(getattr(self, name), dtype=np.float32, name=name)
            for name in names
        }
        for name, value in vectors.items():
            if value.shape != (count, 3):
                raise ValueError(f"{name} must be ({count},3), got {value.shape}")

        contact = np.asarray(self.contact, dtype=bool).reshape(-1).copy()
        if contact.shape != (count,):
            raise ValueError(f"contact must have length {count}, got {contact.shape}")
        indices = np.flatnonzero(contact)
        if indices.size < 2:
            raise ValueError("dense mechanics requires at least two contact samples")
        contact.setflags(write=False)
        contact_start = float(self.contact_start_s)
        contact_end = float(self.contact_end_s)
        if not np.isfinite([contact_start, contact_end]).all() or contact_end <= contact_start:
            raise ValueError("contact_start_s/contact_end_s must be finite and ordered")
        if not np.isclose(contact_start, time[indices[0]], atol=1e-7):
            raise ValueError("contact_start_s must equal the first contact timestamp")
        if not np.isclose(contact_end, time[indices[-1]], atol=1e-7):
            raise ValueError("contact_end_s must equal the last contact timestamp")

        rotation = Rotation.from_quat(quaternion).as_matrix()
        predicted_force_reference = np.einsum(
            "nij,nj->ni", rotation, vectors["force_tool_n"]
        )
        predicted_torque_reference = np.einsum(
            "nij,nj->ni", rotation, vectors["torque_tool_nm"]
        )
        if not np.allclose(
            predicted_force_reference, vectors["force_reference_n"], atol=2e-4, rtol=2e-4
        ):
            raise ValueError("force_tool_n and force_reference_n use inconsistent frames")
        if not np.allclose(
            predicted_torque_reference,
            vectors["torque_reference_nm"],
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError("torque_tool_nm and torque_reference_nm use inconsistent frames")

        max_force = float(self.max_force_n)
        expected_max = float(np.linalg.norm(vectors["force_reference_n"], axis=1)[contact].max())
        if not np.isfinite(max_force) or not np.isclose(
            max_force, expected_max, atol=1e-4, rtol=1e-4
        ):
            raise ValueError(
                f"max_force_n={max_force} is inconsistent with dense contact peak={expected_max}"
            )
        force_mid = 0.5 * (
            vectors["force_reference_n"][:-1] + vectors["force_reference_n"][1:]
        )
        displacement = position[1:] - position[:-1]
        expected_work = float(
            np.einsum("ni,ni->n", force_mid, displacement)[contact[:-1]].sum()
        )
        work = float(self.work_j)
        if not np.isfinite(work) or not np.isclose(
            work, expected_work, atol=2e-5, rtol=2e-5
        ):
            raise ValueError(f"work_j={work} is inconsistent with dense integral={expected_work}")

        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "tool_position_m", position)
        object.__setattr__(self, "tool_quat_xyzw", quaternion)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "contact", contact)
        object.__setattr__(self, "contact_start_s", contact_start)
        object.__setattr__(self, "contact_end_s", contact_end)
        object.__setattr__(self, "max_force_n", max_force)
        object.__setattr__(self, "work_j", work)


def _validate_rgb_clip(value, name: str) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"{name} must be (K,H,W,3) uint8, got {rgb.shape} {rgb.dtype}")
    if rgb.shape[0] < 1:
        raise ValueError(f"{name} cannot be empty")
    return rgb


@dataclass(frozen=True)
class PrivilegedClipSignals:
    gripper_rgb: np.ndarray | None = None
    digit_rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.gripper_rgb is not None:
            object.__setattr__(
                self, "gripper_rgb", _validate_rgb_clip(self.gripper_rgb, "gripper_rgb")
            )
        if self.digit_rgb is not None:
            object.__setattr__(self, "digit_rgb", _validate_rgb_clip(self.digit_rgb, "digit_rgb"))


@dataclass(frozen=True)
class HoiClipSample:
    rgb: np.ndarray
    frame_times_s: np.ndarray
    intrinsics: np.ndarray
    task: ClipTask
    object_index: int
    action: str
    prompt: str
    waypoints: WaypointTrajectoryLabel | None = None
    screw: ScrewLabel | None = None
    force: ForceTrajectoryLabel | None = None
    dense_mechanics: DenseMechanicsLabel | None = None
    privileged: PrivilegedClipSignals | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = _validate_rgb_clip(self.rgb, "rgb")
        times = _finite_array(self.frame_times_s, dtype=np.float64, name="frame_times_s").reshape(-1)
        if times.shape != (rgb.shape[0],):
            raise ValueError(f"frame_times_s must have length {rgb.shape[0]}, got {times.shape}")
        if not np.isclose(times[0], 0.0, atol=1e-6) or (
            times.size > 1 and not np.all(np.diff(times) > 0.0)
        ):
            raise ValueError("frame_times_s must start at zero and be strictly increasing")
        intrinsics = _finite_array(self.intrinsics, dtype=np.float64, name="intrinsics")
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3,3), got {intrinsics.shape}")
        task = ClipTask(self.task)
        object_index = int(self.object_index)
        if object_index < 1:
            raise ValueError("object_index must be positive")
        action = str(self.action).strip().lower()
        if action not in {"open", "close"}:
            raise ValueError(f"action must be 'open' or 'close', got {action!r}")
        prompt = str(self.prompt).strip()
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        if self.force is not None and self.waypoints is not None:
            if self.force.vector_n.shape[0] != self.waypoints.num_waypoints:
                raise ValueError("force and waypoint targets must have the same T")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "frame_times_s", times)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "object_index", object_index)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "prompt", prompt)

    @property
    def num_frames(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def num_waypoints(self) -> int:
        return self.waypoints.num_waypoints if self.waypoints is not None else 0

    @property
    def present_labels(self) -> tuple[str, ...]:
        present = []
        if self.waypoints is not None:
            present.append("waypoints")
        if self.screw is not None:
            present.append("screw")
        if self.force is not None:
            present.append("force")
        if self.dense_mechanics is not None:
            present.append("dense_mechanics")
        return tuple(present)
