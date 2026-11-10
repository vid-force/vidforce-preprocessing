"""Basic label containers: joint type, trajectory, force, articulation and the interaction sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class JointType(IntEnum):

    UNKNOWN = -1
    REVOLUTE = 0
    PRISMATIC = 1


LABEL_CHANNELS = ("trajectory", "force", "articulation")


def _as_f32(x, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


@dataclass(frozen=True)
class TrajectoryLabel:

    position: np.ndarray
    rotation: np.ndarray
    force_profile: np.ndarray | None = None

    def __post_init__(self) -> None:
        pos = _as_f32(self.position, "position", self.position.shape)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"position must be (T,3), got {pos.shape}")
        T = pos.shape[0]
        if T < 2:
            raise ValueError(f"trajectory needs >=2 waypoints, got T={T}")
        rot = _as_f32(self.rotation, "rotation", (T, 4))
        norms = np.linalg.norm(rot, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"rotation quaternions must be unit-norm, got norms {norms}")
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "rotation", rot)
        if self.force_profile is not None:
            fp = _as_f32(self.force_profile, "force_profile", (T,))
            object.__setattr__(self, "force_profile", fp)

    @property
    def num_waypoints(self) -> int:
        return self.position.shape[0]


@dataclass(frozen=True)
class ForceLabel:

    max_force_n: float
    work_j: float

    def __post_init__(self) -> None:
        for name in ("max_force_n", "work_j"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
            object.__setattr__(self, name, v)


@dataclass(frozen=True)
class ArticulationLabel:

    joint_type: JointType
    axis: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_type", JointType(int(self.joint_type)))
        axis = _as_f32(self.axis, "axis", (3,))
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            raise ValueError("articulation axis is degenerate (near-zero norm)")
        object.__setattr__(self, "axis", (axis / n).astype(np.float32))


@dataclass(frozen=True)
class InteractionSample:


    rgb: np.ndarray
    intrinsics: np.ndarray
    prompt: str
    depth_m: np.ndarray | None = None


    trajectory: TrajectoryLabel | None = None
    force: ForceLabel | None = None
    articulation: ArticulationLabel | None = None
    contact_point: np.ndarray | None = None


    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be (H,W,3) uint8, got {rgb.shape} {rgb.dtype}")
        object.__setattr__(self, "rgb", rgb)
        K = np.asarray(self.intrinsics, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3,3), got {K.shape}")
        object.__setattr__(self, "intrinsics", K)
        if self.depth_m is not None:
            d = np.asarray(self.depth_m, dtype=np.float32)
            if d.shape != rgb.shape[:2]:
                raise ValueError(f"depth {d.shape} != rgb {rgb.shape[:2]}")
            object.__setattr__(self, "depth_m", d)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.contact_point is not None:
            object.__setattr__(
                self, "contact_point", _as_f32(self.contact_point, "contact_point", (3,))
            )

    @property
    def present_labels(self) -> tuple[str, ...]:
        present = []
        if self.trajectory is not None:
            present.append("trajectory")
        if self.force is not None:
            present.append("force")
        if self.articulation is not None:
            present.append("articulation")
        return tuple(present)
