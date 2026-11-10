"""Confirmed interaction windows and object articulation annotations of a Hoi! session, robust fits of the
articulation axis (line of a prismatic joint, circle of a revolute joint) and the contact-gated work.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from vidforce.data.hoi.sample import JointType


@dataclass(frozen=True)
class ConfirmedInteractionWindow:
    name: str
    start_ns: int
    end_ns: int
    action: str
    object_index: int

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError(
                f"end_ns must be greater than start_ns, got {self.start_ns}..{self.end_ns}"
            )
        if self.action not in {"open", "close"}:
            raise ValueError(f"action must be 'open' or 'close', got {self.action!r}")
        if self.object_index < 1:
            raise ValueError(f"object_index must be a positive integer, got {self.object_index}")

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000_000.0


@dataclass(frozen=True)
class ObjectArticulationAnnotation:
    object_index: int
    joint_type: JointType
    description: str


def _window_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"confirmed window key must end in an integer, got {name!r}")
    return int(match.group(1))


def load_confirmed_interaction_windows(
    source: str | pathlib.Path | Mapping[str, Mapping],
    skip_unindexed: bool = False,
) -> list[ConfirmedInteractionWindow]:

    if isinstance(source, Mapping):
        payload = source
    else:
        path = pathlib.Path(source)
        payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("confirmed interaction metadata must be a non-empty object")

    out: list[ConfirmedInteractionWindow] = []
    for name in sorted(payload, key=_window_number):
        row = payload[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} must map to an object")
        try:
            start_ns = int(row["start_ns"])
            end_ns = int(row["end_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} has invalid start_ns/end_ns") from exc
        action = str(row.get("single_word_description", "")).strip().lower()
        raw_index = row.get("index")
        if raw_index is None:
            if skip_unindexed:
                print(f"[confirmed] {name}: object_index missing — skipping window")
                continue
            raise ValueError(f"{name} object_index is missing")
        try:
            object_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            if skip_unindexed:
                print(f"[confirmed] {name}: object_index invalid {raw_index!r} — skipping window")
                continue
            raise ValueError(f"{name} object_index is invalid: {raw_index!r}") from exc
        out.append(ConfirmedInteractionWindow(name, start_ns, end_ns, action, object_index))
    return out


def load_object_articulation_annotations(
    source: str | pathlib.Path | Mapping,
) -> dict[int, ObjectArticulationAnnotation]:

    if isinstance(source, Mapping):
        payload = source
    else:
        payload = json.loads(pathlib.Path(source).read_text())
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list) or not items:
        raise ValueError("object articulation annotations must contain a non-empty items list")
    type_by_name = {"revolute": JointType.REVOLUTE, "prismatic": JointType.PRISMATIC}
    out: dict[int, ObjectArticulationAnnotation] = {}
    for row in items:
        if not isinstance(row, Mapping):
            raise ValueError("each object annotation must be an object")
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("object annotation has an invalid index") from exc
        if index in out:
            raise ValueError(f"duplicate object annotation index {index}")
        class_name = str(row.get("class", "")).strip().lower()
        if class_name not in type_by_name:
            raise ValueError(f"object {index} has unknown articulation class {class_name!r}")
        description = str(row.get("description", "")).strip()
        if not description:
            raise ValueError(f"object {index} has an empty description")
        out[index] = ObjectArticulationAnnotation(index, type_by_name[class_name], description)
    return out


def contact_gated_work_j_calibrated(
    force_sensor: np.ndarray,
    rotations_world_sensor: np.ndarray,
    tool_positions_world: np.ndarray,
    contact_mask: np.ndarray,
    *,
    force_sign: float = -1.0,
) -> float:

    force = np.asarray(force_sensor, dtype=np.float64)
    rotations = np.asarray(rotations_world_sensor, dtype=np.float64)
    positions = np.asarray(tool_positions_world, dtype=np.float64)
    contact = np.asarray(contact_mask, dtype=bool).reshape(-1)
    if force.ndim != 2 or force.shape[1] != 3:
        raise ValueError(f"force_sensor must be (N,3), got {force.shape}")
    n = force.shape[0]
    if rotations.shape != (n, 3, 3):
        raise ValueError(f"rotations_world_sensor must be ({n},3,3), got {rotations.shape}")
    if positions.shape != (n, 3):
        raise ValueError(f"tool_positions_world must be ({n},3), got {positions.shape}")
    if contact.shape != (n,):
        raise ValueError(f"contact_mask must have length {n}, got {contact.shape}")
    if not np.isfinite(force).all() or not np.isfinite(rotations).all() or not np.isfinite(positions).all():
        raise ValueError("force, rotations, and positions must be finite")
    if n < 2:
        return 0.0

    force_world = float(force_sign) * np.einsum("nij,nj->ni", rotations, force)
    force_mid = 0.5 * (force_world[:-1] + force_world[1:])
    displacement = positions[1:] - positions[:-1]
    segment_work = np.einsum("ni,ni->n", force_mid, displacement)
    return float(segment_work[contact[:-1]].sum())


@dataclass(frozen=True)
class ArticulationAxisFit:
    axis: np.ndarray
    origin: np.ndarray
    residual_rms_m: float
    inlier_fraction: float


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalize a degenerate vector")
    return vector / norm


def _robust_scale(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    return max(1.4826 * mad, 1e-7)


def _tukey_weights(residual: np.ndarray, scale: float, cutoff: float = 4.685) -> np.ndarray:
    u = np.asarray(residual, dtype=np.float64) / (cutoff * scale)
    weights = np.square(1.0 - np.square(u))
    weights[np.abs(u) >= 1.0] = 0.0
    if int(np.count_nonzero(weights)) < 3:
        return np.ones_like(weights)
    return weights


def _weighted_center_cov(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("robust fit has no positive weights")
    center = (points * weights[:, None]).sum(axis=0) / total
    q = points - center
    cov = (q * weights[:, None]).T @ q / total
    return center, cov


def _fit_prismatic(points: np.ndarray) -> ArticulationAxisFit:
    weights = np.ones(points.shape[0], dtype=np.float64)
    axis = np.array([1.0, 0.0, 0.0])
    center = np.median(points, axis=0)
    for _ in range(12):
        center, cov = _weighted_center_cov(points, weights)
        axis = _unit(np.linalg.eigh(cov)[1][:, -1])
        q = points - center
        residual = np.linalg.norm(q - (q @ axis)[:, None] * axis, axis=1)
        weights = _tukey_weights(residual, _robust_scale(residual))
    q = points - center
    residual = np.linalg.norm(q - (q @ axis)[:, None] * axis, axis=1)
    scale = _robust_scale(residual)
    inliers = residual <= max(4.685 * scale, 1e-6)
    return ArticulationAxisFit(
        axis=axis.astype(np.float64),
        origin=center.astype(np.float64),
        residual_rms_m=float(np.sqrt(np.mean(np.square(residual[inliers])))),
        inlier_fraction=float(inliers.mean()),
    )


def _weighted_circle(u: np.ndarray, v: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    design = np.stack([u, v, np.ones_like(u)], axis=1)
    target = u * u + v * v
    root_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
    coef, *_ = np.linalg.lstsq(design * root_w, target * root_w[:, 0], rcond=None)
    a, b = float(coef[0] / 2.0), float(coef[1] / 2.0)
    radius_sq = float(coef[2] + a * a + b * b)
    if radius_sq <= 1e-12:
        raise ValueError("revolute trajectory has a degenerate circle")
    return a, b, float(np.sqrt(radius_sq))


def _robust_plane(points: np.ndarray) -> tuple[np.ndarray, float]:


    support_size = min(points.shape[0], 32)
    support_indices = np.unique(
        np.rint(np.linspace(0, points.shape[0] - 1, support_size)).astype(int)
    )
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_score = (float("inf"), float("inf"))
    for i, j, k in combinations(support_indices.tolist(), 3):
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10:
            continue
        normal /= norm
        offset = float(np.median(points @ normal))
        residual = np.abs(points @ normal - offset)
        trimmed_count = max(3, int(np.ceil(0.8 * residual.size)))
        score = (
            float(np.median(residual)),
            float(np.mean(np.partition(residual, trimmed_count - 1)[:trimmed_count])),
        )
        if score < best_score:
            best_score = score
            best_normal = normal.copy()
            best_offset = offset
    if best_normal is None:
        raise ValueError("revolute trajectory is collinear; no circle plane can be fit")
    return _unit(best_normal), best_offset


def _fit_revolute(points: np.ndarray) -> ArticulationAxisFit:
    axis, plane_offset = _robust_plane(points)
    center = np.median(points, axis=0)
    initial_plane_residual = np.abs(points @ axis - plane_offset)
    weights = _tukey_weights(initial_plane_residual, _robust_scale(initial_plane_residual))
    circle_center = center.copy()
    for _ in range(16):
        center, cov = _weighted_center_cov(points, weights)
        eigvec = np.linalg.eigh(cov)[1]
        axis = _unit(eigvec[:, 0])
        basis_u = _unit(eigvec[:, -1])
        basis_v = _unit(np.cross(axis, basis_u))
        q = points - center
        u, v = q @ basis_u, q @ basis_v
        a, b, radius = _weighted_circle(u, v, weights)
        circle_center = center + a * basis_u + b * basis_v
        plane_residual = np.abs(q @ axis)
        radial = np.sqrt(np.square(u - a) + np.square(v - b))
        radial_residual = np.abs(radial - radius)
        combined = np.hypot(plane_residual, radial_residual)
        weights = _tukey_weights(combined, _robust_scale(combined))
    q = points - center
    u, v = q @ basis_u, q @ basis_v
    plane_residual = np.abs(q @ axis)
    radial = np.sqrt(np.square(u - a) + np.square(v - b))
    combined = np.hypot(plane_residual, np.abs(radial - radius))
    scale = _robust_scale(combined)
    inliers = combined <= max(4.685 * scale, 1e-6)
    return ArticulationAxisFit(
        axis=axis.astype(np.float64),
        origin=circle_center.astype(np.float64),
        residual_rms_m=float(np.sqrt(np.mean(np.square(combined[inliers])))),
        inlier_fraction=float(inliers.mean()),
    )


def fit_articulation_axis(points_world: np.ndarray, joint_type: JointType) -> ArticulationAxisFit:

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError(f"points_world must be (N>=4,3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_world contains non-finite values")
    jt = JointType(int(joint_type))
    if jt == JointType.PRISMATIC:
        return _fit_prismatic(points)
    if jt == JointType.REVOLUTE:
        return _fit_revolute(points)
    raise ValueError("joint_type must be REVOLUTE or PRISMATIC")
