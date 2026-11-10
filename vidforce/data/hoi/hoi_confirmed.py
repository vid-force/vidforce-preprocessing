"""Reader of a gripper session of the Hoi! release: pose track of the tracking device, compensated and filtered
wrench of the force-torque sensor, contact trigger, camera frames and calibration, organised by the confirmed
interaction windows and their object annotations.
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
import zipfile
from dataclasses import dataclass
from itertools import chain
from typing import Iterator, Mapping

import cv2
import numpy as np
import yaml

from vidforce.data.hoi.calibrated_streams import (
    GripperCalibration,
    aria_rectified_camera_transform,
    interpolate_pose_matrices,
    nearest_boolean_samples,
)
from vidforce.data.hoi.clip_sample import (
    ClipTask,
    HoiClipSample,
    PrivilegedClipSignals,
    ScrewLabel,
)
from vidforce.data.hoi.confirmed_interactions import (
    ArticulationAxisFit,
    ConfirmedInteractionWindow,
    ObjectArticulationAnnotation,
    fit_articulation_axis,
    load_confirmed_interaction_windows,
    load_object_articulation_annotations,
)
from vidforce.data.hoi.interaction_labels import build_calibrated_interaction_labels


@dataclass(frozen=True)
class _ImageRef:
    member: str
    timestamp_ns: float


@dataclass(frozen=True)
class _PoseTrack:
    times_ns: np.ndarray
    positions_world: np.ndarray
    quats_xyzw_world: np.ndarray


def _read_zip_member(path: pathlib.Path, suffix: str) -> bytes:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one member ending {suffix!r} in {path}, found {matches}"
            )
        return archive.read(matches[0])


def _iter_csv(path: pathlib.Path, suffix: str) -> Iterator[dict[str, str]]:

    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one member ending {suffix!r} in {path}, found {matches}"
            )
        with archive.open(matches[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            yield from csv.DictReader(text)


def _read_json(path: pathlib.Path, suffix: str) -> dict:
    return json.loads(_read_zip_member(path, suffix).decode("utf-8"))


def _load_json_source(
    source: str | pathlib.Path | Mapping,
    *,
    zip_suffix: str,
) -> Mapping:

    if isinstance(source, Mapping):
        return source
    path = pathlib.Path(source)
    if path.suffix.lower() == ".zip":
        return _read_json(path, zip_suffix)
    return json.loads(path.read_text())


def _dedupe_sorted(times: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(np.asarray(times, dtype=np.float64))
    t = np.asarray(times, dtype=np.float64)[order]
    arrays = tuple(np.asarray(value)[order] for value in values)
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    return (t[keep], *(value[keep] for value in arrays))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing/invalid numeric CSV field {key!r}") from exc


def _read_pose_track(
    path: pathlib.Path,
    member_suffix: str = "slam/closed_loop_trajectory_aligned/data.csv",
) -> _PoseTrack:
    records = []
    for row in _iter_csv(path, member_suffix):
        records.append(
            (
                _float(row, "timestamp"),
                _float(row, "tx_world_device"),
                _float(row, "ty_world_device"),
                _float(row, "tz_world_device"),
                _float(row, "qx_world_device"),
                _float(row, "qy_world_device"),
                _float(row, "qz_world_device"),
                _float(row, "qw_world_device"),
            )
        )
    if not records:
        raise ValueError(f"empty pose CSV in {path}")
    values = np.asarray(records, dtype=np.float64)
    times, positions, quats = values[:, 0], values[:, 1:4], values[:, 4:8]
    times, positions, quats = _dedupe_sorted(times, positions, quats)
    if times.size < 2:
        raise ValueError(f"pose track {path} contains fewer than two unique timestamps")
    return _PoseTrack(times, positions.astype(np.float64), quats.astype(np.float64))


def _read_wrench(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = iter(_iter_csv(path, "wrench/data.csv"))
    first = next(rows, None)
    if first is None:
        raise ValueError(f"empty wrench CSV in {path}")
    force_candidates = (
        ("wrench_ext.force.x_filt", "wrench_ext.force.y_filt", "wrench_ext.force.z_filt"),
        ("wrench_ext.force.x", "wrench_ext.force.y", "wrench_ext.force.z"),
    )
    torque_candidates = (
        ("wrench_ext.torque.x_filt", "wrench_ext.torque.y_filt", "wrench_ext.torque.z_filt"),
        ("wrench_ext.torque.x", "wrench_ext.torque.y", "wrench_ext.torque.z"),
    )
    force_keys = next((keys for keys in force_candidates if all(key in first for key in keys)), None)
    torque_keys = next((keys for keys in torque_candidates if all(key in first for key in keys)), None)
    if force_keys is None or torque_keys is None:
        raise ValueError(f"{path} lacks compensated force and torque columns")
    records = [
        (
            _float(row, "timestamp"),
            *(_float(row, key) for key in force_keys),
            *(_float(row, key) for key in torque_keys),
        )
        for row in chain([first], rows)
    ]
    values = np.asarray(records, dtype=np.float64)
    times, force, torque = values[:, 0], values[:, 1:4], values[:, 4:7]
    times, force, torque = _dedupe_sorted(times, force, torque)
    return times, force.astype(np.float64), torque.astype(np.float64)


def _read_trigger(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    records = [
        (_float(row, "timestamp"), _float(row, "data"))
        for row in _iter_csv(path, "gripper_force_trigger/data.csv")
    ]
    if not records:
        raise ValueError(f"empty trigger CSV in {path}")
    array = np.asarray(records, dtype=np.float64)
    times, values = array[:, 0], array[:, 1]
    times, values = _dedupe_sorted(times, values)
    return times, values


def _list_images(path: pathlib.Path, prefix: str) -> list[_ImageRef]:
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith(prefix.rstrip("/") + "/")
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    refs: list[_ImageRef] = []
    for name in names:
        try:
            timestamp = float(pathlib.PurePosixPath(name).stem)
        except ValueError:
            continue
        refs.append(_ImageRef(name, timestamp))
    refs.sort(key=lambda ref: ref.timestamp_ns)
    if not refs:
        raise ValueError(f"no timestamped images under {prefix!r} in {path}")
    return refs


def _select_clip(refs: list[_ImageRef], window: ConfirmedInteractionWindow, count: int) -> list[_ImageRef]:
    candidates = [ref for ref in refs if window.start_ns <= ref.timestamp_ns <= window.end_ns]
    if len(candidates) < count:
        raise ValueError(
            f"{window.name} has {len(candidates)} in-window frames, fewer than requested {count}"
        )
    indices = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    if np.unique(indices).size != count:
        raise RuntimeError("uniform frame selection produced duplicate indices")
    return [candidates[index] for index in indices]


def _nearest_images(refs: list[_ImageRef], target_times_ns: np.ndarray) -> list[_ImageRef]:
    times = np.asarray([ref.timestamp_ns for ref in refs], dtype=np.float64)
    out = []
    for target in np.asarray(target_times_ns, dtype=np.float64):
        after = int(np.clip(np.searchsorted(times, target, side="left"), 0, times.size - 1))
        before = max(after - 1, 0)
        index = after if abs(times[after] - target) < abs(target - times[before]) else before
        out.append(refs[index])
    return out


def _decode_images(path: pathlib.Path, refs: list[_ImageRef]) -> np.ndarray:
    images = []
    with zipfile.ZipFile(path) as archive:
        for ref in refs:
            bgr = cv2.imdecode(np.frombuffer(archive.read(ref.member), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"failed to decode {path}:{ref.member}")
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError(f"clip images in {path} do not have one consistent resolution")
    return np.stack(images)


def _camera_calibration(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = _read_json(path, "calib/calib.json")
    try:
        pinhole = payload["PINHOLE"]
        intrinsics = np.asarray(pinhole["K"], dtype=np.float64)
        T_device_camera = aria_rectified_camera_transform(payload)
    except KeyError as exc:
        raise ValueError(f"invalid Aria calibration {path}: missing {exc}") from exc
    if intrinsics.shape != (3, 3) or T_device_camera.shape != (4, 4):
        raise ValueError(f"invalid Aria calibration matrix shape in {path}")
    return intrinsics, T_device_camera, payload


def _prompt(action: str, annotation: ObjectArticulationAnnotation) -> str:
    description = annotation.description
    if description.lower().startswith("object:"):
        description = description.split(":", 1)[1].split("FE:", 1)[0].strip().rstrip(".")
    return f"{action} {description}".strip()


class HoiConfirmedClipDataset:

    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        scene: str = "kitchen_14",
        trial: str = "kitchen_14_1-14_gripper",
        confirmed_path: str | pathlib.Path | Mapping | None = None,
        instances_path: str | pathlib.Path | Mapping | None = None,
        task: ClipTask = ClipTask.FORCE,
        num_frames: int = 4,
        num_waypoints: int = 32,
        num_digit_frames: int = 64,
        input_view: str = "aria_human",
        public_prompt: str | None = None,
        public_action: str | None = None,
        object_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        include_privileged: bool = True,
        contact_threshold: float = 0.5,
        force_sign: float = -1.0,
    ) -> None:
        self.root = pathlib.Path(root)
        self.scene = scene
        self.trial = trial
        self.task = ClipTask(task)
        self.num_frames = int(num_frames)
        self.num_waypoints = int(num_waypoints)
        self.num_digit_frames = int(num_digit_frames)
        self.input_view = str(input_view)
        self.public_prompt = None if public_prompt is None else str(public_prompt).strip()
        self.public_action = None if public_action is None else str(public_action).strip().lower()
        self.include_privileged = bool(include_privileged)
        self.contact_threshold = float(contact_threshold)
        self.force_sign = float(force_sign)
        if self.num_frames < 1 or self.num_waypoints < 3 or self.num_digit_frames < 2:
            raise ValueError(
                "num_frames must be >=1, num_waypoints >=3, and num_digit_frames >=2"
            )
        if self.input_view not in {"aria_human", "aria_gripper"}:
            raise ValueError("input_view must be 'aria_human' or 'aria_gripper'")
        if self.public_prompt is not None and not self.public_prompt:
            raise ValueError("public_prompt cannot be empty")
        if self.public_action is not None and self.public_action not in {"open", "close"}:
            raise ValueError("public_action must be open or close when explicitly provided")
        if self.input_view == "aria_gripper" and self.include_privileged:
            raise ValueError("gripper RGB cannot simultaneously be public and privileged input")

        scene_root = self.root / scene
        modality_root = scene_root / "gripper"
        self.human_dir = modality_root / "aria_human" / f"{trial}_vrs"
        self.gripper_aria_dir = modality_root / "aria_gripper" / f"{trial}_vrs"
        self.rig_dir = modality_root / "gripper" / f"{trial}_bag"
        trial_indices = trial.removeprefix(f"{scene}_").removesuffix("_gripper")
        confirmed = confirmed_path if confirmed_path is not None else (
            modality_root / f"interaction_splitting_info_{trial_indices}_confirmed.json"
        )
        if instances_path is not None:
            instances = instances_path
        else:
            zipped_instances = scene_root / "leica/001/instance_annotations_3d.zip"
            plain_instances = (
                scene_root / "leica/001/instance_annotations_3d/instances.json"
            )
            instances = zipped_instances if zipped_instances.exists() else plain_instances
        confirmed_payload = _load_json_source(
            confirmed, zip_suffix="interaction_splitting_info.json"
        )
        instances_payload = _load_json_source(
            instances, zip_suffix="instance_annotations_3d/instances.json"
        )
        windows = load_confirmed_interaction_windows(
            confirmed_payload, skip_unindexed=True)
        if object_indices is not None:
            allowed = {int(index) for index in object_indices}
            windows = [window for window in windows if window.object_index in allowed]
        if not windows:
            raise ValueError("no confirmed windows remain after object filtering")
        self.windows = windows
        self.annotations = load_object_articulation_annotations(instances_payload)
        missing = sorted({window.object_index for window in windows} - self.annotations.keys())
        if missing:
            raise ValueError(f"confirmed objects have no articulation annotation: {missing}")

        self.human_rgb_zip = self.human_dir / "camera_rgb.zip"
        self.gripper_rgb_zip = self.gripper_aria_dir / "camera_rgb.zip"
        self.digit_zip = self.rig_dir / "digit.zip"
        self.human_frames = _list_images(self.human_rgb_zip, "camera_rgb")
        self.gripper_frames = (
            _list_images(self.gripper_rgb_zip, "camera_rgb") if self.include_privileged else []
        )
        if self.include_privileged and self.digit_zip.exists():
            self.digit_left = _list_images(self.digit_zip, "digit/left/image_raw")
            self.digit_right = _list_images(self.digit_zip, "digit/right/image_raw")
        else:


            self.digit_left, self.digit_right = [], []

        self.human_pose = _read_pose_track(self.human_dir / "slam.zip")
        self.gripper_pose = _read_pose_track(self.gripper_aria_dir / "slam.zip")
        human_intrinsics, self.T_human_device_camera, _ = _camera_calibration(
            self.human_dir / "calib.zip"
        )
        gripper_intrinsics, self.T_gripper_device_camera, gripper_aria_calib = (
            _camera_calibration(self.gripper_aria_dir / "calib.zip")
        )
        if self.input_view == "aria_human":
            self.input_rgb_zip = self.human_rgb_zip
            self.input_frames = self.human_frames
            self.input_pose = self.human_pose
            self.T_input_device_camera = self.T_human_device_camera
            self.intrinsics = human_intrinsics
        else:
            self.input_rgb_zip = self.gripper_rgb_zip
            self.input_frames = _list_images(self.gripper_rgb_zip, "camera_rgb")
            self.input_pose = self.gripper_pose
            self.T_input_device_camera = self.T_gripper_device_camera
            self.intrinsics = gripper_intrinsics
        gripper_kalibr = yaml.safe_load(
            _read_zip_member(
                self.rig_dir / "calib.zip",
                "calib/merged_calibration-camchain-imucam.yaml",
            ).decode("utf-8")
        )
        gripper_imu = yaml.safe_load(
            _read_zip_member(
                self.rig_dir / "calib.zip",
                "calib/merged_calibration-imu.yaml",
            ).decode("utf-8")
        )
        if not isinstance(gripper_kalibr, dict) or not isinstance(gripper_imu, dict):
            raise ValueError("gripper camera/IMU calibration YAML must contain mappings")
        overlap = gripper_kalibr.keys() & gripper_imu.keys()
        if overlap:
            raise ValueError(f"duplicate camera/IMU calibration sections: {sorted(overlap)}")
        gripper_kalibr.update(gripper_imu)
        self.calibration = GripperCalibration.from_release_metadata(
            gripper_aria_calib, gripper_kalibr
        )
        self.force_times, self.force_sensor, self.torque_sensor = _read_wrench(
            self.rig_dir / "force_torque.zip"
        )
        self.trigger_times, self.trigger_values = _read_trigger(
            self.rig_dir / "gripper_force_trigger.zip"
        )
        self.object_axis_fits, self.object_axis_counts = self._fit_object_axes()

    def _fit_object_axes(
        self,
    ) -> tuple[dict[int, ArticulationAxisFit], dict[int, tuple[int, int]]]:
        points_by_object: dict[int, list[np.ndarray]] = {}
        counts_by_object: dict[int, list[int]] = {}
        for window in self.windows:


            mask = (
                (self.force_times >= window.start_ns)
                & (self.force_times <= window.end_ns)
                & (self.force_times >= self.gripper_pose.times_ns[0])
                & (self.force_times <= self.gripper_pose.times_ns[-1])
                & (self.force_times >= self.trigger_times[0])
                & (self.force_times <= self.trigger_times[-1])
            )
            query = self.force_times[mask]
            contact = nearest_boolean_samples(
                self.trigger_times,
                self.trigger_values,
                query,
                threshold=self.contact_threshold,
            )
            if int(contact.sum()) < 4:
                raise ValueError(f"{window.name} has fewer than four contact pose timestamps")
            device = interpolate_pose_matrices(
                self.gripper_pose.times_ns,
                self.gripper_pose.positions_world,
                self.gripper_pose.quats_xyzw_world,
                query,
            )
            tool = device @ self.calibration.T_device_tool
            points_by_object.setdefault(window.object_index, []).append(tool[contact, :3, 3])
            count = counts_by_object.setdefault(window.object_index, [0, 0])
            count[0] += int(contact.sum())
            count[1] += int(query.size)
        fits = {
            object_index: fit_articulation_axis(
                np.concatenate(parts, axis=0), self.annotations[object_index].joint_type
            )
            for object_index, parts in points_by_object.items()
        }
        return fits, {index: (counts[0], counts[1]) for index, counts in counts_by_object.items()}

    def __len__(self) -> int:
        return len(self.windows)

    def _reference_camera(self, timestamp_ns: float) -> np.ndarray:
        device = interpolate_pose_matrices(
            self.input_pose.times_ns,
            self.input_pose.positions_world,
            self.input_pose.quats_xyzw_world,
            np.asarray([timestamp_ns]),
        )[0]
        return device @ self.T_input_device_camera

    def _privileged(
        self, frame_times_ns: np.ndarray, window: ConfirmedInteractionWindow
    ) -> PrivilegedClipSignals | None:
        if not self.include_privileged:
            return None
        gripper = _decode_images(
            self.gripper_rgb_zip, _nearest_images(self.gripper_frames, frame_times_ns)
        )
        if not self.digit_left:
            return PrivilegedClipSignals(gripper_rgb=gripper, digit_rgb=None)
        left_refs = _select_clip(self.digit_left, window, self.num_digit_frames)
        digit_times = np.asarray([ref.timestamp_ns for ref in left_refs], dtype=np.float64)
        left = _decode_images(self.digit_zip, left_refs)
        right = _decode_images(self.digit_zip, _nearest_images(self.digit_right, digit_times))
        if left.shape[:3] != right.shape[:3]:
            raise ValueError("left/right DIGIT clips do not share shape")
        return PrivilegedClipSignals(gripper_rgb=gripper, digit_rgb=np.concatenate([left, right], axis=2))

    def __getitem__(self, index: int) -> HoiClipSample:
        window = self.windows[index]
        refs = _select_clip(self.input_frames, window, self.num_frames)
        frame_times = np.asarray([ref.timestamp_ns for ref in refs], dtype=np.float64)
        rgb = _decode_images(self.input_rgb_zip, refs)
        T_world_reference_camera = self._reference_camera(frame_times[0])
        labels = build_calibrated_interaction_labels(
            window,
            device_times_ns=self.gripper_pose.times_ns,
            device_positions_world=self.gripper_pose.positions_world,
            device_quats_xyzw_world=self.gripper_pose.quats_xyzw_world,
            force_times_ns=self.force_times,
            force_sensor_n=self.force_sensor,
            torque_sensor_nm=self.torque_sensor,
            trigger_times_ns=self.trigger_times,
            trigger_values=self.trigger_values,
            calibration=self.calibration,
            T_world_reference_camera=T_world_reference_camera,
            num_waypoints=self.num_waypoints,
            contact_threshold=self.contact_threshold,
            force_sign=self.force_sign,
        )
        fit = self.object_axis_fits[window.object_index]
        contact_count, total_count = self.object_axis_counts[window.object_index]
        T_reference_world = np.linalg.inv(T_world_reference_camera)
        axis_reference = T_reference_world[:3, :3] @ fit.axis
        origin_reference = (T_reference_world @ np.r_[fit.origin, 1.0])[:3]
        confidence = float(
            np.clip(fit.inlier_fraction * np.exp(-fit.residual_rms_m / 0.02), 0.0, 1.0)
        )
        annotation = self.annotations[window.object_index]
        screw = ScrewLabel(annotation.joint_type, axis_reference, origin_reference, confidence)
        return HoiClipSample(
            rgb=rgb,
            frame_times_s=(frame_times - frame_times[0]) / 1_000_000_000.0,
            intrinsics=self.intrinsics,
            task=self.task,
            object_index=window.object_index,
            action=self.public_action or window.action,
            prompt=self.public_prompt or _prompt(window.action, annotation),
            waypoints=labels.waypoints,
            screw=screw,
            force=labels.force,
            dense_mechanics=labels.dense_mechanics,
            privileged=self._privileged(frame_times, window),
            meta={
                "scene": self.scene,
                "trial": self.trial,
                "window_name": window.name,
                "input_view": self.input_view,
                "label_action": window.action,
                "public_action_policy": (
                    "official_label" if self.public_action is None else f"constant_{self.public_action}"
                ),
                "label_frame": "first_input_camera",
                "trajectory_source": "calibrated_gripper_tool",
                "wrench_source": "compensated_filtered_bota",
                "work_definition": "contact_gated_trapezoidal_force_dot_tool_displacement",
                "force_sign": self.force_sign,
                "object_axis_residual_rms_m": fit.residual_rms_m,
                "object_axis_inlier_fraction": fit.inlier_fraction,
                "object_axis_fit_contact_samples": contact_count,
                "object_axis_fit_total_samples": total_count,
                "privileged_digit_frames": self.num_digit_frames if self.include_privileged else 0,
                "waypoint_phase": "first_to_last_contact",
                "waypoint_start_ns": labels.waypoint_start_ns,
                "waypoint_end_ns": labels.waypoint_end_ns,
            },
        )
