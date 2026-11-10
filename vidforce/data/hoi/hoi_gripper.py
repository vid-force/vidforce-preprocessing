"""Low-level access to the streams of a session: trigger windows, force series, device trajectory and
calibration, camera poses and frames.
"""
from __future__ import annotations

import csv
import io
import json
import math
import pathlib
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from vidforce.data.hoi.geometry import pose_world_to_camera, resample_pose_track
from vidforce.data.hoi.sample import InteractionSample, ForceLabel, TrajectoryLabel

NS_PER_SECOND = 1_000_000_000.0


@dataclass(frozen=True)
class TriggerWindow:

    index: int
    trigger_start_ns: float
    trigger_end_ns: float
    start_ns: float
    end_ns: float
    peak_trigger: float


@dataclass(frozen=True)
class _FrameRef:
    name: str
    timestamp_ns: float


def _read_csv_member(zip_path: pathlib.Path, suffix: str) -> list[dict[str, str]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"missing Hoi zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"{zip_path} has no member ending {suffix!r}")
        raw = zf.read(matches[0])
    return list(csv.DictReader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="")))


def _read_json_member(zip_path: pathlib.Path, suffix: str) -> dict:
    if not zip_path.exists():
        raise FileNotFoundError(f"missing Hoi zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"{zip_path} has no member ending {suffix!r}")
        return json.loads(zf.read(matches[0]).decode("utf-8"))


def _as_float(row: dict[str, str], key: str) -> float:
    val = row.get(key, "")
    if val == "":
        raise ValueError(f"missing numeric column {key!r}")
    return float(val)


def _dedupe_sorted(
    times: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    order = np.argsort(times)
    times = np.asarray(times, dtype=np.float64)[order]
    arrays = tuple(np.asarray(a)[order] for a in arrays)
    keep = np.ones(times.shape[0], dtype=bool)
    keep[1:] = np.diff(times) > 0
    return (times[keep], *(a[keep] for a in arrays))


def discover_trigger_windows(
    trigger_zip: str | pathlib.Path,
    *,
    threshold: float = 0.5,
    pre_seconds: float = 0.6,
    post_seconds: float = 0.8,
    min_window_seconds: float = 0.25,
) -> list[TriggerWindow]:

    rows = _read_csv_member(pathlib.Path(trigger_zip), "gripper_force_trigger/data.csv")
    if not rows:
        raise ValueError(f"empty trigger stream: {trigger_zip}")
    times = np.asarray([_as_float(r, "timestamp") for r in rows], dtype=np.float64)
    values = np.asarray([_as_float(r, "data") for r in rows], dtype=np.float64)
    times, values = _dedupe_sorted(times, values)
    active = values > float(threshold)

    windows: list[TriggerWindow] = []
    start: int | None = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        last = i == len(active) - 1
        if start is not None and ((not is_active) or last):
            end = i - 1 if not is_active else i
            trigger_start = float(times[start])
            trigger_end = float(times[end])
            duration_s = (trigger_end - trigger_start) / NS_PER_SECOND
            if duration_s >= float(min_window_seconds):
                windows.append(
                    TriggerWindow(
                        index=len(windows),
                        trigger_start_ns=trigger_start,
                        trigger_end_ns=trigger_end,
                        start_ns=trigger_start - float(pre_seconds) * NS_PER_SECOND,
                        end_ns=trigger_end + float(post_seconds) * NS_PER_SECOND,
                        peak_trigger=float(values[start : end + 1].max()),
                    )
                )
            start = None
    if not windows:
        raise ValueError(
            f"no trigger windows in {trigger_zip} above threshold={threshold} "
            f"and min_window_seconds={min_window_seconds}"
        )
    return windows


def _read_force_series(force_zip: pathlib.Path) -> tuple[np.ndarray, np.ndarray, str]:
    rows = _read_csv_member(force_zip, "wrench/data.csv")
    preferred = (
        ("wrench_ext.force.x_filt", "wrench_ext.force.y_filt", "wrench_ext.force.z_filt"),
        ("wrench_ext.force.x", "wrench_ext.force.y", "wrench_ext.force.z"),
        ("wrench.force.x", "wrench.force.y", "wrench.force.z"),
    )
    keys = next((ks for ks in preferred if all(ks[0] in r and ks[1] in r and ks[2] in r for r in rows[:1])), None)
    if keys is None:
        raise ValueError(f"{force_zip} does not contain a known force column triplet")
    times = np.asarray([_as_float(r, "timestamp") for r in rows], dtype=np.float64)
    force = np.asarray([[_as_float(r, keys[0]), _as_float(r, keys[1]), _as_float(r, keys[2])] for r in rows])
    times, force = _dedupe_sorted(times, force)
    if len(times) < 2:
        raise ValueError(f"need >=2 force samples in {force_zip}")
    return times, force.astype(np.float32), ".".join(keys)


def _read_aria_trajectory(slam_zip: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    member = "slam/closed_loop_trajectory_aligned/data.csv"
    rows = _read_csv_member(slam_zip, member)
    times = np.asarray([_as_float(r, "timestamp") for r in rows], dtype=np.float64)
    pos = np.asarray(
        [[_as_float(r, "tx_world_device"), _as_float(r, "ty_world_device"), _as_float(r, "tz_world_device")] for r in rows],
        dtype=np.float64,
    )
    quat = np.asarray(
        [[_as_float(r, "qx_world_device"), _as_float(r, "qy_world_device"), _as_float(r, "qz_world_device"), _as_float(r, "qw_world_device")] for r in rows],
        dtype=np.float64,
    )
    times, pos, quat = _dedupe_sorted(times, pos, quat)
    if len(times) < 2:
        raise ValueError(f"need >=2 trajectory poses in {slam_zip}")
    quat = Rotation.from_quat(quat).as_quat()
    return times, pos.astype(np.float32), quat.astype(np.float32), member.removesuffix("/data.csv")


def _read_aria_calib(calib_zip: pathlib.Path) -> tuple[np.ndarray, np.ndarray, str]:
    raw = _read_json_member(calib_zip, "calib/calib.json")
    if "PINHOLE" not in raw:
        raise ValueError(f"{calib_zip} does not contain PINHOLE calibration")
    pinhole = raw["PINHOLE"]
    K = np.asarray(pinhole["K"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Aria PINHOLE K must be (3,3), got {K.shape}")
    T_device_camera = np.asarray(pinhole.get("T_device_camera"), dtype=np.float64)
    if T_device_camera.shape != (4, 4):
        raise ValueError(f"Aria T_device_camera must be (4,4), got {T_device_camera.shape}")
    return K, T_device_camera, "aria_gripper/calib.json:PINHOLE"


def _read_camera_poses(poses_zip: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv_member(poses_zip, "poses_aligned/data.csv")
    times = np.asarray([_as_float(r, "timestamp") for r in rows], dtype=np.float64)
    pos = np.asarray(
        [[_as_float(r, "tx_world_cam"), _as_float(r, "ty_world_cam"), _as_float(r, "tz_world_cam")] for r in rows],
        dtype=np.float64,
    )
    quat = np.asarray(
        [[_as_float(r, "qx_world_cam"), _as_float(r, "qy_world_cam"), _as_float(r, "qz_world_cam"), _as_float(r, "qw_world_cam")] for r in rows],
        dtype=np.float64,
    )
    return _dedupe_sorted(times, pos, Rotation.from_quat(quat).as_quat())


def _pose_matrix_at(times: np.ndarray, positions: np.ndarray, quats: np.ndarray, timestamp_ns: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(times - float(timestamp_ns))))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quats[idx]).as_matrix()
    T[:3, 3] = positions[idx]
    return T


def _list_frames(camera_zip: pathlib.Path) -> list[_FrameRef]:
    if not camera_zip.exists():
        raise FileNotFoundError(f"missing Hoi RGB zip: {camera_zip}")
    with zipfile.ZipFile(camera_zip) as zf:
        names = [
            n for n in zf.namelist()
            if n.startswith("camera_rgb/") and n.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    frames = [_FrameRef(n, float(pathlib.PurePosixPath(n).stem)) for n in names]
    return sorted(frames, key=lambda f: f.timestamp_ns)


def _select_frame(frames: list[_FrameRef], window: TriggerWindow, policy: str) -> _FrameRef:
    if not frames:
        raise ValueError("empty camera_rgb.zip")
    if policy == "first":
        for f in frames:
            if f.timestamp_ns >= window.start_ns:
                return f
        return frames[-1]
    if policy == "last":
        candidates = [f for f in frames if f.timestamp_ns <= window.end_ns]
        return candidates[-1] if candidates else frames[0]
    if policy == "middle":
        mid = 0.5 * (window.start_ns + window.end_ns)
        return min(frames, key=lambda f: abs(f.timestamp_ns - mid))
    raise ValueError(f"unknown frame_policy {policy!r} (first|middle|last)")


def _decode_frame(camera_zip: pathlib.Path, frame: _FrameRef) -> np.ndarray:
    with zipfile.ZipFile(camera_zip) as zf:
        raw = zf.read(frame.name)
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode {camera_zip}:{frame.name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _heuristic_intrinsics(width: int, height: int, fov_deg: float = 70.0) -> np.ndarray:
    fx = 0.5 * float(width) / math.tan(0.5 * math.radians(fov_deg))
    fy = fx
    return np.asarray(
        [[fx, 0.0, (width - 1.0) / 2.0], [0.0, fy, (height - 1.0) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_dir(base: pathlib.Path, camera_module: str, trial: str) -> pathlib.Path:
    if camera_module.startswith("aria_"):
        return base / camera_module / f"{trial}_vrs"
    return base / camera_module / trial


def _view_role(camera_module: str) -> str:
    if camera_module == "aria_gripper":
        return "gripper_ego"
    if camera_module == "aria_human":
        return "human_ego"
    if camera_module.startswith("iphone"):
        return "static_third_person"
    return camera_module


def _force_window(times: np.ndarray, force_xyz: np.ndarray, window: TriggerWindow) -> tuple[np.ndarray, np.ndarray]:
    mask = (times >= window.start_ns) & (times <= window.end_ns)
    if mask.sum() >= 2:
        return times[mask], force_xyz[mask]
    center = 0.5 * (window.start_ns + window.end_ns)
    idx = int(np.argmin(np.abs(times - center)))
    lo = max(0, idx - 1)
    hi = min(len(times), idx + 2)
    return times[lo:hi], force_xyz[lo:hi]


class HoiGripperWindowDataset:

    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        scene: str = "kitchen_7",
        trial: str = "kitchen_7_1-3-5-7-9_gripper",
        prompt: str = "manipulate the articulated object",
        num_waypoints: int = 16,
        camera_module: str = "iphone_1",
        frame_policy: str = "first",
        trigger_threshold: float = 0.5,
        pre_seconds: float = 0.6,
        post_seconds: float = 0.8,
        min_window_seconds: float = 0.25,
        force_contact_threshold_n: float = 1.0,
        limit: int | None = None,
    ) -> None:
        self.root = pathlib.Path(root)
        self.scene = scene
        self.trial = trial
        self.prompt = prompt
        self.num_waypoints = int(num_waypoints)
        self.camera_module = camera_module
        self.view_role = _view_role(camera_module)
        self.frame_policy = frame_policy
        self.force_contact_threshold_n = float(force_contact_threshold_n)

        base = self.root / scene / "gripper"
        self.camera_dir = _camera_dir(base, camera_module, trial)
        self.aria_dir = base / "aria_gripper" / f"{trial}_vrs"
        self.gripper_dir = base / "gripper" / f"{trial}_bag"
        self.camera_zip = self.camera_dir / "camera_rgb.zip"
        self.slam_zip = self.aria_dir / "slam.zip"
        self.force_zip = self.gripper_dir / "force_torque.zip"
        self.trigger_zip = self.gripper_dir / "gripper_force_trigger.zip"

        self.frames = _list_frames(self.camera_zip)
        self.camera_intrinsics: np.ndarray | None = None
        self.intrinsics_source = "heuristic_fov70"
        self.T_device_camera: np.ndarray | None = None
        if camera_module == "aria_gripper":
            self.camera_times, self.camera_positions_w, self.camera_quats_w, _ = _read_aria_trajectory(
                self.camera_dir / "slam.zip"
            )
            self.camera_intrinsics, self.T_device_camera, self.intrinsics_source = _read_aria_calib(
                self.camera_dir / "calib.zip"
            )
        else:
            self.camera_poses_zip = self.camera_dir / "poses_aligned.zip"
            self.camera_times, self.camera_positions_w, self.camera_quats_w = _read_camera_poses(
                self.camera_poses_zip
            )
        self.traj_times, self.traj_positions_w, self.traj_quats_w, self.trajectory_source = _read_aria_trajectory(self.slam_zip)
        self.force_times, self.force_xyz, self.force_source = _read_force_series(self.force_zip)
        windows = discover_trigger_windows(
            self.trigger_zip,
            threshold=trigger_threshold,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            min_window_seconds=min_window_seconds,
        )
        self.windows = windows[:limit] if limit is not None else windows
        if not self.windows:
            raise ValueError("limit removed all trigger windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> InteractionSample:
        window = self.windows[idx]
        frame = _select_frame(self.frames, window, self.frame_policy)
        rgb = _decode_frame(self.camera_zip, frame)
        height, width = rgb.shape[:2]
        K = self.camera_intrinsics.copy() if self.camera_intrinsics is not None else _heuristic_intrinsics(width, height)
        T_world_cam = _pose_matrix_at(self.camera_times, self.camera_positions_w, self.camera_quats_w, frame.timestamp_ns)
        if self.T_device_camera is not None:


            T_world_cam = T_world_cam @ self.T_device_camera

        t0 = max(window.start_ns, float(self.traj_times[0]))
        t1 = min(window.end_ns, float(self.traj_times[-1]))
        if not t1 > t0:
            raise ValueError(f"window {idx} does not overlap trajectory stream")
        pos_w, quat_w = resample_pose_track(
            self.traj_times,
            self.traj_positions_w,
            self.traj_quats_w,
            self.num_waypoints,
            t_start=t0,
            t_end=t1,
        )
        pos_cam, quat_cam = pose_world_to_camera(pos_w, quat_w, T_world_cam)

        query = np.linspace(t0, t1, self.num_waypoints)
        force_mag_all = np.linalg.norm(self.force_xyz, axis=1)
        force_profile = np.interp(query, self.force_times, force_mag_all).astype(np.float32)
        fw_times, fw_xyz = _force_window(self.force_times, self.force_xyz, window)
        fw_mag = np.linalg.norm(fw_xyz, axis=1)
        max_force = float(fw_mag.max()) if len(fw_mag) else 0.0

        step_force = force_profile[:-1]
        step_dist = np.linalg.norm(np.diff(pos_cam, axis=0), axis=1)
        contact = step_force >= self.force_contact_threshold_n
        work_proxy = float(np.sum(step_force[contact] * step_dist[contact]))

        trajectory = TrajectoryLabel(position=pos_cam, rotation=quat_cam, force_profile=force_profile)
        force = ForceLabel(max_force_n=max_force, work_j=work_proxy)
        interaction = f"{self.scene}/{self.trial}/trigger_{window.index:03d}"
        return InteractionSample(
            rgb=rgb,
            intrinsics=K,
            prompt=self.prompt,
            trajectory=trajectory,
            force=force,
            contact_point=pos_cam[0],
            meta={
                "interaction": interaction,
                "scene": self.scene,
                "trial": self.trial,
                "camera_module": self.camera_module,
                "view_role": self.view_role,
                "window_index": window.index,
                "trigger_start_ns": window.trigger_start_ns,
                "trigger_end_ns": window.trigger_end_ns,
                "window_start_ns": window.start_ns,
                "window_end_ns": window.end_ns,
                "peak_trigger": window.peak_trigger,
                "input_frame": frame.name,
                "input_frame_timestamp": int(frame.timestamp_ns),
                "frame_policy": self.frame_policy,
                "label_source": "hoi_full_real",
                "trajectory_source": self.trajectory_source,
                "force_source": self.force_source,
                "target_frame": "aria_gripper_device_in_input_camera",
                "work_definition": "force_magnitude_path_integral",
                "intrinsics_source": self.intrinsics_source,
                "force_window_samples": int(len(fw_times)),
                "n_frames": len(self.frames),
                "width": width,
                "height": height,
            },
        )
