"""Record of one human recording seen from a third-person camera.

``build_hand_record`` reads one confirmed interaction window of a human session: frames of the chosen camera, the
thumb-tip trajectory of the manipulating hand from the hand tracking of the head-worn device, mapped into the camera
frame at 16 waypoints, the camera pose, the prompt, and depth (the phone's depth stream, or Depth Anything 3 with
intrinsics for the first frame or for every frame). Human recordings carry no force. ``HandRecord`` validates the
result.
"""
from __future__ import annotations

import csv
import fnmatch
import io
import pathlib
import zipfile
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from vidforce.data.hoi.confirmed_interactions import load_confirmed_interaction_windows
from vidforce.data.hoi.geometry import pose_world_to_camera
from vidforce.data.hoi.hoi_gripper import _pose_matrix_at, _read_camera_poses

try:
    import cv2 as _cv2
except ImportError as _e:
    raise ImportError("hand record build requires cv2") from _e


NS = 1_000_000_000.0
N_WAYPOINTS = 16
MAX_FRAME_GAP_S = 0.08
MAX_POSE_GAP_S = 0.5
MIN_ARC_LENGTH_M = 0.05


MAX_ARC_LENGTH_M = 6.0
DEFAULT_FRAME_SIZE = 640

PHONE_PREFIXES = ("tx_world_cam", "ty_world_cam", "tz_world_cam",
                  "qx_world_cam", "qy_world_cam", "qz_world_cam", "qw_world_cam")


class _Layout:

    def __init__(self, path: pathlib.Path, *, prefer: str | None = None):
        self.path = pathlib.Path(path)
        self.is_zip = self.path.is_file() and self.path.suffix.lower() == ".zip"
        self.prefer = prefer.replace("\\", "/").strip("/") if prefer else None

    def _members(self, pattern: str):
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                names = archive.namelist()
        else:
            names = [str(p.relative_to(self.path)).replace("\\", "/")
                     for p in self.path.rglob(pattern)]


        out = []
        for n in names:
            if fnmatch.fnmatch(posix_leaf(n), pattern):
                if self.prefer is None or fnmatch.fnmatch(n, self.prefer):
                    out.append(n)
        return out

    def _zip_read(self, member: str) -> bytes:
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(member)

    def _dir_read(self, member: str) -> bytes:
        return (self.path / member).read_bytes()

    def read(self, pattern: str) -> bytes:
        members = self._members(pattern)
        if len(members) != 1:
            raise FileNotFoundError(
                f"{self.path}: expected exactly one member matching {pattern!r}"
                f"{' under ' + self.prefer if self.prefer else ''}, found {members}"
            )
        return self._zip_read(members[0]) if self.is_zip else self._dir_read(members[0])

    def read_csv(self, suffix: str) -> list[dict[str, str]]:
        raw = self.read(suffix)
        return list(csv.DictReader(
            io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="")
        ))

    def read_json(self, suffix: str) -> dict:
        import json
        return json.loads(self.read(suffix).decode("utf-8"))

    def list_images(self) -> list[_ImageLike]:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                names = [n for n in archive.namelist()
                         if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:

            names = [p.name for p in self.path.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        refs: list[_ImageLike] = []
        for name in names:
            try:
                stamp = float(pathlib.Path(name).stem)
            except ValueError:
                continue
            refs.append(_ImageLike(name, stamp))
        refs.sort(key=lambda r: r.timestamp_ns)
        if not refs:
            raise ValueError(f"no timestamped images in {self.path}")
        return refs

    def decode_image(self, name: str) -> np.ndarray:
        data = self._zip_read(name) if self.is_zip else (self.path / name).read_bytes()
        bgr = _cv2.imdecode(np.frombuffer(data, np.uint8), _cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode {self.path}:{name}")
        return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)

    def has_member(self, name: str) -> bool:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return any(n.endswith(name) for n in archive.namelist())
        return (self.path / name).exists()

    def list_members(self, suffix: str) -> list[str]:
        return self._members(suffix)

    def read_npy(self, name: str) -> np.ndarray:
        data = self._zip_read(name) if self.is_zip else (self.path / name).read_bytes()
        return np.load(io.BytesIO(data), allow_pickle=False)


def posix_leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return frame
    out = _cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=_cv2.INTER_AREA)
    return out


@dataclass(frozen=True)
class _ImageLike:
    member: str
    timestamp_ns: float


@dataclass(frozen=True)
class HandRecord:

    rgb: np.ndarray
    frame_times_ns: np.ndarray
    depth_map: np.ndarray
    traj_m: np.ndarray
    waypoint_times_s: np.ndarray
    force_valid: bool
    force_vec_n: np.ndarray
    peak_force_n: float
    contact: np.ndarray
    T_world_iphone: np.ndarray
    prompt: str
    provenance: dict


    depth_K: np.ndarray | None = None
    depth_maps: np.ndarray | None = None
    depth_Ks: np.ndarray | None = None

    @property
    def arc_length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.traj_m, axis=0), axis=1).sum())

    def __post_init__(self) -> None:
        T = self.traj_m.shape[0]
        if self.rgb.ndim != 4 or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be (K,H,W,3) uint8, got {self.rgb.shape} {self.rgb.dtype}")
        if self.frame_times_ns.shape != (self.rgb.shape[0],):
            raise ValueError("frame_times_ns must have one entry per frame")
        if self.depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2-D, got {self.depth_map.shape}")
        if self.depth_K is None and self.depth_map.shape[:2] != self.rgb.shape[1:3]:

            raise ValueError("depth_map must be (H, W) at rgb resolution, "
                             f"got {self.depth_map.shape} vs rgb {self.rgb.shape}")
        if self.depth_K is not None and self.depth_K.shape != (3, 3):
            raise ValueError("depth_K must be (3,3)")
        if self.depth_maps is not None:
            if self.depth_maps.ndim != 3 or self.depth_maps.shape[0] != self.rgb.shape[0]:
                raise ValueError("depth_maps must be (K, H', W') with one map per frame")
            if self.depth_Ks is None or self.depth_Ks.shape != (self.rgb.shape[0], 3, 3):
                raise ValueError("depth_Ks must be (K,3,3) alongside depth_maps")
        if not np.isfinite(self.depth_map).any():
            raise ValueError("depth_map is entirely non-finite")
        if self.traj_m.shape != (T, 3) or self.waypoint_times_s.shape != (T,):
            raise ValueError("traj_m/waypoint_times_s must be (16,3)/(16,)")
        if self.force_vec_n.shape != (T, 3) or self.contact.shape != (T,):
            raise ValueError("force_vec_n/contact must match (16,3)/(16,)")
        if not np.isfinite(self.traj_m).all():
            raise ValueError("traj_m contains non-finite values")
        if not np.all(np.diff(self.waypoint_times_s) > 0.0):
            raise ValueError("waypoint_times_s must be strictly increasing")
        if self.T_world_iphone.shape != (4, 4):
            raise ValueError("T_world_iphone must be (4,4)")


def _nearest_images(refs: list[_ImageLike], targets: np.ndarray) -> list[_ImageLike]:
    times = np.asarray([r.timestamp_ns for r in refs], dtype=np.float64)
    out = []
    for target in np.asarray(targets, dtype=np.float64):
        after = int(np.clip(np.searchsorted(times, target, side="left"), 0, times.size - 1))
        before = max(after - 1, 0)
        idx = after if abs(times[after] - target) < abs(target - times[before]) else before
        out.append(refs[idx])
    return out


def read_depth_map(depth_layout: _Layout, frame_stamp_ns: float,
                   rgb_h: int, rgb_w: int) -> np.ndarray:
    members = depth_layout.list_members("*.npy")
    stamps = []
    for name in members:
        try:
            stamps.append(float(pathlib.Path(name).stem))
        except ValueError:
            continue
    if not stamps:
        raise FileNotFoundError(
            f"{depth_layout.path}: no timestamped .npy depth frames")
    stamps = np.asarray(stamps, dtype=np.float64)
    i = int(np.argmin(np.abs(stamps - float(frame_stamp_ns))))
    d = depth_layout.read_npy(members[i]).astype(np.float64)
    if d.ndim != 2:
        raise ValueError(f"depth frame {members[i]} not 2D: {d.shape}")


    d = _cv2.resize(d, (rgb_w, rgb_h), interpolation=_cv2.INTER_LINEAR)
    return d.astype(np.float32)


def _read_aria_slam(layout: _Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = layout.read_csv("data.csv")
    if not rows:
        raise ValueError(f"empty slam trajectory in {layout.path}")
    times = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    pos = np.asarray(
        [[float(r["tx_world_device"]), float(r["ty_world_device"]), float(r["tz_world_device"])]
         for r in rows], dtype=np.float64,
    )
    quat = np.asarray(
        [[float(r["qx_world_device"]), float(r["qy_world_device"]),
          float(r["qz_world_device"]), float(r["qw_world_device"])] for r in rows],
        dtype=np.float64,
    )
    order = np.argsort(times)
    times, pos, quat = times[order], pos[order], quat[order]
    keep = np.ones(times.size, dtype=bool)
    keep[1:] = np.diff(times) > 0
    times, pos, quat = times[keep], pos[keep], quat[keep]
    if times.size < 2:
        raise ValueError(f"slam trajectory {layout.path} has <2 unique timestamps")

    quat = Rotation.from_quat(quat).as_quat()
    return times, pos.astype(np.float32), quat.astype(np.float32)


def _read_hand_tracking(layout: _Layout) -> tuple[np.ndarray, dict]:
    times, by_time = [], {}
    for row in layout.read_csv("data.csv"):
        by_time[float(row["timestamp"])] = row
        times.append(float(row["timestamp"]))
    times = np.asarray(sorted(times), dtype=np.float64)
    return times, by_time


def _choose_hand_window(
    rows_by_time, window_start, window_end, *, use_hand: str | None = None,
) -> str:
    if use_hand is not None:
        return use_hand
    conf = {"left": [], "right": []}
    for t, row in rows_by_time.items():
        if not window_start <= t <= window_end:
            continue
        for hand in ("left", "right"):
            raw = row.get(f"{hand}_tracking_confidence")
            if raw in (None, ""):
                continue
            try:
                conf[hand].append(max(float(raw), 0.0))
            except ValueError:
                pass
    left = float(np.mean(conf["left"])) if conf["left"] else 0.0
    right = float(np.mean(conf["right"])) if conf["right"] else 0.0
    if left == right:

        left = sum(1 for t, row in rows_by_time.items()
                   if window_start <= t <= window_end and _row_fields_ok(row, "left"))
        right = sum(1 for t, row in rows_by_time.items()
                    if window_start <= t <= window_end and _row_fields_ok(row, "right"))
    return "right" if right >= left else "left"


def _row_fields_ok(row, hand: str) -> bool:
    try:
        values = []
        for k in (f"tx_{hand}_landmark_0_device", f"ty_{hand}_landmark_0_device",
                  f"tz_{hand}_landmark_0_device"):
            v = row.get(k, "")
            if v == "":
                return False
            values.append(float(v))
        return float(np.abs(values).sum()) > 1e-9
    except (ValueError, TypeError):
        return False


def _nearest_visible(v_world: np.ndarray, ok: np.ndarray) -> np.ndarray:
    v = np.where(ok[:, None], v_world, np.nan)
    idx = np.arange(len(v))
    good = np.flatnonzero(ok)
    if good.size == 0:
        raise ValueError("no visible landmark samples in window")

    nearest_good = good[np.argmin(np.abs(idx[:, None] - good[None, :]), axis=1)]
    return np.where(ok[:, None], v_world, v_world[nearest_good])


def _waypoint_positions(world_pos: np.ndarray, times_s: np.ndarray,
                         wpt_times_s: np.ndarray) -> np.ndarray:
    out = np.empty((wpt_times_s.size, 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(wpt_times_s, times_s, world_pos[:, k])
    return out


def _observation_times(window, *, num_frames: int) -> np.ndarray:
    if num_frames < 2:
        raise ValueError("num_frames must be >= 2")
    return np.linspace(float(window.start_ns), float(window.end_ns), num_frames)


def _prompt(action: str, annotation=None) -> str:
    desc = ""
    if annotation is not None:
        d = annotation.get("description") or getattr(annotation, "description", None)
        if d:
            if str(d).lower().startswith("object:"):
                d = str(d).split(":", 1)[1].split("FE:", 1)[0].strip().rstrip(".")
            desc = f" {d}"
    return f"{action}{desc}".strip()


def _world_to_camera(world: np.ndarray, T_world_iphone: np.ndarray) -> np.ndarray:

    n = world.shape[0]
    ident = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    pos_cam, _ = pose_world_to_camera(world, ident, T_world_iphone)
    return pos_cam


MANIPULATION_SPAN_S = 2.5
_SPEED_SMOOTH_S = 0.2


def _trim_to_manipulation(window, slam_times, slam_pos):
    import dataclasses
    times = np.asarray(slam_times, dtype=np.float64)
    sel = (times >= window.start_ns) & (times <= window.end_ns)
    if sel.sum() < 10:
        return window, "window", float("nan")
    ts = times[sel] / NS
    ps = np.asarray(slam_pos, dtype=np.float64)[sel]
    span = min(MANIPULATION_SPAN_S, ts[-1] - ts[0])
    if span < 0.6:
        return window, "window", float("nan")
    stride = max(1, int(round(_SPEED_SMOOTH_S / max(np.median(np.diff(ts)), 1e-6))))
    dt = ts[stride:] - ts[:-stride]
    speed = np.linalg.norm(ps[stride:] - ps[:-stride], axis=1) / np.maximum(dt, 1e-9)
    s_ts = 0.5 * (ts[stride:] + ts[:-stride])

    best = None
    j0 = 0
    csum = np.concatenate([[0.0], np.cumsum(speed)])
    for i in range(speed.size):
        while s_ts[i] - s_ts[j0] > span:
            j0 += 1
        if s_ts[i] - s_ts[j0] < 0.8 * span:
            continue
        mean_v = (csum[i + 1] - csum[j0]) / (i + 1 - j0)
        if best is None or mean_v < best[2]:
            best = (s_ts[j0], s_ts[i], mean_v)
    if best is None:
        return window, "window", float(np.median(speed))
    trimmed = dataclasses.replace(
        window, start_ns=int(round(best[0] * NS)), end_ns=int(round(best[1] * NS)))
    return trimmed, "manipulation", float(best[2])


def build_hand_record(
    root,
    *,
    scene: str,
    trial: str,
    iphone_module: str,
    window_name: str,
    num_frames: int = 16,
    num_waypoints: int = N_WAYPOINTS,
    prompt_action: str | None = None,
    frame_size: int = DEFAULT_FRAME_SIZE,
    label_span_mode: str = "manipulation",
    depth_source: str = "lidar",
    depth_frames: str = "first",
) -> HandRecord:
    root = pathlib.Path(root)
    scene_root = root / scene
    hand_root = scene_root / "hand"
    aria_dir = hand_root / "aria_human" / f"{trial}_vrs"
    iphone_dir = hand_root / iphone_module / trial

    trial_indices = trial.removeprefix(f"{scene}_").removesuffix("_hand")
    confirmed = hand_root / f"interaction_splitting_info_{trial_indices}_confirmed.json"
    windows = load_confirmed_interaction_windows(confirmed, skip_unindexed=True)
    matches = [w for w in windows if w.name == window_name or w.name.endswith(window_name)]
    if len(matches) != 1:
        raise ValueError(
            f"window {window_name!r} matched {len(matches)} of {[w.name for w in windows]}"
        )
    window = matches[0]


    rgb_layout = _Layout(
        iphone_dir / "camera_rgb.zip"
        if (iphone_dir / "camera_rgb.zip").exists() else iphone_dir / "camera_rgb")
    poses_layout = _Layout(
        iphone_dir / "poses_aligned.zip"
        if (iphone_dir / "poses_aligned.zip").exists() else iphone_dir / "poses_aligned")
    if depth_source not in ("lidar", "da3"):
        raise ValueError(f"depth_source must be 'lidar' or 'da3', got {depth_source!r}")
    depth_layout = None
    if depth_source == "lidar":


        depth_base = iphone_dir / "camera_depth.zip" \
            if (iphone_dir / "camera_depth.zip").exists() else iphone_dir / "camera_depth"
        depth_layout = _Layout(depth_base)


    slam_zip = aria_dir / "slam.zip"
    if slam_zip.exists():
        slam_layout = _Layout(slam_zip, prefer="*closed_loop_trajectory_aligned/data.csv")
    else:
        slam_layout = _Layout(aria_dir / "slam/closed_loop_trajectory_aligned")
    slam_times, slam_pos, slam_quat = _read_aria_slam(slam_layout)


    if label_span_mode == "manipulation":
        window, label_span, body_speed_med = _trim_to_manipulation(window, slam_times, slam_pos)
    else:
        label_span, body_speed_med = "window", float("nan")

    targets = _observation_times(window, num_frames=num_frames)
    refs = _nearest_images(rgb_layout.list_images(), targets)
    frame_times = np.asarray([r.timestamp_ns for r in refs], dtype=np.float64)
    gaps_s = np.abs(frame_times - targets) / NS
    if float(gaps_s.max()) > MAX_FRAME_GAP_S:
        raise ValueError(
            f"nearest iphone frame gap {gaps_s.max():.3f} s exceeds {MAX_FRAME_GAP_S} s")
    if np.unique(frame_times).size != num_frames:
        raise ValueError("iphone frame selection collapsed to duplicate frames")
    rgb = np.stack([rgb_layout.decode_image(r.member) for r in refs])
    rgb = np.stack([resize_long_side(f, int(frame_size)) for f in rgb])
    depth_K = depth_maps = depth_Ks = None
    if depth_source == "lidar":

        depth_map = read_depth_map(depth_layout, float(frame_times[0]),
                                   rgb.shape[1], rgb.shape[2])
    else:


        import torch
        from vidforce.perception.depth import da3_depth_stack
        if depth_frames not in ("first", "all"):
            raise ValueError(f"depth_frames must be 'first' or 'all', got {depth_frames!r}")
        stack, ks = da3_depth_stack(
            rgb if depth_frames == "all" else rgb[:1],
            device="cuda" if torch.cuda.is_available() else "cpu")
        depth_map, depth_K = stack[0], ks[0]
        if depth_frames == "all":
            depth_maps, depth_Ks = stack, ks


    if poses_layout.is_zip:
        pose_data = _read_camera_poses(poses_layout.path)
    else:
        rows = poses_layout.read_csv("data.csv")
        times = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
        pos_ = np.asarray([[float(r["tx_world_cam"]), float(r["ty_world_cam"]), float(r["tz_world_cam"])]
                           for r in rows], dtype=np.float64)
        quat_ = np.asarray([[float(r["qx_world_cam"]), float(r["qy_world_cam"]),
                             float(r["qz_world_cam"]), float(r["qw_world_cam"])] for r in rows],
                           dtype=np.float64)
        from vidforce.data.hoi.hoi_gripper import _dedupe_sorted
        pose_data = _dedupe_sorted(times, pos_, Rotation.from_quat(quat_).as_quat())
    pose_times, pose_pos, pose_quat = pose_data
    T_world_iphone = _pose_matrix_at(pose_times, pose_pos, pose_quat, frame_times[0])
    pose_gap = float(np.min(np.abs(pose_times - frame_times[0]))) / NS
    if pose_gap > MAX_POSE_GAP_S:
        raise ValueError(f"iphone pose gap {pose_gap:.3f} s exceeds {MAX_POSE_GAP_S} s")


    ht_zip = aria_dir / "hand_tracking.zip"
    if ht_zip.exists():
        ht_layout = _Layout(ht_zip, prefer="*hand_tracking_aligned/data.csv")
    else:
        ht_dir = aria_dir / "hand_tracking/hand_tracking_aligned"
        ht_zip_flat = aria_dir / "hand_tracking_aligned.zip"
        ht_layout = _Layout(ht_zip_flat if ht_zip_flat.exists()
                            else ht_dir)
    ht_times, ht_rows = _read_hand_tracking(ht_layout)
    hand = _choose_hand_window(ht_rows, window.start_ns, window.end_ns)


    sel_idx = np.flatnonzero(
        (ht_times >= window.start_ns) & (ht_times <= window.end_ns))
    if not sel_idx.size:
        raise ValueError(f"no hand-tracking samples in window {window_name}")
    sel_t = np.asarray([ht_times[j] for j in sel_idx], dtype=np.float64)
    order = np.argsort(sel_t)
    sel_t = sel_t[order]
    sel_rows = [ht_rows[ht_times[sel_idx[j]]] for j in order]
    dev_pos = np.array(
        [[float(row[f"tx_{hand}_landmark_0_device"]),
          float(row[f"ty_{hand}_landmark_0_device"]),
          float(row[f"tz_{hand}_landmark_0_device"])] for row in sel_rows],
        dtype=np.float64,
    )
    ok = np.isfinite(dev_pos).all(axis=1) & (np.abs(dev_pos).sum(axis=1) > 1e-9)
    if not ok.any():
        raise ValueError(f"no valid {hand}-hand landmark samples in window {window_name}")

    world_pos = np.empty_like(dev_pos, dtype=np.float64)
    slam_times_arr = np.asarray(slam_times, dtype=np.float64)
    slam_R = Rotation.from_quat(slam_quat).as_matrix()
    for j in np.flatnonzero(ok):
        i = int(np.argmin(np.abs(slam_times_arr - sel_t[j])))
        world_pos[j] = slam_R[i] @ dev_pos[j] + slam_pos[i]

    world_pos = _nearest_visible(world_pos, ok)


    wpt_times_s = np.linspace(0.0, (window.end_ns - window.start_ns) / NS, num_waypoints)
    sample_s = np.linspace(window.start_ns, window.end_ns, num_waypoints) / NS

    wpt_world = np.stack(
        [np.interp(sample_s, sel_t[ok] / NS, world_pos[ok, k]) for k in range(3)], axis=1)


    traj_cam = _world_to_camera(wpt_world, T_world_iphone).astype(np.float32)


    frac_front = float(np.mean(traj_cam[:, 2] > 0.0))
    arc = float(np.linalg.norm(np.diff(traj_cam, axis=0), axis=1).sum())
    if not (MIN_ARC_LENGTH_M <= arc <= MAX_ARC_LENGTH_M):
        raise ValueError(
            f"trajectory arc {arc:.3f} m outside [{MIN_ARC_LENGTH_M}, {MAX_ARC_LENGTH_M}] m; "
            f"z>0 {frac_front:.2f}")

    gmax = float(np.nanmax(np.abs(frame_times - frame_times[0]))) / NS
    return HandRecord(
        rgb=rgb,
        frame_times_ns=frame_times,
        depth_map=depth_map,
        traj_m=traj_cam,
        waypoint_times_s=wpt_times_s.astype(np.float64),
        force_valid=False,
        force_vec_n=np.zeros((num_waypoints, 3), dtype=np.float32),
        peak_force_n=0.0,
        contact=np.zeros(num_waypoints, dtype=bool),
        T_world_iphone=T_world_iphone,
        prompt=_prompt(prompt_action or window.action, None),
        depth_K=depth_K,
        depth_maps=depth_maps,
        depth_Ks=depth_Ks,
        provenance={
            "scene": scene,
            "trial": trial,
            "window": window.name,
            "iphone_module": iphone_module,
            "trajectory_source": "hand_tracking_aligned_landmark0",
            "hand_used": hand,
            "force_valid": False,
            "waypoint_rule": "uniform_over_window",


            "contact_start_ns": int(window.start_ns),


            "label_span": label_span,
            "body_speed_median_m_s": body_speed_med,
            "window_start_ns": window.start_ns,
            "window_end_ns": window.end_ns,
            "label_frame": "iphone_camera_at_observation",


            "depth_source": depth_source,
            "depth_frames": depth_frames if depth_source == "da3" else "first",
            "frame_gaps_s": gaps_s.tolist(),
            "max_frame_gap_s": float(gaps_s.max()),
            "pose_gap_s": pose_gap,
            "in_window_samples": int(sel_idx.size),
            "arc_m": float(arc),
            "frac_front": float(frac_front),
            "action": window.action or "",
            "iphone_span_s": gmax,
        },
    )
