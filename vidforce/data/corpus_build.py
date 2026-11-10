"""Record of one gripper recording seen from a third-person camera.

``build_exo_record`` reads one confirmed interaction window of a gripper session: frames of the chosen camera at the
observation times, the tool-point trajectory expressed in that camera at 16 waypoints, the measured force at the
waypoints and as a dense stream with the contact flag of the trigger, the peak force over the contact samples, the
articulation axis, joint type and work, the camera pose, the prompt, and Depth Anything 3 depth with intrinsics for
the first frame or for every frame. ``ExoRecord`` validates the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vidforce.data.hoi.calibrated_streams import (
    interpolate_pose_matrices,
    nearest_boolean_samples,
)
from vidforce.data.hoi.sample import JointType

try:
    import cv2 as _cv2
except ImportError as _e:
    raise ImportError("exo build requires cv2") from _e


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return frame
    out = _cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=_cv2.INTER_AREA)
    return out
from vidforce.data.hoi.hoi_confirmed import (
    HoiConfirmedClipDataset,
    _decode_images,
    _list_images,
    _nearest_images,
    _prompt,
)
from vidforce.data.hoi.hoi_gripper import _pose_matrix_at, _read_camera_poses
from vidforce.data.hoi.interaction_labels import build_calibrated_interaction_labels

MAX_FRAME_GAP_S = 0.1
MAX_POSE_GAP_S = 0.5
MIN_ARC_LENGTH_M = 0.03
MIN_PEAK_N = 0.5


@dataclass(frozen=True)
class ExoRecord:

    rgb: np.ndarray
    frame_times_ns: np.ndarray
    prompt: str
    traj_m: np.ndarray
    force_vec_n: np.ndarray
    waypoint_times_s: np.ndarray
    contact: np.ndarray
    peak_force_n: float
    T_world_iphone: np.ndarray


    T_world_reference_frames: np.ndarray | None = None
    K_reference: np.ndarray | None = None
    rgb_ego: np.ndarray | None = None
    frame_times_ego_ns: np.ndarray | None = None
    rgb_gripper: np.ndarray | None = None
    frame_times_gripper_ns: np.ndarray | None = None
    depth_map: np.ndarray | None = None
    depth_K: np.ndarray | None = None


    depth_maps: np.ndarray | None = None
    depth_Ks: np.ndarray | None = None
    axis: np.ndarray | None = None
    axis_origin: np.ndarray | None = None
    joint_type: str | None = None
    work_j: float | None = None
    dense_time_s: np.ndarray | None = None
    dense_force_mag_n: np.ndarray | None = None
    dense_contact: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K = self.rgb.shape[0]
        T = self.traj_m.shape[0]
        if self.rgb.ndim != 4 or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be (K,H,W,3) uint8, got {self.rgb.shape} {self.rgb.dtype}")
        if self.frame_times_ns.shape != (K,):
            raise ValueError("frame_times_ns must have one entry per frame")
        if self.rgb_ego is not None:
            if self.rgb_ego.ndim != 4 or self.rgb_ego.dtype != np.uint8:
                raise ValueError("rgb_ego must be (K2,H,W,3) uint8")
            if self.frame_times_ego_ns is None or self.frame_times_ego_ns.shape != (self.rgb_ego.shape[0],):
                raise ValueError("frame_times_ego_ns must have one entry per ego frame")
            if np.unique(self.frame_times_ego_ns).size != self.rgb_ego.shape[0]:
                raise ValueError("ego frame selection collapsed to duplicate frames")
        elif self.frame_times_ego_ns is not None:
            raise ValueError("frame_times_ego_ns set without rgb_ego")
        if self.rgb_gripper is not None:
            if self.rgb_gripper.ndim != 4 or self.rgb_gripper.dtype != np.uint8:
                raise ValueError("rgb_gripper must be (K3,H,W,3) uint8")
            if (
                self.frame_times_gripper_ns is None
                or self.frame_times_gripper_ns.shape != (self.rgb_gripper.shape[0],)
            ):
                raise ValueError("frame_times_gripper_ns must have one entry per gripper frame")
            if np.unique(self.frame_times_gripper_ns).size != self.rgb_gripper.shape[0]:
                raise ValueError("gripper frame selection collapsed to duplicate frames")
        elif self.frame_times_gripper_ns is not None:
            raise ValueError("frame_times_gripper_ns set without rgb_gripper")
        if self.traj_m.shape != (T, 3) or self.force_vec_n.shape != (T, 3):
            raise ValueError("traj_m/force_vec_n must be (T,3)")
        if self.contact.shape != (T,) or self.waypoint_times_s.shape != (T,):
            raise ValueError("contact/waypoint_times_s must have one entry per waypoint")
        if self.T_world_iphone.shape != (4, 4):
            raise ValueError("T_world_iphone must be (4,4)")
        if self.T_world_reference_frames is not None and (
            self.T_world_reference_frames.shape != (K, 4, 4)
            or not np.isfinite(self.T_world_reference_frames).all()
        ):
            raise ValueError("T_world_reference_frames must be (K,4,4) finite")
        if self.K_reference is not None and self.K_reference.shape != (3, 3):
            raise ValueError("K_reference must be (3,3)")
        for name in ("traj_m", "force_vec_n", "waypoint_times_s"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains non-finite values")
        if not np.all(np.diff(self.waypoint_times_s) > 0.0):
            raise ValueError("waypoint_times_s must be strictly increasing")
        arc = float(
            np.linalg.norm(np.diff(self.traj_m, axis=0), axis=1).sum()
        ) if T >= 2 else 0.0
        if arc < MIN_ARC_LENGTH_M:
            raise ValueError(f"trajectory arc length {arc:.3f} m below {MIN_ARC_LENGTH_M} m")
        if not np.isfinite(self.peak_force_n) or self.peak_force_n < MIN_PEAK_N:
            raise ValueError(f"peak force {self.peak_force_n} N below {MIN_PEAK_N} N")

    @property
    def arc_length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.traj_m, axis=0), axis=1).sum())

    @property
    def force_mag_n(self) -> np.ndarray:
        return np.linalg.norm(self.force_vec_n, axis=1).astype(np.float32)


def _contact_segment(
    ds: HoiConfirmedClipDataset, window, *, contact_threshold: float
) -> tuple[float, float, int]:
    times = ds.force_times
    mask = (times >= window.start_ns) & (times <= window.end_ns)
    dense = times[mask]
    contact = nearest_boolean_samples(
        ds.trigger_times, ds.trigger_values, dense, threshold=contact_threshold
    )
    idx = np.flatnonzero(contact)
    if idx.size < 2:
        raise ValueError(f"{window.name}: fewer than two contact samples in window")
    return float(dense[idx[0]]), float(dense[idx[-1]]), int(idx.size)


def observation_times_ns(
    contact_start_ns: float,
    contact_end_ns: float,
    *,
    num_frames: int,
    pre_contact_s: float,
    watch_s: float,
) -> np.ndarray:
    if num_frames < 2:
        raise ValueError("num_frames must be >= 2")
    t0 = float(contact_start_ns) - pre_contact_s * 1e9
    t1 = min(float(contact_start_ns) + watch_s * 1e9, float(contact_end_ns))
    if t1 <= t0:
        raise ValueError("observation span is degenerate")
    return np.linspace(t0, t1, num_frames)


def _revolute_plane_axis_angle(traj_m: np.ndarray, axis: np.ndarray) -> float:
    centred = traj_m - traj_m.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    denom = np.linalg.norm(normal) * np.linalg.norm(axis)
    if denom <= 0.0:
        return float("nan")
    cos = abs(float(np.dot(normal, axis)) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def build_exo_record(
    root,
    *,
    scene: str,
    trial: str,
    window_name: str,
    camera_module: str = "iphone_2",
    reference_camera: str = "iphone",
    num_frames: int = 4,
    num_waypoints: int = 16,
    pre_contact_s: float = 0.3,
    watch_s: float = 0.7,
    include_ego: bool = False,
    include_gripper: bool = False,
    frame_size: int | None = None,
    include_depth: bool = False,
    contact_threshold: float = 0.5,
    force_sign: float = -1.0,
    public_prompt: str | None = None,
    depth_frames: str = "first",
) -> ExoRecord:
    if reference_camera not in ("iphone", "ego"):
        raise ValueError(f"reference_camera must be 'iphone' or 'ego', got {reference_camera!r}")
    if reference_camera == "iphone" and not camera_module.startswith("iphone"):
        raise ValueError(f"camera_module must be an iphone view, got {camera_module!r}")

    ds = HoiConfirmedClipDataset(
        root,
        scene=scene,
        trial=trial,
        input_view="aria_human",
        num_waypoints=num_waypoints,
        num_digit_frames=2,
        include_privileged=include_gripper,
        contact_threshold=contact_threshold,
        force_sign=force_sign,
    )
    matches = [w for w in ds.windows if w.name == window_name or w.name.endswith(window_name)]
    if len(matches) != 1:
        raise ValueError(
            f"window {window_name!r} matched {len(matches)} of {[w.name for w in ds.windows]}"
        )
    window = matches[0]

    contact_start, contact_end, n_contact = _contact_segment(
        ds, window, contact_threshold=contact_threshold
    )
    targets = observation_times_ns(
        contact_start,
        contact_end,
        num_frames=num_frames,
        pre_contact_s=pre_contact_s,
        watch_s=watch_s,
    )

    if reference_camera == "ego":


        refs = _nearest_images(ds.human_frames, targets)
        frames_zip = ds.human_rgb_zip
    else:
        iphone_dir = ds.root / scene / "gripper" / camera_module / trial
        frames_zip = iphone_dir / "camera_rgb.zip"
        refs = _nearest_images(_list_images(frames_zip, "camera_rgb"), targets)
    frame_times = np.asarray([ref.timestamp_ns for ref in refs], dtype=np.float64)
    gaps_s = np.abs(frame_times - targets) / 1e9
    if float(gaps_s.max()) > MAX_FRAME_GAP_S:
        raise ValueError(
            f"nearest {reference_camera} frame gap {gaps_s.max():.3f} s exceeds {MAX_FRAME_GAP_S} s"
        )
    if np.unique(frame_times).size != num_frames:
        raise ValueError("frame selection collapsed to duplicate frames")
    rgb = _decode_images(frames_zip, refs)
    native_hw = rgb.shape[1:3]

    rgb_ego = None
    ego_times = None
    ego_gaps_s = None
    if include_ego:
        ego_refs = _nearest_images(ds.human_frames, targets)
        ego_times = np.asarray([ref.timestamp_ns for ref in ego_refs], dtype=np.float64)
        ego_gaps_s = (np.abs(ego_times - targets) / 1e9).tolist()
        if max(ego_gaps_s) > MAX_FRAME_GAP_S:
            raise ValueError(
                f"nearest ego frame gap {max(ego_gaps_s):.3f} s exceeds {MAX_FRAME_GAP_S} s"
            )
        if np.unique(ego_times).size != num_frames:
            raise ValueError("ego frame selection collapsed to duplicate frames")
        rgb_ego = _decode_images(ds.human_rgb_zip, ego_refs)

    rgb_gripper = None
    gripper_times = None
    gripper_gaps_s = None
    if include_gripper:
        gripper_refs = _nearest_images(ds.gripper_frames, targets)
        gripper_times = np.asarray(
            [ref.timestamp_ns for ref in gripper_refs], dtype=np.float64
        )
        gripper_gaps_s = (np.abs(gripper_times - targets) / 1e9).tolist()
        if max(gripper_gaps_s) > MAX_FRAME_GAP_S:
            raise ValueError(
                f"nearest gripper frame gap {max(gripper_gaps_s):.3f} s exceeds {MAX_FRAME_GAP_S} s"
            )
        if np.unique(gripper_times).size != num_frames:
            raise ValueError("gripper frame selection collapsed to duplicate frames")
        rgb_gripper = _decode_images(ds.gripper_rgb_zip, gripper_refs)

    T_world_reference_frames = None
    if reference_camera == "ego":


        T_world_reference = ds._reference_camera(frame_times[0])
        pose_gap_s = float(np.min(np.abs(ds.human_pose.times_ns - frame_times[0]))) / 1e9
        if pose_gap_s > MAX_POSE_GAP_S:
            raise ValueError(f"ego pose gap {pose_gap_s:.3f} s exceeds {MAX_POSE_GAP_S} s")


        T_world_reference_frames = (
            interpolate_pose_matrices(
                ds.human_pose.times_ns,
                ds.human_pose.positions_world,
                ds.human_pose.quats_xyzw_world,
                frame_times,
            )
            @ ds.T_human_device_camera
        )
    else:
        pose_times, pose_pos, pose_quats = _read_camera_poses(iphone_dir / "poses_aligned.zip")
        T_world_reference = _pose_matrix_at(pose_times, pose_pos, pose_quats, frame_times[0])
        pose_gap_s = float(np.min(np.abs(pose_times - frame_times[0]))) / 1e9
        if pose_gap_s > MAX_POSE_GAP_S:
            raise ValueError(f"iphone pose gap {pose_gap_s:.3f} s exceeds {MAX_POSE_GAP_S} s")
    T_world_iphone = T_world_reference

    labels = build_calibrated_interaction_labels(
        window,
        device_times_ns=ds.gripper_pose.times_ns,
        device_positions_world=ds.gripper_pose.positions_world,
        device_quats_xyzw_world=ds.gripper_pose.quats_xyzw_world,
        force_times_ns=ds.force_times,
        force_sensor_n=ds.force_sensor,
        torque_sensor_nm=ds.torque_sensor,
        trigger_times_ns=ds.trigger_times,
        trigger_values=ds.trigger_values,
        calibration=ds.calibration,
        T_world_reference_camera=T_world_iphone,
        num_waypoints=num_waypoints,
        contact_threshold=contact_threshold,
        force_sign=force_sign,
    )

    annotation = ds.annotations[window.object_index]
    fit = ds.object_axis_fits[window.object_index]
    joint_type = JointType(int(annotation.joint_type))
    axis_iphone = T_world_iphone[:3, :3].T @ fit.axis
    plane_angle_deg: float | None = None
    if joint_type is JointType.REVOLUTE:
        plane_angle_deg = _revolute_plane_axis_angle(
            labels.waypoints.position_m.astype(np.float64), axis_iphone
        )

    if frame_size is not None:
        rgb = np.stack([resize_long_side(f, int(frame_size)) for f in rgb])
        if rgb_ego is not None:
            rgb_ego = np.stack([resize_long_side(f, int(frame_size)) for f in rgb_ego])
        if rgb_gripper is not None:
            rgb_gripper = np.stack([resize_long_side(f, int(frame_size)) for f in rgb_gripper])


    K_reference = None
    if reference_camera == "ego":
        K_reference = np.asarray(ds.intrinsics, dtype=np.float64).copy()
        if frame_size is not None:
            K_reference[:2] *= float(max(rgb.shape[1:3])) / float(max(native_hw))

    depth_map = None
    depth_K = None
    depth_maps = None
    depth_Ks = None
    if include_depth:
        from vidforce.perception.depth import da3_depth_stack

        if depth_frames not in ("first", "all"):
            raise ValueError(f"depth_frames must be 'first' or 'all', got {depth_frames!r}")


        stack, ks = da3_depth_stack(
            rgb if depth_frames == "all" else rgb[:1],
            device="cuda" if _cuda_ok() else "cpu")
        depth_map, depth_K = stack[0], ks[0]
        if depth_frames == "all":
            depth_maps, depth_Ks = stack, ks

    T_ref_world = np.linalg.inv(T_world_iphone)
    axis_ref = (T_ref_world[:3, :3] @ fit.axis).astype(np.float32)
    axis_ref = axis_ref / (np.linalg.norm(axis_ref) + 1e-8)
    origin_ref = (T_ref_world @ np.r_[fit.origin, 1.0])[:3].astype(np.float32)
    prompt = public_prompt or _prompt(window.action, annotation)
    return ExoRecord(
        rgb=rgb,
        frame_times_ns=frame_times,
        prompt=prompt,
        traj_m=labels.waypoints.position_m,
        force_vec_n=labels.force.vector_n,
        waypoint_times_s=labels.waypoints.time_s.astype(np.float64),
        contact=labels.force.contact.astype(bool),
        peak_force_n=float(labels.force.max_force_n),
        T_world_iphone=T_world_iphone,
        T_world_reference_frames=T_world_reference_frames,
        K_reference=K_reference,
        rgb_ego=rgb_ego,
        frame_times_ego_ns=ego_times,
        rgb_gripper=rgb_gripper,
        frame_times_gripper_ns=gripper_times,
        depth_map=depth_map,
        depth_K=depth_K,
        depth_maps=depth_maps,
        depth_Ks=depth_Ks,
        axis=axis_ref,
        axis_origin=origin_ref,
        joint_type=joint_type.name.lower(),
        work_j=float(labels.force.work_j),
        dense_time_s=labels.dense_mechanics.time_s.astype(np.float64),
        dense_force_mag_n=np.linalg.norm(
            labels.dense_mechanics.force_reference_n.astype(np.float64), axis=1
        ),
        dense_contact=labels.dense_mechanics.contact.astype(bool),
        provenance={
            "scene": scene,
            "trial": trial,
            "window": window.name,
            "camera_module": "aria_human" if reference_camera == "ego" else camera_module,
            "joint_type": joint_type.name.lower(),
            "ego_view": "aria_human" if include_ego else None,
            "ego_frame_gaps_s": ego_gaps_s,
            "gripper_view": "aria_gripper" if include_gripper else None,
            "gripper_frame_gaps_s": gripper_gaps_s,
            "action": window.action,
            "contact_start_ns": contact_start,
            "contact_end_ns": contact_end,
            "contact_samples": n_contact,
            "frame_gaps_s": gaps_s.tolist(),
            "pose_gap_s": pose_gap_s,
            "plane_axis_angle_deg": plane_angle_deg,
            "axis_fit_residual_rms_m": fit.residual_rms_m,
            "label_frame": (
                "ego_camera_at_observation" if reference_camera == "ego"
                else "iphone_camera_at_observation"
            ),
            "depth_source": "da3" if include_depth else None,
            "depth_frames": depth_frames if include_depth else None,
            "trajectory_source": "calibrated_gripper_tool",
            "wrench_source": "compensated_filtered_bota",
            "peak_rule": "raw_unbinned_contact_gated_max",
        },
    )
