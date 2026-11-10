"""First-frame conditioning of a third-person record.

``exo_conditioning`` projects the trimmed span of the trajectory into the image and returns the contact pixel and
depth (the first point of the span inside the image), the object window (a square crop around the contact that covers
the projected span) and the scene window (1.5 times wider), each with the pinhole intrinsics of its crop.
``first_frame_index`` maps a waypoint of the span to the nearest frame; ``scale_K`` rescales intrinsics between
resolutions of the same aspect ratio.
"""
from __future__ import annotations

import numpy as np

from vidforce.data.crops import crop_K, object_crop, scene_crop

DEFAULT_MARGIN = 1.3
DEFAULT_MIN_SIDE_FRAC = 0.4
DEFAULT_SCENE_FACTOR = 1.5


def first_frame_index(frame_times_ns: np.ndarray,
                      waypoint_times_s: np.ndarray, start_i: int) -> int:
    ft = np.asarray(frame_times_ns, np.float64)
    t0 = float(ft[0]) / 1e9
    wt = np.asarray(waypoint_times_s, np.float64)
    return int(np.argmin(np.abs(ft / 1e9 - t0 - float(wt[int(start_i)]))))


def _proj_cam(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    u = p_cam[0] / p_cam[2] * K[0, 0] + K[0, 2]
    v = p_cam[1] / p_cam[2] * K[1, 1] + K[1, 2]
    return float(u), float(v)


def _zoom_K(K: np.ndarray, box: tuple[int, int, int, int]) -> list[list]:
    return np.asarray(crop_K(K, box, box[2] - box[0], box[3] - box[1])
                      ).tolist()


def scale_K(K: np.ndarray, src_w: int, W: int, H: int) -> np.ndarray:
    K = np.asarray(K, np.float64).copy()
    sx, sy = W / float(src_w), H / float(K[1, 2] * 2)
    if abs(sx / sy - 1.0) > 1e-6:
        raise ValueError(f"non-uniform aspect: sx {sx} vs sy {sy}")
    K[[0, 0, 1, 1], [0, 2, 1, 2]] *= sx
    return K


def exo_conditioning(
    traj_m: np.ndarray,
    K: np.ndarray,
    start_i: int,
    end_i: int,
    *,
    W: int,
    H: int,
    margin: float = DEFAULT_MARGIN,
    min_side_frac: float = DEFAULT_MIN_SIDE_FRAC,
    scene_factor: float = DEFAULT_SCENE_FACTOR,
) -> dict:
    span = np.asarray(traj_m, np.float64)[int(start_i):int(end_i) + 1]
    uv = np.stack([_proj_cam(p, K) for p in span])
    inside = (span[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W) \
        & (uv[:, 1] >= 0) & (uv[:, 1] <= H)
    if not inside.any():
        return {"behind_camera": True}
    contact = tuple(uv[int(np.argmax(inside))])
    bbox_n = (float(uv[:, 0].min() / W), float(uv[:, 1].min() / H),
              float(uv[:, 0].max() / W), float(uv[:, 1].max() / H))
    obj = object_crop((contact[0] / W, contact[1] / H), bbox_n, W, H,
                      margin=margin, min_side_frac=min_side_frac)
    scn = scene_crop(obj, W, H, factor=scene_factor)

    def entry(box):
        return {"box": [int(v) for v in box],
                "K": _zoom_K(np.asarray(K, np.float64), box)}

    return {
        "behind_camera": False,
        "contact_uv": [round(contact[0], 2), round(contact[1], 2)],
        "contact_uv_norm": [round(contact[0] / W, 4), round(contact[1] / H, 4)],
        "contact_depth_m": round(float(span[int(np.argmax(inside))][2]), 4),
        "bbox": {
            "center_uv": [round((obj[0] + obj[2]) / 2, 2),
                          round((obj[1] + obj[3]) / 2, 2)],
            "half_extents_px": [round((obj[2] - obj[0]) / 2, 2),
                                round((obj[3] - obj[1]) / 2, 2)],
        },
        "grasp_inframe": True,
        "spans": {"scene": entry(scn), "object": {"box": list(obj),
                                                  "K": entry(obj)["K"]}},
        "zoom_default": "exo_object_crop",
    }


