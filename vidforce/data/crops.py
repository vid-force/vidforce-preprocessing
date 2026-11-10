"""Crop geometry: the object window around the contact, the wider scene window, the intrinsics of a crop, and
mirror helpers.
"""
from __future__ import annotations

import numpy as np


def articulation_bbox_uv(traj, start_i: int, end_i: int, project_uv):
    uvs = []
    for i in range(int(start_i), int(end_i) + 1):
        uv = project_uv(traj[i])
        if uv is not None and 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1:
            uvs.append((float(uv[0]), float(uv[1]), i))
    if not uvs:
        return None, None
    contact = next((u for u in uvs if u[2] == int(start_i)), uvs[0])[:2]
    us, vs = [u[0] for u in uvs], [u[1] for u in uvs]
    return contact, (min(us), min(vs), max(us), max(vs))


def _clamp_box(cx, cy, half_w, half_h, W, H):
    w, h = min(2 * half_w, W), min(2 * half_h, H)
    x0 = int(round(min(max(cx - w / 2, 0), W - w)))
    y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return x0, y0, x0 + int(round(w)), y0 + int(round(h))


def object_crop(contact, bbox, W: int, H: int, *,
                margin: float = 1.3, min_side_frac: float = 0.4):
    cx, cy = contact[0] * W, contact[1] * H
    x0, y0, x1, y1 = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
    reach = max(cx - x0, x1 - cx, cy - y0, y1 - cy, 1.0)
    half = max(margin * reach, min_side_frac * max(W, H) / 2)
    return _clamp_box(cx, cy, half, half, W, H)


def scene_crop(object_box, W: int, H: int, *, factor: float = 1.5):
    x0, y0, x1, y1 = object_box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = factor * max(x1 - x0, y1 - y0) / 2
    return _clamp_box(cx, cy, half, half, W, H)


def bbox_area_frac(bbox, box, W: int, H: int) -> float:
    bx0, by0, bx1, by1 = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
    ix0, iy0 = max(bx0, box[0]), max(by0, box[1])
    ix1, iy1 = min(bx1, box[2]), min(by1, box[3])
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    return inter / area if area > 0 else 0.0


def hflip_K(K, w: int):
    K = np.asarray(K, np.float64).copy()
    K[0, 2] = w - K[0, 2]
    return K


def hflip_points(pts):
    pts = np.asarray(pts, np.float64).copy()
    pts[..., 0] *= -1.0
    return pts


def crop_K(K, box, out_w: int, out_h: int):
    K = np.asarray(K, np.float64).copy()
    sx = out_w / float(box[2] - box[0])
    sy = out_h / float(box[3] - box[1])
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - box[0]) * sx
    K[1, 2] = (K[1, 2] - box[1]) * sy
    return K
