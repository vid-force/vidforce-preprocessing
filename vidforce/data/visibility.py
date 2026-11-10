"""First-frame checks of a record.

``contact_in_frame``: the contact pixel lies inside the image. ``traj_in_frame_rate``: share of the waypoints of the
span that project inside the image (at least 0.5 required). ``object_visible_frac``: of the depth pixels of the object
window that are not more than 0.35 m behind the contact, the share within 0.35 m of the contact depth; the others are
occluders in front of it (at least 0.3 required). ``record_valid`` combines them.
"""
from __future__ import annotations

import numpy as np

MARGIN_M = 0.35
MIN_TRAJ_IN_FRAME = 0.5
MIN_OBJECT_VISIBLE = 0.3


def contact_in_frame(cu, width: int, height: int) -> bool:
    if cu is None:
        return False
    return 0 <= cu[0] < width and 0 <= cu[1] < height


def traj_in_frame_rate(traj_cam, K, width: int, height: int):
    if K is None or traj_cam is None or len(traj_cam) == 0:
        return None
    tr = np.asarray(traj_cam, np.float64)
    K = np.asarray(K, np.float64)
    m = tr[:, 2] > 0
    inside = np.zeros(len(tr), bool)
    u = tr[m, 0] / tr[m, 2] * K[0, 0] + K[0, 2]
    v = tr[m, 1] / tr[m, 2] * K[1, 1] + K[1, 2]
    inside[m] = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return float(inside.mean())


def front_offset_m(depth_map, rgb_shape, cu, z_contact, half_px: int = 2):
    if depth_map is None or cu is None or z_contact is None:
        return None
    dm = np.asarray(depth_map, np.float64)
    dh, dw = dm.shape[:2]
    H, W = rgb_shape[:2]
    x = min(int(round(cu[0] / W * dw)), dw - 1)
    y = min(int(round(cu[1] / H * dh)), dh - 1)
    patch = dm[max(0, y - half_px):y + half_px + 1,
               max(0, x - half_px):x + half_px + 1]
    with np.errstate(invalid="ignore"):
        zd = np.nanmedian(patch) if patch.size else np.nan
    return float(zd - float(z_contact)) if np.isfinite(zd) and zd > 0 else None


def object_visible_frac(depth_map, rgb_shape, crop_entry, z_contact,
                        zoom_default=None, margin_m: float = MARGIN_M):
    if depth_map is None or crop_entry is None or z_contact is None:
        return None
    if isinstance(crop_entry, dict) and "box" not in crop_entry \
            and "img" not in crop_entry:
        keys = list(crop_entry)
        if not keys:
            return None
        crop_entry = crop_entry[zoom_default] \
            if zoom_default in keys else crop_entry[keys[0]]
    b = crop_entry["box"]
    dm = np.asarray(depth_map, np.float64)
    dh, dw = dm.shape[:2]
    H, W = rgb_shape[:2]
    x0 = max(0, int(b[0] / W * dw)); y0 = max(0, int(b[1] / H * dh))
    x1 = min(dw, int(np.ceil(b[2] / W * dw)))
    y1 = min(dh, int(np.ceil(b[3] / H * dh)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = dm[y0:y1, x0:x1]
    zc = float(z_contact)
    with np.errstate(invalid="ignore"):
        vis = np.isfinite(patch) & (np.abs(patch - zc) <= margin_m)
        occ = np.isfinite(patch) & (patch < zc - margin_m)
    n = int(vis.sum() + occ.sum())
    if n < 20:
        return None
    return float(vis.sum() / n)


def record_valid(status_ok: bool, c_in: bool, tifr, ovf,
                 min_tifr: float = MIN_TRAJ_IN_FRAME,
                 min_ovf: float = MIN_OBJECT_VISIBLE) -> bool:
    if not (status_ok and c_in):
        return False
    if tifr is None or tifr < min_tifr:
        return False
    return ovf is None or ovf >= min_ovf


def visibility_fields(status_ok, cu, K, traj, rgb_shape, depth_map,
                      crop_entry, contact_depth, zoom_default=None) -> dict:
    H, W = rgb_shape[:2]
    c_in = contact_in_frame(cu, W, H)
    tifr = traj_in_frame_rate(traj, K, W, H)
    fov = front_offset_m(depth_map, rgb_shape, cu,
                         contact_depth if c_in else None)
    ovf = object_visible_frac(depth_map, rgb_shape, crop_entry, contact_depth,
                              zoom_default=zoom_default)
    return {
        "contact_in_frame": bool(c_in),
        "traj_in_frame_rate": tifr,
        "front_offset_m": fov,
        "object_visible_frac": ovf,
        "record_valid": bool(record_valid(status_ok, c_in, tifr, ovf)),
    }
