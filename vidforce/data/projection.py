"""Pinhole projection of a point in the camera frame to image coordinates normalised by width and height."""
from __future__ import annotations


def project_point_to_uv(p0_cam, K, W, H):
    import numpy as _np
    p = _np.asarray(p0_cam, _np.float64)
    try:
        Km = _np.asarray(K, _np.float64).reshape(3, 3)
    except ValueError:
        return None
    z = p[2]
    if z <= 1e-6:
        return None
    uv = Km @ p
    return _np.array([float(uv[0] / uv[2] / W), float(uv[1] / uv[2] / H)])
