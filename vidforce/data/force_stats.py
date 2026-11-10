"""Force targets of a gripper recording.

The contact phase of the dense force stream is smoothed (Savitzky-Golay, about 0.2 s) and sampled on a uniform
32-point grid over the span (``force_profile``); ``scale_to_peak`` rescales the profile so that its maximum equals the
measured peak. ``detect_peaks`` counts the peaks of the profile, ``smooth_waypoint_force`` smooths the force vectors
at the waypoints, and ``force_class`` bins the peak by fixed edges.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import find_peaks, savgol_filter
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

PROFILE_POINTS = 32
SG_WINDOW_S = 0.2
PEAK_PROMINENCE = 0.25
PEAK_MIN_SEPARATION_S = 0.15
WP_SG_WINDOW = 5


FORCE_BIN_EDGES_N = (8.37, 12.61, 17.42, 27.33)


def force_class(peak_force_n: float) -> int:
    for k, e in enumerate(FORCE_BIN_EDGES_N):
        if peak_force_n < e:
            return k
    return len(FORCE_BIN_EDGES_N)


def contact_phase_span(dense_mag, dense_t, dense_contact):
    if dense_t is None or dense_contact is None:
        return None
    ci = np.flatnonzero(np.asarray(dense_contact, bool))
    if ci.size < 5:
        return None
    dt = np.asarray(dense_t, np.float64)
    t0 = float(dt[ci[0]])
    return (np.asarray(dense_mag, np.float64)[ci[0]:ci[-1] + 1],
            dt[ci[0]:ci[-1] + 1] - t0)


def smooth_dense(mag: np.ndarray) -> np.ndarray:
    if not _HAVE_SCIPY or len(mag) < 5:
        return np.asarray(mag, np.float64)
    n = len(mag)
    win = min(21, n if n % 2 else n - 1)
    return savgol_filter(mag, win, 2)


def force_profile(dense_mag, dense_t, t0: float, tE: float,
                  wp_t=None, wp_mag=None,
                  n: int = PROFILE_POINTS) -> np.ndarray:
    grid = np.linspace(t0, tE, n)
    if dense_t is not None and len(np.atleast_1d(dense_t)) >= 5:
        dt = np.asarray(dense_t, np.float64)
        m = (dt >= t0) & (dt <= tE)
        if m.sum() >= 5:
            sm = smooth_dense(np.asarray(dense_mag, np.float64)[m])
            return np.interp(grid, dt[m], sm)
    if wp_t is None or len(np.atleast_1d(wp_t)) < 2:
        return np.zeros(n)
    wt, wm = np.asarray(wp_t, np.float64), np.asarray(wp_mag, np.float64)
    if _HAVE_SCIPY and len(wt) >= 3:
        return np.clip(PchipInterpolator(wt, wm)(grid), 0.0, None)
    return np.interp(grid, wt, wm)


def detect_peaks(dense_mag, dense_t, t0: float, tE: float,
                 wp_t=None, wp_mag=None):
    src = None
    if dense_t is not None and len(np.atleast_1d(dense_t)) >= 5:
        dt = np.asarray(dense_t, np.float64)
        m = (dt >= t0) & (dt <= tE)
        if m.sum() >= 5:
            ts = dt[m]
            sm = smooth_dense(np.asarray(dense_mag, np.float64)[m])
            fs = 1.0 / max(np.median(np.diff(ts)), 1e-6)
            src = (ts - t0, sm, max(1, int(PEAK_MIN_SEPARATION_S * fs)))
    if src is None:
        if wp_t is None or len(np.atleast_1d(wp_t)) < 3:
            wm = np.asarray(wp_mag, np.float64) if wp_mag is not None \
                else np.zeros(1)
            k = int(np.argmax(wm))
            return (np.array([float(k)]), wm[k:k + 1], 1)


        wt = np.asarray(wp_t, np.float64) - t0
        wm = np.asarray(wp_mag, np.float64)
        src = (wt, wm, 2)

    ts, sm, dist = src
    idx, _ = find_peaks(sm, prominence=PEAK_PROMINENCE * max(sm.max(), 1e-9),
                        distance=dist)
    g = int(np.argmax(sm))
    if not len(idx) or np.abs(idx - g).min() >= dist:
        idx = np.sort(np.append(idx, g))
    keep = [i for j, i in enumerate(idx)
            if j == 0 or ts[i] - ts[idx[j - 1]] >= PEAK_MIN_SEPARATION_S]
    return ts[keep], sm[keep], len(keep)


def smooth_waypoint_force(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, np.float64)
    if not _HAVE_SCIPY or F.shape[0] < WP_SG_WINDOW:
        return F
    out = np.empty_like(F)
    for k in range(3):
        out[:, k] = savgol_filter(F[:, k], WP_SG_WINDOW, 2)
    return out


def scale_to_peak(profile: np.ndarray, peak_force_n: float) -> np.ndarray:
    m = float(np.max(profile))
    if m <= 1e-6:
        return np.asarray(profile, np.float64)
    return np.asarray(profile, np.float64) * (float(peak_force_n) / m)


def peak_time_s(profile: np.ndarray, t0: float, tE: float) -> float:
    grid = np.linspace(t0, tE, len(profile))
    return float(grid[int(np.argmax(profile))] - t0)
