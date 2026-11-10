"""Quality filter and span trimming (thresholds in ``FilterParams``; lengths in metres, steps per waypoint).

Depth-ray gap (``frame_gap``): depth of a trajectory point minus the depth image at its projection. Up to 0.20 m the
point is visible, up to 0.45 m borderline, beyond that occluded (``occlusion_at``).

Gripper record: the gap is the median over the first four waypoints; rejected when the contact is occluded or outside
the image or when the path runs behind a person (``person_path_gap``); ambiguous when borderline or without depth.

Human record: the span starts at the grip onset (``grip_index``, refined by ``refine_contact_start``) and ends at the
earliest of the kinematic end (``kinematic_end``), the stop of the object (``object_stop_end``, needs tracked scene
points) and the per-scene cap (``capped_end``), refined by ``refine_release_end``. ``hand_visibility_scan`` finds the
first visible waypoint, which starts the supervised span. Rejected when nothing is visible, when the start lies inside
the person mask, or when the visible part covers less than half of the path; ambiguous when borderline, when the end
is uncertain, or when fewer than four waypoints remain.

A span longer than 2.5 m is rejected. ``score_record`` returns the label, the tags and the span of one record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FilterParams:

    cap_default: float = 0.525
    slow_step: float = 0.08
    fast_step: float = 0.10
    settle_step: float = 0.04
    occ_visible: float = 0.20
    occ_border: float = 0.45
    min_depth: float = 0.05
    os_radius: float = 0.10
    os_range_thr: float = 0.10
    os_floor: float = 0.015
    os_hand_dist: float = 0.12
    os_min_movers: int = 2
    os_settle_steps: int = 2
    min_span: int = 4
    person_edge_frac: float = 0.04

    gate_floor: float = 0.012


    person_min_wpts: int = 3

    max_sup_len: float = 2.5

    refine_dip: float = 0.035


    occ_win_frac: float = 0.015


    release_rise: float = 0.12


    release_sustain: float = 0.10


    release_min_pre: float = 0.10


    vis_frac_min: float = 0.5


DEFAULT_PARAMS = FilterParams()


def step_lengths(traj: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(np.asarray(traj, np.float64), axis=0), axis=1)


def grip_index(traj, p: FilterParams = DEFAULT_PARAMS):
    d = step_lengths(traj)
    if len(d) < 4 or d[0] < p.slow_step:
        return 0, "at_grip"
    return int(np.argmin(d[:max(3, int(len(d) * 0.6))])) + 1, "cut"


def kinematic_end(traj, gi: int, p: FilterParams = DEFAULT_PARAMS):
    d = step_lengths(traj)
    T = len(traj)
    if len(d) < 4 or d[-1] < p.slow_step:
        return T - 1, "ends_slow"
    j = len(d) - 1
    while j > 0 and d[j] >= p.fast_step:
        j -= 1
    settle = j < len(d) - 2 and d[j] < p.settle_step
    q = min(T - 1, max(gi + 1, int(T * 0.75)))
    v = traj[q] - traj[gi]
    v = v / max(np.linalg.norm(v), 1e-9)
    segs = [traj[i + 1] - traj[i] for i in range(len(d) - 3, len(d))]
    reversal = any(float((s / max(np.linalg.norm(s), 1e-9)) @ v) < 0
                   for s in segs)
    if settle or reversal:
        cut = j + 1 if settle else len(d) - 3
        return max(cut, gi + 1), ("settle" if settle else "reversal")
    return T - 1, "ambiguous_tail"


def capped_end(traj, gi: int, cap: float):
    d = step_lengths(traj)
    cum = 0.0
    for i in range(gi, len(d)):
        cum += d[i]
        if cum >= cap:
            return i + 1, "capped"
    return len(traj) - 1, "natural"


def traj_to_cache_step(i: float, n_traj: int, n_cache: int) -> int:
    return int(round(i * (n_cache - 1) / max(1, n_traj - 1)))


def cache_step_to_traj(s: float, n_traj: int, n_cache: int) -> int:
    return int(round(s * (n_traj - 1) / max(1, n_cache - 1)))


def mover_motion_profile(traj, cache: dict | None, uv0,
                         p: FilterParams = DEFAULT_PARAMS):
    if cache is None or uv0 is None:
        return None
    uv = cache["uv"].float()
    xyz = cache["xyz"].float()
    vis = cache["vis"].float()
    near = ((uv - torch.tensor(uv0, dtype=torch.float32)).norm(dim=-1)
            < p.os_radius).nonzero().flatten()
    if len(near) < p.os_min_movers:
        return None
    xyz_n, vis_n = xyz[near], vis[near]
    K = xyz_n.shape[1]
    T = len(traj)
    rng = (xyz_n.max(1).values - xyz_n.min(1).values).norm(dim=-1)
    ti = np.linspace(0, T - 1, K)
    tp = np.stack([np.interp(ti, np.arange(T), np.asarray(traj)[:, k])
                   for k in range(3)], axis=1)


    dmed = (xyz_n - torch.tensor(tp, dtype=torch.float32)) \
        .norm(dim=-1).median(1).values
    keep = (rng > p.os_range_thr) & (dmed > p.os_hand_dist)
    if int(keep.sum()) < p.os_min_movers:
        return None
    step = (xyz_n[:, 1:] - xyz_n[:, :-1]).norm(dim=-1)
    ok = (vis_n[:, 1:] > 0.5) & (vis_n[:, :-1] > 0.5) & keep[:, None]
    prof = np.full(step.shape[1], np.nan)
    for s in range(step.shape[1]):
        m = ok[:, s]
        if m.any():
            prof[s] = float(step[m, s].median())
    return prof


def object_stop_end(traj, gi: int, cap_i: int, cache: dict | None, uv0,
                    p: FilterParams = DEFAULT_PARAMS, profile=None):
    prof = profile if profile is not None else \
        mover_motion_profile(traj, cache, uv0, p)
    if prof is None:
        return None
    T, K = len(traj), len(prof) + 1
    s_lo = traj_to_cache_step(gi, T, K)
    s_hi = min(traj_to_cache_step(cap_i, T, K), len(prof) - 1)
    moving = np.where(np.nan_to_num(prof, nan=-1.0) > p.os_floor)[0]
    moving = moving[(moving >= s_lo) & (moving <= s_hi)]
    if len(moving) == 0:
        return None
    last = int(moving[-1])
    if last + 1 >= len(prof):


        return None
    after = prof[last + 1: last + 1 + p.os_settle_steps]
    if np.any(np.nan_to_num(after, nan=-1.0) > p.os_floor):
        return None
    i_end = cache_step_to_traj(last, T, K)
    return max(gi + 1, min(cap_i, i_end))


def object_moving_at_start(traj, gi: int, profile,
                           p: FilterParams = DEFAULT_PARAMS) -> bool:
    if profile is None or gi <= 0:
        return False
    s_gi = traj_to_cache_step(gi, len(traj), len(profile) + 1)
    pre = profile[:max(0, s_gi)]
    pre = pre[~np.isnan(pre)]
    if len(pre) < 2:
        return False
    return float(np.median(pre)) > p.gate_floor


def articulation_end(traj, gi: int, cap: float, cache: dict | None = None,
                     uv0=None, p: FilterParams = DEFAULT_PARAMS,
                     profile=None):
    kin_i, kin_tag = kinematic_end(traj, gi, p)
    cap_i, cap_tag = capped_end(traj, gi, cap)
    if cap_i >= len(traj) - 1:
        cap_tag = "natural"
    obj_i = object_stop_end(traj, gi, cap_i, cache, uv0, p, profile=profile)
    cands = []
    if kin_tag in ("settle", "reversal"):
        cands.append((kin_i, kin_tag))
    if obj_i is not None:
        cands.append((obj_i, "object_stop"))
    cands.append((cap_i, cap_tag))
    end_i, tag = min(cands, key=lambda c: c[0])
    if kin_tag == "ambiguous_tail" and cap_tag == "natural" and obj_i is None:
        tag = "ambiguous_tail"
    elif kin_tag == "ends_slow" and end_i == len(traj) - 1:
        tag = "ends_slow"
    score = 0.5 if tag == "ambiguous_tail" else 1.0
    return end_i, tag, score


def frame_gap(point, depth_map, project_uv, p: FilterParams = DEFAULT_PARAMS,
              win_frac: float = 0.0):
    uv = project_uv(point)
    if uv is None or not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1):
        return None, "oob"
    dH, dW = depth_map.shape[:2]
    y = min(int(uv[1] * dH), dH - 1)
    x = min(int(uv[0] * dW), dW - 1)
    if win_frac > 0:
        r = max(2, int(win_frac * float(np.hypot(dW, dH))))
        win = np.asarray(depth_map[max(0, y - r):y + r + 1,
                                   max(0, x - r):x + r + 1], np.float64)
        valid = win[win > p.min_depth]
        if valid.size == 0:
            return None, "no_depth"
        return float(point[2] - valid.max()), None
    z_px = float(depth_map[y, x])
    if z_px <= p.min_depth:
        return None, "no_depth"
    return float(point[2]) - z_px, None


def occlusion_at(traj, depth_map, project_uv, i0: int = 0, n: int = 4,
                 p: FilterParams = DEFAULT_PARAMS):
    gaps, reasons = [], []
    for i in range(i0, min(i0 + n, len(traj))):
        g, why = frame_gap(traj[i], depth_map, project_uv, p)
        if g is not None:
            gaps.append(g)
        else:
            reasons.append(why)
    if not gaps:
        tag = "out_of_frame" if "oob" in reasons else "no_depth"
        return None, tag, 0.0
    g = float(np.median(gaps))
    if g > p.occ_border:
        return g, "occluded", 0.0
    if g > p.occ_visible:
        return g, "borderline", 0.5
    return g, "visible", 1.0


def hand_visibility_scan(traj, gi: int, ae: int, depth_map, project_uv,
                         p: FilterParams = DEFAULT_PARAMS):
    first_border = None
    reasons = []
    for i in range(gi, ae + 1):
        g, why = frame_gap(traj[i], depth_map, project_uv, p,
                           win_frac=p.occ_win_frac)
        if g is None:
            reasons.append(why)
            continue
        if g <= p.occ_visible:
            return i, "visible", g
        if g <= p.occ_border and first_border is None:
            first_border = (i, g)
    if first_border is not None:
        return first_border[0], "borderline", first_border[1]
    if reasons and all(w == "oob" for w in reasons):
        return -1, "out_of_frame", None
    return -1, "occluded", None


def refine_contact_start(traj, gi: int, start_tag: str, ae: int,
                         p: FilterParams = DEFAULT_PARAMS):
    if gi != 0 or start_tag not in ("at_grip", "object_moving_at_start"):
        return gi, start_tag
    d = step_lengths(traj)
    if len(d) < 5:
        return gi, start_tag
    window = d[1:4]
    dip = 1 + int(np.argmin(window))
    after = d[dip + 1:dip + 3]
    if (d[dip] <= p.refine_dip
            and d[dip - 1] > d[dip]
            and len(after) > 0
            and float(np.max(after)) >= max(2.0 * d[dip], 0.05)
            and ae - (dip + 1) + 1 >= p.min_span):
        return dip + 1, "contact_refined"
    return gi, start_tag


def refine_release_end(traj, gi: int, ae: int, end_tag: str,
                       p: FilterParams = DEFAULT_PARAMS):
    d = step_lengths(traj)
    for m in range(gi, min(ae, len(d))):
        if d[m] > p.refine_dip:
            continue
        if float(d[gi:m].sum()) < p.release_min_pre:
            continue
        rest = m
        while (rest + 1 < min(ae, len(d))
               and d[rest + 1] <= max(p.refine_dip, 2.0 * d[m])):
            rest += 1
        k = rest + 1
        if k > ae - 1 or k >= len(d):
            continue
        if d[k] < max(3.0 * d[m], p.release_rise):
            continue
        seg = d[k:min(k + 2, ae)]
        if len(seg) > 0 and bool(np.all(seg >= p.release_sustain)) \
                and rest + 1 < ae:
            return rest + 1, "release_cut"
    return ae, end_tag


def person_point_status(uv, person_mask,
                        edge_frac: float = DEFAULT_PARAMS.person_edge_frac):
    if uv is None or person_mask is None:
        return "clear"
    mask = np.asarray(person_mask, bool)
    H, W = mask.shape[:2]
    diag = float(np.hypot(W, H))
    x = min(int(uv[0] * (W - 1)), W - 1)
    y = min(int(uv[1] * (H - 1)), H - 1)
    w = max(2, int(0.03 * diag))
    ys, xs = slice(max(0, y - w), y + w + 1), slice(max(0, x - w), x + w + 1)
    on_person = bool(mask[y, x]) or float(mask[ys, xs].mean()) >= 0.25
    if not on_person:
        return "clear"


    from scipy.ndimage import distance_transform_edt
    depth_px = float(distance_transform_edt(mask)[ys, xs].max())
    return "deep" if depth_px > edge_frac * diag else "edge"


def person_path_gap(traj, project_uv, person_mask, depth_map,
                    p: FilterParams = DEFAULT_PARAMS):
    if person_mask is None:
        return 0, None
    mask = np.asarray(person_mask, bool)
    H, W = mask.shape[:2]
    gaps = []
    for pt in traj:
        uv = project_uv(pt)
        if uv is None or not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1):
            continue
        x = min(int(uv[0] * (W - 1)), W - 1)
        y = min(int(uv[1] * (H - 1)), H - 1)
        if not mask[y, x]:
            continue
        g, why = frame_gap(pt, depth_map, project_uv, p)
        if g is not None:
            gaps.append(g)
    if not gaps:
        return 0, None
    return len(gaps), float(np.median(gaps))


def bad_pose_hand(traj, check_i: int, cache: dict | None, project_uv,
                  thresh: float = 0.12):
    uv = project_uv(traj[check_i])
    if uv is None or not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1):
        return True
    if cache is None:
        return False
    d = (cache["uv"].float() - torch.tensor(uv, dtype=torch.float32)) \
        .norm(dim=-1)
    return float(d.min()) > thresh


def score_record(name: str, traj, *, is_hand: bool, depth_map, project_uv,
                 cache: dict | None = None, bad_pose_names=frozenset(),
                 cap: float | None = None, scene_caps: dict | None = None,
                 person_mask=None, hand_trim: dict | None = None,
                 p: FilterParams = DEFAULT_PARAMS) -> dict:
    traj = np.asarray(traj, np.float64)
    traj = traj[0] if traj.ndim == 3 else traj
    scene = name.split("__")[0]
    if cap is None:
        cap = (scene_caps or {}).get(scene, p.cap_default)

    def in_frame_uv(pt):
        uv = project_uv(pt)
        if uv is not None and not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1):
            return None
        return uv

    if is_hand:
        if hand_trim is not None:


            gi, start_tag = int(hand_trim["grip_i"]), str(hand_trim["start"])
            ae, end_tag = int(hand_trim["end_i"]), str(hand_trim["end"])
        else:
            gi, start_tag = grip_index(traj, p)


            uv_start = in_frame_uv(traj[0])
            gate_profile = mover_motion_profile(traj, cache, uv_start, p)
            if object_moving_at_start(traj, gi, gate_profile, p):


                gi, start_tag = 0, "object_moving_at_start"
                uv0, profile = uv_start, gate_profile
            else:
                uv0 = in_frame_uv(traj[gi])
                profile = mover_motion_profile(traj, cache, uv0, p)
            ae, end_tag, _ = articulation_end(traj, gi, cap, cache, uv0, p,
                                              profile=profile)


        gi, start_tag = refine_contact_start(traj, gi, start_tag, ae, p)
        if start_tag == "contact_refined":


            uv_r = in_frame_uv(traj[gi])
            ae, end_tag, _ = articulation_end(
                traj, gi, cap, cache, uv_r, p,
                profile=mover_motion_profile(traj, cache, uv_r, p))

        ae, end_tag = refine_release_end(traj, gi, ae, end_tag, p)
        start_i, occ_tag, gap = hand_visibility_scan(
            traj, gi, ae, depth_map, project_uv, p)
        if start_i < 0:
            l0, label = "n/a", "REJECT"
        else:


            st_start = person_point_status(in_frame_uv(traj[start_i]),
                                           person_mask, p.person_edge_frac)
            if st_start == "deep" or (
                    st_start == "edge"
                    and person_point_status(in_frame_uv(traj[0]), person_mask,
                                            p.person_edge_frac) == "deep"):
                occ_tag = "body_occluded"


            dh = step_lengths(traj)
            full_path = float(dh[gi:ae].sum())
            if start_i > gi and full_path > 0 \
                    and float(dh[start_i:ae].sum()) / full_path \
                    < p.vis_frac_min:
                occ_tag = "partial_span"
            l0 = ("bad_pose" if name in bad_pose_names or
                  bad_pose_hand(traj, start_i, cache, project_uv,
                                p.os_hand_dist) else "ok")
            if l0 == "bad_pose" or occ_tag in ("occluded", "body_occluded",
                                               "partial_span"):
                label = "REJECT"
            elif occ_tag in ("borderline", "no_depth") \
                    or end_tag == "ambiguous_tail":
                label = "AMBIGUOUS"
            elif ae - start_i + 1 < p.min_span:
                label = "AMBIGUOUS"
            else:
                label = "PASS"
    else:
        gi, ae, start_i = 0, len(traj) - 1, 0
        start_tag, end_tag = "gripper_native", "gripper_native"
        gap, occ_tag, occ_sc = occlusion_at(traj, depth_map, project_uv, 0, 4, p)


        if occ_tag not in ("out_of_frame",):
            n_pw, pgap = person_path_gap(traj, in_frame_uv, person_mask,
                                         depth_map, p)
            if n_pw >= p.person_min_wpts and pgap is not None \
                    and pgap > p.occ_visible:
                occ_tag = "person_in_front"
        l0 = "bad_pose" if name in bad_pose_names else "ok"
        if l0 == "bad_pose" or occ_tag in ("occluded", "out_of_frame",
                                           "person_in_front"):
            label = "REJECT"
        elif occ_tag in ("borderline", "no_depth"):
            label = "AMBIGUOUS"
        else:
            label = "PASS"

    d = step_lengths(traj)
    sup_i = start_i if start_i >= 0 else max(gi, 0)
    sup_len = float(d[sup_i:ae].sum()) if ae > sup_i else 0.0
    if sup_len > p.max_sup_len:


        occ_tag, label = "traj_outlier", "REJECT"
    return dict(
        name=name, embodiment="hand" if is_hand else "gripper",
        bad_pose=l0, occlusion=occ_tag,
        gap=round(gap, 3) if gap is not None else None,
        start=start_tag, end=end_tag, grip_i=int(gi), end_i=int(ae),
        start_i=int(sup_i), start_shift=int(sup_i - gi),
        sup_len=round(sup_len, 3),
        score={"PASS": 1.0, "AMBIGUOUS": 0.5, "REJECT": 0.0}[label],
        label=label,
        episode_fields=["start", "end", "grip_i", "end_i", "sup_len"],
    )
