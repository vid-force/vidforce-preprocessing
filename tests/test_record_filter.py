"""Tests of the quality filter and the span trimming.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vidforce.data.filtering import (
    DEFAULT_PARAMS, articulation_end, bad_pose_hand, cache_step_to_traj,
    capped_end, grip_index, hand_visibility_scan, kinematic_end,
    object_stop_end, occlusion_at, score_record, traj_to_cache_step,
)


def traj_from_steps(steps, start=(0.0, 0.0, 5.0), axis=0):
    pts = [np.array(start, np.float64)]
    for s in steps:
        nxt = pts[-1].copy()
        nxt[axis] += s
        pts.append(nxt)
    return np.stack(pts)


def test_grip_index_cut_and_at_grip():
    t = traj_from_steps([0.2, 0.2, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05,
                         0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02])
    gi, tag = grip_index(t)
    assert (gi, tag) == (3, "cut")
    slow = traj_from_steps([0.02] * 15)
    assert grip_index(slow) == (0, "at_grip")


def test_kinematic_end_tags():
    ends_slow = traj_from_steps([0.2, 0.2, 0.05] + [0.05] * 11 + [0.02])
    assert kinematic_end(ends_slow, 2)[1] == "ends_slow"

    settle = traj_from_steps([0.2, 0.05] + [0.05] * 9 + [0.02, 0.15, 0.15, 0.15])
    i, tag = kinematic_end(settle, 1)
    assert tag == "settle" and i == 12

    rev_steps = [0.2, 0.05] + [0.05] * 10 + [-0.15, -0.15, -0.15]
    reversal = traj_from_steps(rev_steps)
    i, tag = kinematic_end(reversal, 1)
    assert tag == "reversal" and i == len(rev_steps) - 3

    ambig = traj_from_steps([0.2, 0.05] + [0.06] * 10 + [0.15, 0.15, 0.15])
    assert kinematic_end(ambig, 1)[1] == "ambiguous_tail"


def test_capped_end_crossing_and_natural():
    t = traj_from_steps([0.1] * 15)
    i, tag = capped_end(t, 0, cap=0.35)
    assert (i, tag) == (4, "capped")
    assert capped_end(t, 0, cap=99.0) == (15, "natural")


def test_index_mapping_roundtrip():
    assert traj_to_cache_step(0, 16, 32) == 0
    assert traj_to_cache_step(15, 16, 32) == 31
    assert cache_step_to_traj(31, 16, 32) == 15

    assert traj_to_cache_step(8, 16, 32) == 17


def _mover_cache(traj, n_movers=3, move_until=20, K=32, step=0.05):
    uv = torch.full((n_movers, 2), 0.5) + torch.linspace(0, 0.04, n_movers)[:, None]
    xyz = torch.zeros(n_movers, K, 3)
    for s in range(1, K):
        xyz[:, s, 1] = xyz[:, s - 1, 1] + (step if s <= move_until else 0.001)
    xyz[:, :, 0] -= 5.0
    vis = torch.ones(n_movers, K)
    return {"uv": uv, "xyz": xyz, "vis": vis}


def test_object_stop_cuts_where_object_settles():
    t = traj_from_steps([0.06] * 15)
    cache = _mover_cache(t, move_until=20)
    i = object_stop_end(t, gi=1, cap_i=15, cache=cache, uv0=(0.5, 0.5))
    assert i == cache_step_to_traj(19, 16, 32)


def test_object_stop_declines_without_movers_or_settle():
    t = traj_from_steps([0.06] * 15)
    assert object_stop_end(t, 1, 15, None, (0.5, 0.5)) is None
    one = _mover_cache(t, n_movers=1)
    assert object_stop_end(t, 1, 15, one, (0.5, 0.5)) is None
    still_moving = _mover_cache(t, move_until=31)
    assert object_stop_end(t, 1, 15, still_moving, (0.5, 0.5)) is None

    hand_cache = _mover_cache(t, move_until=20)
    T, K = len(t), hand_cache["xyz"].shape[1]
    ee = np.stack([np.interp(np.linspace(0, T - 1, K), np.arange(T), t[:, k])
                   for k in range(3)], axis=1)
    hand_cache["xyz"] = torch.tensor(ee, dtype=torch.float32).expand(3, K, 3).clone()
    assert object_stop_end(t, 1, 15, hand_cache, (0.5, 0.5)) is None


def test_articulation_end_policy_precedence():

    settle = traj_from_steps([0.2, 0.05] + [0.05] * 9 + [0.02, 0.15, 0.15, 0.15])
    i, tag, score = articulation_end(settle, 1, cap=99.0)
    assert tag == "settle" and score == 1.0

    ambig = traj_from_steps([0.2, 0.05] + [0.06] * 10 + [0.15, 0.15, 0.15])
    i, tag, score = articulation_end(ambig, 1, cap=0.30)
    assert tag == "capped" and score == 1.0

    i, tag, score = articulation_end(ambig, 1, cap=99.0)
    assert tag == "ambiguous_tail" and score == 0.5 and i == len(ambig) - 1


def _proj_const(uv):
    return lambda p: uv


def test_occlusion_bands_and_no_depth():
    t = traj_from_steps([0.05] * 15, start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)
    assert occlusion_at(t, dm, _proj_const((0.5, 0.5)))[1] == "visible"
    dm_b = np.full((10, 10), 4.7)
    assert occlusion_at(t, dm_b, _proj_const((0.5, 0.5)))[1] == "borderline"
    dm_o = np.full((10, 10), 4.0)
    assert occlusion_at(t, dm_o, _proj_const((0.5, 0.5)))[1] == "occluded"
    dm_z = np.zeros((10, 10))
    assert occlusion_at(t, dm_z, _proj_const((0.5, 0.5)))[1] == "no_depth"
    assert occlusion_at(t, dm, _proj_const(None))[1] == "out_of_frame"


def test_hand_visibility_scan_late_entry():
    t = traj_from_steps([0.05] * 15, start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)

    def proj(p):
        return None if p[0] < 0.15 else (0.5, 0.5)

    start_i, tag, gap = hand_visibility_scan(t, 0, 15, dm, proj)
    assert (start_i, tag) == (3, "visible")

    start_i, tag, _ = hand_visibility_scan(t, 0, 15, dm, _proj_const(None))
    assert (start_i, tag) == (-1, "out_of_frame")


def test_score_record_gripper_ladder():
    t = traj_from_steps([0.05] * 15, start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)
    row = score_record("scene__1-2__window_0__gripper__iphone_1", t,
                       is_hand=False, depth_map=dm,
                       project_uv=_proj_const((0.5, 0.5)))
    assert row["label"] == "PASS" and row["end"] == "gripper_native"
    row = score_record("scene__1-2__window_0__gripper__iphone_1", t,
                       is_hand=False, depth_map=np.full((10, 10), 4.0),
                       project_uv=_proj_const((0.5, 0.5)))
    assert row["label"] == "REJECT" and row["occlusion"] == "occluded"
    row = score_record("scene__1-2__window_0__gripper__iphone_1", t,
                       is_hand=False, depth_map=dm,
                       project_uv=_proj_const((0.5, 0.5)),
                       bad_pose_names={"scene__1-2__window_0__gripper__iphone_1"})
    assert row["label"] == "REJECT" and row["bad_pose"] == "bad_pose"


def test_score_record_hand_pass_and_span_guard():
    t = traj_from_steps([0.2, 0.2, 0.01] + [0.05] * 11 + [0.02],
                        start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)
    row = score_record("scene__1-2__window_0__hand__iphone_1", t,
                       is_hand=True, depth_map=dm,
                       project_uv=_proj_const((0.5, 0.5)), cap=99.0)
    assert row["label"] == "PASS"
    assert row["grip_i"] == 3 and row["end"] == "ends_slow"
    assert set(row["episode_fields"]) == {"start", "end", "grip_i", "end_i",
                                          "sup_len"}

    def proj_late(p):
        return (0.5, 0.5) if p[0] > 0.93 else None


    row = score_record("scene__1-2__window_0__hand__iphone_1", t,
                       is_hand=True, depth_map=dm, project_uv=proj_late,
                       cap=99.0)
    assert row["label"] == "REJECT" and row["occlusion"] == "partial_span"
    assert row["start_shift"] > 0


def test_bad_pose_hand_semantics():
    t = traj_from_steps([0.05] * 15)
    cache = {"uv": torch.tensor([[0.5, 0.5]])}
    assert bad_pose_hand(t, 0, cache, _proj_const((0.51, 0.5))) is False
    assert bad_pose_hand(t, 0, cache, _proj_const((0.9, 0.9))) is True
    assert bad_pose_hand(t, 0, None, _proj_const((0.5, 0.5))) is False
    assert bad_pose_hand(t, 0, cache, _proj_const(None)) is True


def test_grip_gate_flips_mid_articulation_windows():
    from vidforce.data.filtering import (
        mover_motion_profile, object_moving_at_start, score_record)

    steps = [0.09] * 7 + [0.02] + [0.15] * 7
    t = traj_from_steps(steps, start=(0, 0, 5.0))
    gi, tag = grip_index(t)
    assert gi > 0
    cache = _mover_cache(t, move_until=14)
    prof = mover_motion_profile(t, cache, (0.5, 0.5))
    assert prof is not None
    assert object_moving_at_start(t, gi, prof) is True
    dm = np.full((10, 10), 4.9)
    row = score_record("s__1-2__window_0__hand__x", t, is_hand=True,
                       depth_map=dm, project_uv=_proj_const((0.5, 0.5)),
                       cache=cache, cap=99.0)
    assert row["start"] == "object_moving_at_start" and row["grip_i"] == 0
    assert row["end_i"] <= 8

    late = _mover_cache(t, move_until=14)
    late["xyz"][:, :8, :] = late["xyz"][:, :1, :]
    prof_late = mover_motion_profile(t, late, (0.5, 0.5))
    assert object_moving_at_start(t, gi, prof_late) is False


def _person_scene():
    mask = np.zeros((250, 250), bool)
    mask[100:112, 20:120] = True
    mask[40:220, 150:230] = True
    return mask


def test_person_point_status_bands():
    from vidforce.data.filtering import person_point_status
    mask = _person_scene()
    assert person_point_status((0.02, 0.02), mask) == "clear"
    assert person_point_status((0.28, 0.424), mask) == "edge"
    assert person_point_status((0.76, 0.52), mask) == "deep"
    assert person_point_status((0.5, 0.5), None) == "clear"
    assert person_point_status(None, mask) == "clear"


def test_person_mask_verdicts():
    mask = _person_scene()
    dm = np.full((10, 10), 4.9)
    t = traj_from_steps([0.05] * 15, start=(0, 0, 5.0))


    dm_behind = np.full((10, 10), 4.7)
    row = score_record("s__1-2__window_0__gripper__iphone_1", t, is_hand=False,
                       depth_map=dm_behind, project_uv=_proj_const((0.76, 0.52)),
                       person_mask=mask)
    assert row["label"] == "REJECT" and row["occlusion"] == "person_in_front"


    row = score_record("s__1-2__window_0__gripper__iphone_1", t, is_hand=False,
                       depth_map=dm, project_uv=_proj_const((0.76, 0.52)),
                       person_mask=mask)
    assert row["label"] == "PASS"

    def proj_arc(pt):
        return (0.28, max(0.02, 0.424 - pt[0] * 0.4))
    row = score_record("s__1-2__window_0__gripper__iphone_1", t, is_hand=False,
                       depth_map=dm_behind, project_uv=proj_arc,
                       person_mask=mask)
    assert row["occlusion"] != "person_in_front"

    th = traj_from_steps([0.2, 0.2, 0.01] + [0.05] * 11 + [0.02],
                         start=(0, 0, 5.0))
    row = score_record("s__1-2__window_0__hand__x", th, is_hand=True,
                       depth_map=dm, project_uv=_proj_const((0.76, 0.52)),
                       person_mask=mask, cap=99.0)
    assert row["label"] == "REJECT" and row["occlusion"] == "body_occluded"

    row = score_record("s__1-2__window_0__hand__x", th, is_hand=True,
                       depth_map=dm, project_uv=_proj_const((0.28, 0.424)),
                       person_mask=mask, cap=99.0)
    assert row["label"] == "PASS"


def test_open_drawer_window_min_rescue():
    t = traj_from_steps([0.05] * 15, start=(0, 0, 5.0))
    dm = np.full((20, 20), 4.0)
    dm[11, 13] = 4.95
    proj = _proj_const((0.55, 0.55))
    start_i, tag, gap = hand_visibility_scan(t, 0, 15, dm, proj)
    assert (start_i, tag) == (0, "visible") and gap == pytest.approx(0.05)
    assert occlusion_at(t, dm, proj)[1] == "occluded"


def test_body_occlusion_probed_at_start_not_frame0():
    mask = _person_scene()
    dm = np.full((10, 10), 4.9)
    th = traj_from_steps([0.2, 0.2, 0.01] + [0.05] * 11 + [0.02],
                         start=(0, 0, 5.0))

    def proj_frame0_deep(pt, after=(0.02, 0.02)):
        return (0.76, 0.52) if pt[0] < 0.1 else after

    row = score_record("s__1-2__window_0__hand__x", th, is_hand=True,
                       depth_map=dm, project_uv=proj_frame0_deep,
                       person_mask=mask, cap=99.0)
    assert row["label"] == "PASS" and row["occlusion"] == "visible"

    def proj_edge_start(pt):
        return proj_frame0_deep(pt, after=(0.28, 0.424))

    row = score_record("s__1-2__window_0__hand__x", th, is_hand=True,
                       depth_map=dm, project_uv=proj_edge_start,
                       person_mask=mask, cap=99.0)
    assert row["label"] == "REJECT" and row["occlusion"] == "body_occluded"


def test_refine_contact_start_measured_profiles():
    from vidforce.data.filtering import refine_contact_start

    def traj_of(steps):
        return traj_from_steps(steps, start=(0, 0, 5.0))


    k15w12 = [0.121, 0.069, 0.028, 0.075, 0.110, 0.100, 0.077, 0.057,
              0.054, 0.105, 0.199, 0.296, 0.232, 0.0, 0.0]
    gi, tag = refine_contact_start(traj_of(k15w12), 0,
                                   "object_moving_at_start", 7)
    assert (gi, tag) == (3, "contact_refined")

    ov8 = [0.144, 0.060, 0.018, 0.051, 0.140, 0.169, 0.115, 0.077,
           0.075, 0.045, 0.029, 0.093, 0.182, 0.247, 0.162]
    gi, tag = refine_contact_start(traj_of(ov8), 0,
                                   "object_moving_at_start", 6)
    assert (gi, tag) == (3, "contact_refined")

    b5w4 = [0.084, 0.014, 0.055, 0.087, 0.067, 0.007, 0.006, 0.056,
            0.143, 0.196, 0.203, 0.176, 0.159, 0.184, 0.191]
    gi, tag = refine_contact_start(traj_of(b5w4), 0,
                                   "object_moving_at_start", 4)
    assert (gi, tag) == (0, "object_moving_at_start")

    k17 = [0.138, 0.074, 0.084, 0.147, 0.159, 0.097, 0.038, 0.047,
           0.337, 0.595, 0.0, 0.0, 0.0, 0.0, 0.0]
    gi, tag = refine_contact_start(traj_of(k17), 0,
                                   "object_moving_at_start", 5)
    assert (gi, tag) == (0, "object_moving_at_start")

    gi, tag = refine_contact_start(traj_of(k15w12), 7, "cut", 10)
    assert (gi, tag) == (7, "cut")


def test_refine_release_end_measured_profiles():
    from vidforce.data.filtering import refine_release_end

    def traj_of(steps):
        return traj_from_steps(steps, start=(0, 0, 5.0))


    w8 = [0.014, 0.044, 0.100, 0.092, 0.062, 0.033, 0.011, 0.035,
          0.169, 0.294, 0.283, 0.134, 0.066, 0.040, 0.082]
    ae, tag = refine_release_end(traj_of(w8), 0, 10, "capped")
    assert (ae, tag) == (8, "release_cut")

    k14w18 = [0.030, 0.020, 0.015, 0.088, 0.104, 0.080, 0.005, 0.226,
              0.313, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    ae, tag = refine_release_end(traj_of(k14w18), 3, 8, "capped")
    assert (ae, tag) == (7, "release_cut")

    b5w4 = [0.084, 0.014, 0.055, 0.087, 0.067, 0.007, 0.006, 0.056,
            0.143, 0.196, 0.203, 0.176, 0.159, 0.184, 0.191]
    ae, tag = refine_release_end(traj_of(b5w4), 0, 4, "settle")
    assert (ae, tag) == (4, "settle")

    lr7 = [0.104, 0.030, 0.075, 0.107, 0.058, 0.033, 0.002, 0.138,
           0.160, 0.157, 0.157, 0.157, 0.100, 0.155, 0.214]
    ae, tag = refine_release_end(traj_of(lr7), 7, 10, "capped")
    assert (ae, tag) == (10, "capped")

    repull = [0.090, 0.080, 0.030, 0.020, 0.080, 0.090, 0.080, 0.070,
              0.060, 0.050, 0.040, 0.030, 0.020, 0.010, 0.010]
    ae, tag = refine_release_end(traj_of(repull), 0, 10, "capped")
    assert (ae, tag) == (10, "capped")


def test_partial_span_view_rejected():
    from vidforce.data.filtering import score_record

    steps = [0.011, 0.007, 0.030, 0.063, 0.083, 0.094, 0.094, 0.086,
             0.062, 0.030, 0.023, 0.002, 0.009, 0.012, 0.027]
    t = traj_from_steps(steps, start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)

    def proj_late(pt):

        i = int(np.argmin(np.linalg.norm(t - pt, axis=1)))
        return (2.0, 2.0) if i < 8 else (0.5, 0.5)

    row = score_record("wardrobe_1__x__hand", t, is_hand=True,
                       depth_map=dm, project_uv=proj_late,
                       hand_trim={"grip_i": 2, "start": "contact_refined",
                                  "end_i": 15, "end": "ends_slow"})
    assert row["occlusion"] == "partial_span"
    assert row["label"] == "REJECT"


    def proj_barely_late(pt):
        i = int(np.argmin(np.linalg.norm(t - pt, axis=1)))
        return (2.0, 2.0) if i < 3 else (0.5, 0.5)

    row = score_record("wardrobe_1__y__hand", t, is_hand=True,
                       depth_map=dm, project_uv=proj_barely_late,
                       hand_trim={"grip_i": 2, "start": "contact_refined",
                                  "end_i": 15, "end": "ends_slow"})
    assert row["occlusion"] == "visible"
    assert row["label"] == "PASS"


def test_hand_trim_override():
    t = traj_from_steps([0.05] * 15, start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)
    trim = {"grip_i": 2, "end_i": 9, "start": "cut", "end": "object_stop"}
    row = score_record("s__1-2__window_0__hand__iphone_2", t, is_hand=True,
                       depth_map=dm, project_uv=_proj_const((0.5, 0.5)),
                       hand_trim=trim, cap=99.0)
    assert (row["grip_i"], row["end_i"]) == (2, 9)
    assert (row["start"], row["end"]) == ("cut", "object_stop")
    assert row["label"] == "PASS" and row["occlusion"] == "visible"

    row = score_record("s__1-2__window_0__hand__iphone_2", t, is_hand=True,
                       depth_map=np.full((10, 10), 4.0),
                       project_uv=_proj_const((0.5, 0.5)),
                       hand_trim=trim, cap=99.0)
    assert row["label"] == "REJECT" and row["occlusion"] == "occluded"


def test_traj_outlier_guard():
    t = traj_from_steps([0.35] * 15, start=(0, 0, 5.0))
    dm = np.full((10, 10), 4.9)
    row = score_record("s__1-2__window_0__gripper__iphone_1", t, is_hand=False,
                       depth_map=dm, project_uv=_proj_const((0.5, 0.5)))
    assert row["label"] == "REJECT" and row["occlusion"] == "traj_outlier"
