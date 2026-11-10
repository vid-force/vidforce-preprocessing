#!/usr/bin/env python3
"""First-frame package of every conditioned record.

The first frame is the contact frame: the first trigger contact of a gripper recording (its stored window starts
0.3 s earlier) and the start of the trimmed span of a human recording. A package holds that frame with its full-frame
intrinsics, the span of the trajectory in the camera frame of the first frame, the contact pixel and depth
(re-projected from the first point of the span), the object and scene windows as exact pixel crops, the articulation
axis and joint coordinate (gripper), the force targets of a gripper recording (peak, 32-sample profile over the span
rescaled to the peak, smoothed waypoint forces, peak count), and the first-frame checks of
``vidforce/data/visibility.py`` with their verdict ``record_valid``.

  python scripts/data/build_first_image_corpus.py --sidecar <conditioning.json> --corpus <records> \
      --out-dir <packages> [--shard i --num-shards n]
  ... --manifest            summary of the package directory
  ... --validate N          consistency checks on N random packages
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vidforce.data.articulation import (
    JOINT_UNITS, joint_coordinates)
from vidforce.data.conditioning import first_frame_index
from vidforce.data import visibility as visg
from vidforce.data.force_stats import (
    FORCE_BIN_EDGES_N, contact_phase_span, force_class, force_profile,
    detect_peaks, peak_time_s, scale_to_peak, smooth_waypoint_force)

CROP_KEYS_DOC = "scene,object"


def contact_frame_index(frame_times_ns, waypoint_times_s) -> int:
    ft = np.asarray(frame_times_ns, np.float64) / 1e9
    wt = np.asarray(waypoint_times_s, np.float64)
    lead = max(0.0, (ft[-1] - ft[0]) - float(wt[-1] - wt[0]))
    return int(np.argmin(np.abs(ft - ft[0] - lead)))


def _full_K_from_span(entry) -> np.ndarray:
    K = np.asarray(entry["K"], np.float64).copy()
    K[0, 2] += entry["box"][0]
    K[1, 2] += entry["box"][1]
    return K


def _span_entry(rgb_f, entry):
    b = entry["box"]
    return {"img": np.ascontiguousarray(rgb_f[b[1]:b[3], b[0]:b[2]]),
            "box": list(b),
            "K": np.asarray(entry["K"], np.float64).copy()}


def build_one(name: str, r: dict, rec: dict) -> dict:
    rgb = np.asarray(rec["rgb"])
    ft = np.asarray(rec["frame_times_ns"], np.float64)
    wt = np.asarray(rec["waypoint_times_s"], np.float64)
    si, ae = r["trim"]["start_i"], r["trim"]["end_i"]


    is_gripper = "dense_contact" in rec or r["embodiment"] == "gripper"
    if is_gripper:
        f0 = contact_frame_index(ft, wt)
        f0 = int(np.clip(f0, 0, len(ft) - 1))
    else:
        f0 = int(r.get("first_frame", first_frame_index(ft, wt, si)))
    fE = int(first_frame_index(ft, wt, ae))
    ok = bool(r.get("contact_uv"))

    T0 = np.asarray(rec["T_world_iphone"], np.float64)
    traj = np.asarray(rec["traj_m"], np.float64)[si:ae + 1]
    K = _full_K_from_span(r["spans"]["object"]) if ok else None
    depth_map = rec.get("depth_map")


    if ok and K is not None and len(traj) and traj[0, 2] > 0:
        cu = [float(traj[0, 0] / traj[0, 2] * K[0, 0] + K[0, 2]),
              float(traj[0, 1] / traj[0, 2] * K[1, 1] + K[1, 2])]
        cuv_norm = [cu[0] / rgb.shape[2], cu[1] / rgb.shape[1]]
        c_depth = float(traj[0, 2])
    else:
        cu = r.get("contact_uv")
        cuv_norm = r.get("contact_uv_norm")
        c_depth = r.get("contact_depth_m")


    axis = axis_origin = joint_type = work_j = None
    if "axis" in rec:
        axis = np.asarray(rec["axis"], np.float64).copy()
        axis_origin = np.asarray(rec["axis_origin"], np.float64).copy()
        axis = (axis / np.linalg.norm(axis)).astype(np.float32)
        axis_origin = axis_origin.astype(np.float32)
        joint_type = rec.get("joint_type")
        work_j = float(rec["work_j"])


        joint_q = joint_coordinates(traj, axis, axis_origin,
                                    joint_type).astype(np.float32)
    else:
        joint_q = None
    joint_q_unit = JOINT_UNITS[joint_type] if joint_type in JOINT_UNITS else None


    fv = rec.get("force_valid")
    if fv is None:
        fv = ("dense_contact" in rec) and float(rec["peak_force_n"]) > 0.0
    force_valid = bool(fv)

    prov = rec.get("provenance") or {}
    labels = {
        "action": prov.get("action"),
        "hand_used": prov.get("hand_used"),
        "body_speed_median_m_s": prov.get("body_speed_median_m_s"),
        "axis_fit_residual_rms_m": prov.get("axis_fit_residual_rms_m"),
        "wrench_source": prov.get("wrench_source"),
    }


    if force_valid:
        wt_span = wt[si:ae + 1]
        F_vec = np.asarray(rec["force_vec_n"], np.float64)[si:ae + 1]
        mag = np.linalg.norm(F_vec, axis=1)
        wt_rel = wt_span - float(wt_span[0])
        t0, tE = 0.0, float(wt_rel[-1])
        phase = contact_phase_span(
            rec.get("dense_force_mag_n"), rec.get("dense_time_s"),
            rec.get("dense_contact"))
        if phase is not None:
            dmag, dtime = phase
        else:
            dmag = dtime = None
        profile = force_profile(dmag, dtime, t0, tE, wt_rel, mag)
        profile = scale_to_peak(profile, float(rec["peak_force_n"]))
        pt, pm, npk = detect_peaks(dmag, dtime, t0, tE, wt_rel, mag)
        force_stats = {
            "force_smooth_n": smooth_waypoint_force(F_vec).astype(np.float32),
            "force_profile_n": profile.astype(np.float32),
            "force_class": force_class(float(rec["peak_force_n"])),
            "force_bin_edges": np.asarray(FORCE_BIN_EDGES_N, np.float32),
            "peak_time_s": peak_time_s(profile, t0, tE),
            "peak_force_smooth_n": float(profile.max()),
            "n_peaks": int(npk),
            "peaks_time_s": pt.astype(np.float32),
            "peaks_force_n": pm.astype(np.float32),
        }
    else:
        force_stats = {
            "force_smooth_n": None, "force_profile_n": None, "force_class": None,
            "force_bin_edges": None, "peak_time_s": None,
            "peak_force_smooth_n": None, "n_peaks": None,
            "peaks_time_s": None, "peaks_force_n": None,
        }

    out = {
        "name": name,
        "embodiment": r["embodiment"],
        "camera": r["camera"], "episode": r["episode"], "split": r["split"],
        "prompt_text": r["prompt_text"], "K_source": r.get("K_source"),
        "zoom_default": r.get("zoom_default"),
        "status": "ok" if ok else "no_k",
        "start_i": si, "end_i": ae, "first_frame": f0,
        "start_ts_ns": float(ft[f0]), "end_ts_ns": float(ft[fE]),
        "frame_times_ns": ft,
        "span_waypoint_times_s": wt[si:ae + 1],
        "rgb_first": rgb[f0],
        "K_first": K,
        "T_world_camera_first": T0,
        "traj_cam_first": traj.astype(np.float32),
        "depth_waypoints_m": traj[:, 2].astype(np.float32),
        "contact_uv": cu,
        "contact_uv_norm": cuv_norm,
        "contact_depth_m": c_depth,
        "contact_point_cam": traj[0].astype(np.float32) if len(traj) else None,
        "axis": axis, "axis_origin": axis_origin,
        "joint_type": joint_type, "work_j": work_j,
        "joint_q": joint_q, "joint_q_unit": joint_q_unit,
        "force_valid": force_valid, "labels": labels,
        "bbox": r.get("bbox"),
        "spans": r.get("spans"),
        "depth_map": depth_map,
        "force_vec_n": np.asarray(rec["force_vec_n"])[si:ae + 1],
        "peak_force_n": float(rec["peak_force_n"]),
        "contact": np.asarray(rec["contact"])[si:ae + 1],
        **force_stats,
    }

    if ok:
        first = rgb[f0]
        obj = r["spans"]["object"]
        obj_entries = ({"object": obj} if "box" in obj
                       else obj)
        scene_entry = ({"box": [0, 0, rgb.shape[2], rgb.shape[1]], "K": K}
                       if r["spans"].get("scene") is None
                       else r["spans"]["scene"])
        out["crops"] = {"scene": _span_entry(first, scene_entry),
                        "object": {k: _span_entry(first, v)
                                   for k, v in obj_entries.items()}}
    else:
        out["crops"] = None


    crop_obj = (out["crops"] or {}).get("object")
    out.update(visg.visibility_fields(
        status_ok=ok, cu=out["contact_uv"], K=K, traj=out["traj_cam_first"],
        rgb_shape=rgb[f0].shape, depth_map=depth_map, crop_entry=crop_obj,
        contact_depth=out["contact_depth_m"],
        zoom_default=out.get("zoom_default")))
    return out


def validate(pkgs_dir, sidecar_rows, n, seed):
    import random
    rng = random.Random(seed)
    names = sorted(n for n in sidecar_rows
                   if (pkgs_dir / f"{n}.pt").exists())
    if not names:
        print("no packages found", flush=True)
        return 1
    checked = 0
    for name in rng.sample(names, min(n, len(names))):
        p = torch.load(pkgs_dir / f"{name}.pt", weights_only=False,
                       map_location="cpu")
        rgb, K = np.asarray(p["rgb_first"]), p["K_first"]

        if p["crops"]:
            entries = list(p["crops"]["object"].values()) \
                + [p["crops"]["scene"]]
            for e in entries:
                b = e["box"]
                assert np.array_equal(
                    e["img"], rgb[b[1]:b[3], b[0]:b[2]]), name

        if p["contact_uv"] and K is not None:
            c = np.asarray(p["contact_point_cam"], np.float64)
            u = c[0] / c[2] * K[0, 0] + K[0, 2]
            v = c[1] / c[2] * K[1, 1] + K[1, 2]
            assert abs(u - p["contact_uv"][0]) <= .05, name
            assert abs(v - p["contact_uv"][1]) <= .05, name

        ft = p["frame_times_ns"]
        assert p["start_ts_ns"] == ft[p["first_frame"]], name


        if name.split("__")[-2] == "gripper":
            assert p["first_frame"] == contact_frame_index(
                p["frame_times_ns"], p["span_waypoint_times_s"]), name
        if p["contact_point_cam"] is not None:
            assert np.allclose(p["contact_point_cam"],
                               np.asarray(p["traj_cam_first"])[0], atol=1e-6), \
                name


        if name.split("__")[-2] == "gripper":
            assert p["axis"] is not None and p["joint_type"], name
            assert abs(float(np.linalg.norm(p["axis"])) - 1.0) <= 1e-5, name
            assert np.isfinite(p["axis_origin"]).all() \
                and np.isfinite(p["work_j"]), name


            q = np.asarray(p["joint_q"], np.float64)
            assert q.shape == (len(p["traj_cam_first"]),) and q[0] == 0.0 \
                and np.isfinite(q).all(), name
            assert p["joint_q_unit"] == JOINT_UNITS[p["joint_type"]], name
            ref = joint_coordinates(p["traj_cam_first"], p["axis"],
                                    p["axis_origin"], p["joint_type"])
            assert np.allclose(q, ref, atol=1e-5), name


        assert isinstance(p["force_valid"], bool), name
        assert p["force_valid"] == (float(p["peak_force_n"]) > 0.0), name
        if name.split("__")[-2] == "hand":


            assert (not p["force_valid"]) or p["axis"] is not None, name

        if p["force_valid"]:
            prof = np.asarray(p["force_profile_n"])
            dur = float(p["span_waypoint_times_s"][-1]
                        - p["span_waypoint_times_s"][0])
            assert prof.shape == (32,) and (prof >= 0).all(), name
            assert 0.0 <= p["peak_time_s"] <= dur, name
            assert p["n_peaks"] >= 1 and len(p["peaks_time_s"]) == p["n_peaks"], name


            assert abs(p["peak_force_smooth_n"] - p["peak_force_n"]) \
                <= 1e-3 * p["peak_force_n"], name
            assert p["force_class"] == force_class(p["peak_force_n"]), name
            assert np.allclose(p["force_bin_edges"], FORCE_BIN_EDGES_N), name
            assert p["force_smooth_n"].shape \
                == np.asarray(p["force_vec_n"]).shape, name
        else:
            assert p["force_profile_n"] is None \
                and p["force_class"] is None and p["n_peaks"] is None, name
        assert np.allclose(p["depth_waypoints_m"],
                           np.asarray(p["traj_cam_first"])[:, 2], atol=1e-6), \
            name


        ref = visg.visibility_fields(
            status_ok=(p["status"] == "ok"), cu=p["contact_uv"],
            K=p["K_first"], traj=p["traj_cam_first"],
            rgb_shape=np.asarray(p["rgb_first"]).shape,
            depth_map=p.get("depth_map"),
            crop_entry=(p["crops"] or {}).get("object"),
            contact_depth=p["contact_depth_m"],
            zoom_default=p.get("zoom_default"))
        for k in ("contact_in_frame", "traj_in_frame_rate",
                  "front_offset_m", "object_visible_frac", "record_valid"):
            got, want = p.get(k), ref[k]
            if got is None or want is None:
                assert got is want, (name, k, got, want)
            elif isinstance(want, float):
                assert abs(got - want) <= 1e-6, (name, k, got, want)
            else:
                assert got == want, (name, k, got, want)
        assert isinstance(p["record_valid"], bool), name

        checked += 1
    print(f"validated {checked} packages: crops/contact/ts/depth/traj OK",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", type=pathlib.Path, required=True,
                    help="conditioning file written by make_conditioning.py")
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="records written by build_full_corpus.py")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    doc = json.loads(a.sidecar.read_text())
    rows = doc["records"]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.validate:
        return validate(a.out_dir, rows, a.validate, a.seed)
    if a.manifest:
        return write_manifest(a, rows)

    names = sorted(n for i, n in enumerate(rows) if i % a.num_shards
                   == a.shard)
    if a.limit:
        names = names[:a.limit]
    for i, name in enumerate(names):
        r = rows[name]
        rec = torch.load(a.corpus / f"{name}.pt", weights_only=False,
                         map_location="cpu")
        pkg = build_one(name, r, rec)
        del rec
        torch.save(pkg, a.out_dir / f"{name}.pt")
        del pkg
        if (i + 1) % 50 == 0:
            print(f"[shard {a.shard}] {i + 1}/{len(names)}", flush=True)
    print(f"SHARD_{a.shard}_DONE ({len(names)})", flush=True)
    return 0


def write_manifest(a, rows):
    from collections import Counter
    have = {p.stem for p in a.out_dir.glob("*.pt")}
    cnt, missing = Counter(), []
    for n, r in rows.items():
        if n not in have:
            missing.append(n)
        cnt[(r["embodiment"],
             "ok" if r.get("contact_uv") else "missing_geometry")] += 1


    train_peaks, cls_cnt = [], Counter()
    prof_ratio = []
    vis_cnt, vis_fail = Counter(), Counter()
    for q in sorted(a.out_dir.glob("*.pt")):
        p = torch.load(q, weights_only=False, map_location="cpu")
        key = p["embodiment"]
        vis_cnt[(key, p["record_valid"])] += 1
        if not p["record_valid"]:
            if p["status"] != "ok":
                vis_fail["status_" + p["status"]] += 1
            if not p["contact_in_frame"]:
                vis_fail["contact_out_of_frame"] += 1
            tifr = p["traj_in_frame_rate"]
            if tifr is None:
                vis_fail["no_traj_rate"] += 1
            elif tifr < visg.MIN_TRAJ_IN_FRAME:
                vis_fail["traj_below_%.0f%%" % (visg.MIN_TRAJ_IN_FRAME * 100)] += 1
            if p["object_visible_frac"] is not None \
                    and p["object_visible_frac"] < visg.MIN_OBJECT_VISIBLE:
                vis_fail["object_occluded_lt_%.0f%%"
                         % (visg.MIN_OBJECT_VISIBLE * 100)] += 1
        if p["force_valid"]:
            cls_cnt[p["force_class"]] += 1
            prof_ratio.append(p["peak_force_smooth_n"] / p["peak_force_n"])
            if p["split"] == "train":
                train_peaks.append(float(p["peak_force_n"]))
    edges_now = np.percentile(train_peaks, [20, 40, 60, 80])
    if len(train_peaks) > 100:
        assert np.abs(edges_now - FORCE_BIN_EDGES_N).max() < 0.5, \
            f"train quintiles drifted: {edges_now} vs {FORCE_BIN_EDGES_N}"
    sha = hashlib.sha256(a.sidecar.read_bytes()).hexdigest()
    man = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "sidecar": str(a.sidecar), "sidecar_sha256": sha,
        "sidecar_records": len(rows), "packages": len(have),
        "missing": missing,
        "counts": {"|".join(k): v for k, v in sorted(cnt.items())},
        "crop_keys": CROP_KEYS_DOC,
        "visibility": {
            "thresholds": {"min_traj_in_frame": visg.MIN_TRAJ_IN_FRAME,
                           "min_object_visible": visg.MIN_OBJECT_VISIBLE,
                           "margin_m": visg.MARGIN_M},
            "valid_by_embodiment": {"|".join(map(str, k)): v for k, v in
                              sorted(vis_cnt.items())},
            "invalid_reasons": dict(sorted(vis_fail.items())),
        },
        "force_stats": {
            "n_force_valid": sum(cls_cnt.values()),
            "train_n": len(train_peaks),
            "bin_edges_N": list(map(float, FORCE_BIN_EDGES_N)),
            "train_quintiles_recomputed_N": edges_now.round(2).tolist(),
            "class_counts_all_splits": {str(k): v for k, v in sorted(cls_cnt.items())},


            "profile_peak_ratio_q": np.percentile(prof_ratio,
                                                  [5, 50, 95]).round(3).tolist(),
        },
        "definitions": "force profile: Savitzky-Golay-smoothed |F| of the contact phase of the dense stream on a uniform 32-point grid over the span, rescaled so that its maximum equals the measured peak; peaks: prominence 0.25 x span maximum, at least 0.15 s apart; visibility: contact in frame, at least half of the span in frame, at least 30 % of the object window within 0.35 m of the contact depth",
        "note": "first image = contact frame (human: start of the trimmed span; gripper: first trigger contact); crops are the exact pixels of the fixed windows; K_first is the full-frame pinhole recovered from the intrinsics of the object window; depth_map pairs with K_first",
    }
    (a.out_dir / "manifest_first_image_v1.json").write_text(
        json.dumps(man, indent=1))
    print(json.dumps(man["counts"], indent=1), flush=True)
    print(f"packages {man['packages']}/{len(rows)} missing {len(missing)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
