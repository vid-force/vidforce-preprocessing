#!/usr/bin/env python3
"""Build the record of every recording of the window inventory.

One record per recording, named ``{scene}__{session}__{window}__{gripper|hand}__{camera}.pt``: the frames, the contact
trajectory in the camera frame at 16 waypoints, the camera pose, depth with intrinsics, the prompt and, for gripper
recordings, the measured force (``vidforce/data/corpus_build.py``, ``vidforce/data/hand_records.py``). The run is
resumable (existing records are skipped) and writes one build manifest per shard; a recording for which the release
lacks an input is a manifest row with the reason, not an error.

  python scripts/data/build_full_corpus.py --root $HOI_ROOT --out <records> --inventory <window_inventory.json> \
      [--shard i/n]
  ... --names-file <sequence names> --num-frames 32 --depth-source da3 --depth-frames all     (final records)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vidforce.data.corpus_build import build_exo_record
from vidforce.data.hand_records import build_hand_record


def universe(inventory: dict) -> list[str]:
    names = []
    for key, t in inventory["trials"].items():
        scene, trial = key.split("|")
        for emb in ("hand", "gripper"):
            for w in t[f"{emb}_windows"]:
                for v in t[f"{emb}_views"]:
                    names.append(f"{scene}__{trial}__{w}__{emb}__{v}")
    return sorted(names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--inventory", type=pathlib.Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--frame-size", type=int, default=640)
    ap.add_argument("--only", default="",
                    help="comma-separated record names (to build a few records)")
    ap.add_argument("--names-file", type=pathlib.Path, default=None,
                    help="restrict the build to the record names listed one per line")
    ap.add_argument("--depth-source", choices=("lidar", "da3"), default="lidar",
                    help="depth of human recordings: the phone's depth stream (lidar) or Depth Anything 3 (da3)")
    ap.add_argument("--depth-frames", choices=("first", "all"), default="first",
                    help="Depth Anything 3 depth of the first frame only, or of every frame")
    a = ap.parse_args(argv)

    i, n = (int(x) for x in a.shard.split("/"))
    names = universe(json.loads(a.inventory.read_text()))
    if a.only:
        names = [x for x in names
                 if x in {s.strip() for s in a.only.split(",")}]
    if a.names_file is not None:
        wanted = {x.strip() for x in a.names_file.read_text().split() if x.strip()}
        missing = sorted(wanted - set(names))
        if missing:
            raise SystemExit(f"{len(missing)} names not in the inventory universe: {missing[:5]}")
        names = [x for x in names if x in wanted]
    names = names[i::n]
    a.out.mkdir(parents=True, exist_ok=True)
    if i == 0:
        (a.out / "corpus_config.json").write_text(json.dumps({
            "num_frames": a.num_frames, "frame_size": a.frame_size,
            "depth_source_hand": a.depth_source, "depth_source_gripper": "da3",
            "depth_frames": a.depth_frames,
            "names_file": str(a.names_file) if a.names_file else None,
            "inventory": str(a.inventory)}, indent=1))

    rows, t0, n_ok = [], time.time(), 0
    for name in names:
        scene, trial, window, emb, view = name.split("__")
        out_path = a.out / f"{name}.pt"
        if out_path.exists():
            rows.append({"record": name, "status": "cached"})
            n_ok += 1
            continue
        try:
            if emb == "hand":
                rec = build_hand_record(
                    a.root, scene=scene, trial=f"{scene}_{trial}_hand",
                    iphone_module=view, window_name=window,
                    num_frames=a.num_frames, num_waypoints=16,
                    frame_size=a.frame_size, label_span_mode="manipulation",
                    depth_source=a.depth_source, depth_frames=a.depth_frames)
                payload = {
                    "rgb": rec.rgb, "frame_times_ns": rec.frame_times_ns,
                    "depth_map": rec.depth_map, "depth_K": rec.depth_K,
                    "depth_maps": rec.depth_maps, "depth_Ks": rec.depth_Ks,
                    "traj_m": rec.traj_m,
                    "waypoint_times_s": rec.waypoint_times_s,
                    "force_valid": rec.force_valid,
                    "force_vec_n": rec.force_vec_n,
                    "peak_force_n": rec.peak_force_n, "contact": rec.contact,
                    "T_world_iphone": rec.T_world_iphone,
                    "prompt": rec.prompt, "provenance": rec.provenance,
                }
            else:
                rec = build_exo_record(
                    a.root, scene=scene, trial=f"{scene}_{trial}_gripper",
                    window_name=window, camera_module=view,
                    reference_camera="iphone", num_frames=a.num_frames,
                    num_waypoints=16, pre_contact_s=0.3, watch_s=9999.0,
                    include_ego=False, include_gripper=True,
                    frame_size=a.frame_size, include_depth=True,
                    depth_frames=a.depth_frames)
                payload = {
                    "rgb": rec.rgb, "frame_times_ns": rec.frame_times_ns,
                    "rgb_gripper": rec.rgb_gripper,
                    "frame_times_gripper_ns": rec.frame_times_gripper_ns,
                    "depth_map": rec.depth_map, "depth_K": rec.depth_K,
                    "depth_maps": rec.depth_maps, "depth_Ks": rec.depth_Ks,
                    "axis": rec.axis, "axis_origin": rec.axis_origin,
                    "joint_type": rec.joint_type, "work_j": rec.work_j,
                    "prompt": rec.prompt, "traj_m": rec.traj_m,
                    "force_vec_n": rec.force_vec_n,
                    "waypoint_times_s": rec.waypoint_times_s,
                    "contact": rec.contact,
                    "peak_force_n": rec.peak_force_n,
                    "T_world_iphone": rec.T_world_iphone,
                    "dense_time_s": rec.dense_time_s,
                    "dense_force_mag_n": rec.dense_force_mag_n,
                    "dense_contact": rec.dense_contact,
                    "provenance": rec.provenance,
                }
        except Exception as e:
            rows.append({"record": name, "status": "skip",
                         "reason": f"{type(e).__name__}: {e}"[:300]})
            continue
        payload["camera"] = view
        torch.save(payload, out_path)
        rows.append({"record": name, "status": "ok"})
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"[build {a.shard}] {n_ok}/{len(names)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    (a.out / f"corpus_manifest_shard{i}of{n}.json").write_text(
        json.dumps(rows, indent=1))
    n_skip = sum(1 for r in rows if r["status"] == "skip")
    print(f"[build {a.shard}] DONE {n_ok} ok / {n_skip} skip "
          f"/ {len(names)} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
