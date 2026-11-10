#!/usr/bin/env python3
"""Scalar fields of the first-frame packages.

Reads every package once and writes the two files the selection stage joins: ``index_first_image.json`` (per package:
status, span length, ``record_valid``) and ``first_image_v1_exo_fields.json`` (per package: its scalar fields, the
first-frame checks and the action).

  python scripts/data/snapshot_first_image_fields.py --pkg-dir <packages> --out-dir <fields>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

SCALARS = ("name", "embodiment", "camera", "episode", "split", "prompt_text", "K_source",
           "zoom_default", "status", "start_i", "end_i", "first_frame", "contact_depth_m",
           "joint_type", "work_j", "joint_q_unit", "force_valid", "peak_force_n", "force_class",
           "peak_time_s", "peak_force_smooth_n", "n_peaks", "contact_in_frame", "traj_in_frame_rate",
           "front_offset_m", "object_visible_frac", "record_valid")


def py(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else None
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows, fields = [], {}
    t0 = time.time()
    files = sorted(a.pkg_dir.glob("*.pt"))
    for i, p in enumerate(files):
        try:
            d = torch.load(p, weights_only=False, map_location="cpu")
        except Exception as exc:
            print(f"FAILED {p.name}: {exc}", flush=True)
            continue
        name = d.get("name") or p.stem
        span = d.get("span_waypoint_times_s")
        try:
            span_len = int(len(span)) if span is not None else (int(d["end_i"]) - int(d["start_i"]) + 1)
        except Exception:
            span_len = None
        rows.append({"name": name, "embodiment": d.get("embodiment"),
                     "split": d.get("split"), "status": d.get("status"), "span_len": span_len,
                     "has_axis": d.get("axis") is not None, "record_valid": bool(d.get("record_valid", False))})
        f = {k: py(d.get(k)) for k in SCALARS}
        labels = d.get("labels") or {}
        f["labels"] = {"action": labels.get("action")} if isinstance(labels, dict) else {}
        fields[name] = f
        if i % 200 == 0:
            print(f"[{i}/{len(files)}] {name} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"rows": rows}, open(a.out_dir / "index_first_image.json", "w"))
    json.dump(fields, open(a.out_dir / "first_image_v1_exo_fields.json", "w"))
    ok = sum(1 for r in rows if r["status"] == "ok")
    rv = sum(1 for r in rows if r["record_valid"])
    print(f"packages {len(rows)} (status ok {ok}, record_valid {rv}); exo fields {len(fields)} -> {a.out_dir}")


if __name__ == "__main__":
    main()
