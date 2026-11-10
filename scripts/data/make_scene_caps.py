#!/usr/bin/env python3
"""Per-scene cap on the path length of a human span.

cap(scene) = median path length of the gripper records of the scene that passed the filter (a gripper recording covers
the articulation from start to end by construction).

  python scripts/data/make_scene_caps.py <merged gripper scores.json> <scene_caps.json>
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

scores = json.loads(pathlib.Path(sys.argv[1]).read_text())
by_scene: dict[str, list[float]] = {}
for r in scores:
    if r["embodiment"] == "gripper" and r["label"] == "PASS":
        by_scene.setdefault(r["name"].split("__")[0], []).append(r["sup_len"])
caps = {s: {"median": round(statistics.median(v), 3), "n": len(v)}
        for s, v in sorted(by_scene.items()) if v}
pathlib.Path(sys.argv[2]).write_text(json.dumps(caps, indent=1))
print(json.dumps(caps, indent=1))
print(f"CAPS_DONE {len(caps)} scenes")
