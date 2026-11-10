#!/usr/bin/env python3
"""First-frame conditioning of every record that passed the filter.

For each record with a PASS or AMBIGUOUS verdict the output holds the intrinsics source, the first-frame contact
pixel and depth, the object window (a square crop around the contact that covers the projected articulation span) and
the wider scene window with their exact pinhole intrinsics, the trimmed span, the prompt and the split of the episode.

Intrinsics. A gripper record stores the Depth-Anything-3 intrinsics of its first frame at the depth resolution; they
are scaled to the image resolution. A human record takes the median of those intrinsics over the gripper records of
the same scene and camera (the same physical camera); where no such record exists, it has no intrinsics
(``K_source: none``) and cannot become a sequence.

  python scripts/data/make_conditioning.py --corpus <records> --filter <record_filter.json> --out <conditioning.json>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vidforce.data.conditioning import exo_conditioning, scale_K
from vidforce.data.splits import load_eval_episodes, split_of

VALID = ("PASS", "AMBIGUOUS")


def exo_K(rec, cache, key, W: int, H: int):
    try:
        K = np.asarray(rec.get("depth_K"), np.float64).reshape(3, 3)
        src_w = np.asarray(rec["depth_map"]).shape[1]
        return scale_K(K, src_w, W, H), "stored_depth_K", cache
    except Exception:
        pass
    if key in cache:
        return scale_K(cache[key], SIB_SRC_W, W, H), \
            f"sibling_depth_K({key})", cache
    return None, "none", cache


SIB_SRC_W = 160


def collect_sibling_K(corpus: pathlib.Path) -> dict:
    per_cam: dict[tuple, list] = collections.defaultdict(list)
    for f in sorted(corpus.glob("*.pt")):
        parts = f.stem.split("__")
        if len(parts) < 4 or parts[-2] != "gripper":
            continue
        try:
            r = torch.load(f, weights_only=False, map_location="cpu")
            K = np.asarray(r.get("depth_K"), np.float64).reshape(3, 3)
            src_w = np.asarray(r["depth_map"]).shape[1]
        except Exception:
            continue
        if src_w != SIB_SRC_W:
            raise SystemExit(f"unexpected depth res {src_w} in {f.name}")
        per_cam[(parts[0], parts[-1])].append(K)
        del r
    return {k: np.median(np.stack(v), axis=0) for k, v in per_cam.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="records written by build_full_corpus.py")
    ap.add_argument("--filter", type=pathlib.Path, required=True,
                    help="verdict file written by make_filter_registry.py")
    ap.add_argument("--eval-episodes", type=pathlib.Path,
                    default=pathlib.Path("splits/eval_episodes.json"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--only", default="",
                    help="comma-separated record names (to process a few records)")
    a = ap.parse_args(argv)

    verdicts = json.loads(a.filter.read_text())["records"]
    eval_episodes = load_eval_episodes(a.eval_episodes)
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    names = sorted(n for n, v in verdicts.items()
                   if v["label"] in VALID and (not only or n in only))

    sib = collect_sibling_K(a.corpus)
    print(f"sibling K for {len(sib)} (scene, camera) pairs", flush=True)

    rows: dict[str, dict] = {}
    counts = collections.Counter()
    for name in names:
        f = a.corpus / f"{name}.pt"
        if not f.exists():
            print(f"  record not in the corpus (skipped): {name}", file=sys.stderr)
            continue
        scene, trial, window, emb, _camera = name.split("__")
        episode = f"{scene}__{trial}__{window}"
        split, _partition = split_of(episode, eval_episodes)
        rec = torch.load(f, weights_only=False, map_location="cpu")
        rgb = np.asarray(rec["rgb"])
        H, W = rgb.shape[1:3]
        trim = {"start_i": verdicts[name]["start_i"],
                "end_i": verdicts[name]["end_i"]}
        key = ((rec.get("provenance") or {}).get("scene", "?"),
               rec.get("camera"))
        K, ksrc, _ = exo_K(rec, sib, key, W, H)
        row = {"episode": episode, "embodiment": emb,
               "camera": rec.get("camera"), "split": split,
               "prompt_text": rec.get("prompt"), "trim": trim,
               "K_source": ksrc}
        if K is None:
            row["behind_camera"] = None
        else:
            row.update(exo_conditioning(np.asarray(rec["traj_m"]), K,
                                        trim["start_i"], trim["end_i"],
                                        W=W, H=H))
        rows[name] = row
        counts["records"] += 1
        counts[f"K_{ksrc.split('(')[0]}"] += 1
        del rec

    meta = {
        "semantics": "first-frame conditioning; fixed windows for the whole clip; "
                     "3D ground truth is unchanged by the crop (K' per window)",
        "counts": dict(counts),
        "K_convention": "K at image resolution: gripper = stored first-frame depth K scaled from the depth "
                        "resolution; human = median of the gripper K of the same scene and camera; else none",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": meta, "records": rows}, indent=1))
    print(f"{counts['records']} records -> {a.out}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
