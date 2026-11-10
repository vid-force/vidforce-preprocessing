#!/usr/bin/env python3
"""Score records with the quality filter.

For each record the filter (``vidforce/data/filtering.py``) returns a label (PASS / AMBIGUOUS / REJECT), a visibility
tag, the depth-ray gap of the contact and the articulation span. Depth comes from the record; a human record without
stored intrinsics gets one Depth Anything 3 pass. A person mask of the first frame (Mask R-CNN) detects recordings in
which the person stands between the camera and the object.

Modes: ``full`` scores every record of the corpus (``--shard i/n`` for parallel shards, ``--embodiment`` to restrict),
``merge`` combines the shard files into ``full_corpus_scores.json``, ``sample`` / ``names`` score a few records, and
``--viz`` renders review figures. Gripper records are scored first; the per-scene caps derived from them
(``make_scene_caps.py``) are then passed to the scoring of the human records.

  python scripts/data/filter_records.py --corpus <records> --out-dir <scores> --mode full --embodiment gripper \
      --shard 0/8
  python scripts/data/filter_records.py --corpus <records> --out-dir <scores> --mode merge
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vidforce.data.filtering import (
    DEFAULT_PARAMS, score_record,
)
from vidforce.data.projection import project_point_to_uv


def load_depth_K(rec, rgb):
    H, W = rgb.shape[1:3]
    try:
        K = np.asarray(rec.get("depth_K"), np.float64).reshape(3, 3)
        dm = np.asarray(rec["depth_map"], np.float64)
        dH, dW = dm.shape[:2]
        return K, dm, dW, dH
    except Exception:
        from vidforce.perception.depth import metric_depth
        dmm, K = metric_depth(rgb[0])
        return K, np.asarray(dmm) / 1000.0, W, H


def render(rgb, traj, project_uv, row, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = rgb.shape[1:3]
    gi, ae = row["start_i"], row["end_i"]
    uvs = []
    for pt in traj:
        uv = project_uv(pt)
        uvs.append(uv if (uv is not None and 0 <= uv[0] <= 1
                          and 0 <= uv[1] <= 1) else None)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(rgb[0])
    ok = [i for i, u in enumerate(uvs) if u is not None]
    if ok:
        xs = [uvs[i][0] * W for i in ok]
        ys = [uvs[i][1] * H for i in ok]
        ax.plot(xs, ys, "-", color="black", lw=6, alpha=0.8)
        ax.plot(xs, ys, "-", color="white", lw=3)
        sup = [i for i in ok if gi <= i <= ae]
        if len(sup) >= 2:
            ax.plot([uvs[i][0] * W for i in sup],
                    [uvs[i][1] * H for i in sup], "-", color="navy", lw=5,
                    alpha=0.95)
            ax.plot([uvs[i][0] * W for i in sup],
                    [uvs[i][1] * H for i in sup], "-", color="dodgerblue", lw=3)
        ax.plot(xs[0], ys[0], "*", color="black", ms=20, mew=1)
        ax.plot(xs[0], ys[0], "*", color="lime", ms=17)
        ax.plot(xs[-1], ys[-1], "*", color="black", ms=20, mew=1)
        ax.plot(xs[-1], ys[-1], "*", color="red", ms=17)
        for idx, col in ((row["grip_i"], "yellow"), (ae, "deepskyblue")):
            if 0 <= idx < len(uvs) and uvs[idx] is not None:
                ax.plot(uvs[idx][0] * W, uvs[idx][1] * H, "o", color="black",
                        ms=20, mew=2)
                ax.plot(uvs[idx][0] * W, uvs[idx][1] * H, "o", color=col,
                        ms=17, mfc="none", mew=3)
    color = {"PASS": "green", "AMBIGUOUS": "orange", "REJECT": "red"}[row["label"]]
    sup_txt = f" sup={row['sup_len']:.2f}m" if row["embodiment"] == "hand" else ""
    shift = f" +{row['start_shift']}f" if row["start_shift"] > 0 else ""
    ax.set_title(f"{row['label']}  [{row['embodiment']}]  occ={row['occlusion']}"
                 f"  end={row['end']}{sup_txt}{shift}",
                 fontsize=9, color=color, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


_PERSON_MODEL = None


def person_mask_for(rgb0, device="cuda", score_thr=0.6):
    global _PERSON_MODEL
    import torchvision
    if _PERSON_MODEL is None:
        w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _PERSON_MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=w).to(device).eval()
        print("[filter] person-mask model loaded (one-time)", flush=True)
    img = torch.from_numpy(np.ascontiguousarray(rgb0)).permute(2, 0, 1) \
        .float().div(255.0).to(device)
    with torch.no_grad():
        out = _PERSON_MODEL([img])[0]
    keep = (out["labels"] == 1) & (out["scores"] >= score_thr)
    if not bool(keep.any()):
        return None
    return (out["masks"][keep, 0] > 0.5).any(dim=0).cpu().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=pathlib.Path, required=True)
    ap.add_argument("--scene-caps", type=pathlib.Path, default=None)
    ap.add_argument("--bad-pose", type=pathlib.Path, default=None)
    ap.add_argument("--mode", choices=("sample", "full", "names", "merge"),
                    default="sample")
    ap.add_argument("--shard", default="",
                    help="I/N (full mode): score every Nth record, write "
                         "full_corpus_scores_shard<I>of<N>.json; combine "
                         "with --mode merge afterwards")
    ap.add_argument("--embodiment", choices=("all", "hand", "gripper"),
                    default="all",
                    help="full mode: restrict to one embodiment (gripper records are scored first, the per-scene caps are derived from them, then the human records are scored)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hand", type=int, default=10)
    ap.add_argument("--n-grip", type=int, default=10)
    ap.add_argument("--names", default="")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--no-person-mask", action="store_true",
                    help="skip the Mask R-CNN person-occlusion check")
    ap.add_argument("--episode-trim", type=pathlib.Path, default=None,
                    help="JSON {episode: {grip_i, end_i, start, end}} with initial spans of human recordings; episodes absent from it are trimmed from the trajectory alone")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--viz-dir", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)

    scene_caps = {}
    if a.scene_caps and a.scene_caps.exists():
        scene_caps = {k: v["median"]
                      for k, v in json.loads(a.scene_caps.read_text()).items()}
    bad_pose = frozenset()
    if a.bad_pose and a.bad_pose.exists():
        bad_pose = frozenset(r["name"]
                             for r in json.loads(a.bad_pose.read_text()))
    episode_trim = {}
    if a.episode_trim and a.episode_trim.exists():
        episode_trim = json.loads(a.episode_trim.read_text())
        if "episodes_hand_trim" in episode_trim:
            episode_trim = episode_trim["episodes_hand_trim"]

    if a.mode == "merge":
        shards = sorted(a.out_dir.glob("full_corpus_scores_*shard*of*.json"))
        rows, seen = [], set()
        for s in shards:
            for r in json.loads(s.read_text()):
                if r["name"] not in seen:
                    seen.add(r["name"])
                    rows.append(r)
        rows.sort(key=lambda r: r["name"])
        (a.out_dir / "full_corpus_scores.json").write_text(
            json.dumps(rows, indent=1))
        c = collections.Counter(r["label"] for r in rows)
        e = collections.Counter(r["end"] for r in rows
                                if r["embodiment"] == "hand")
        print(f"merged {len(shards)} shard files -> {len(rows)} records")
        print("TALLY:", dict(c), "of", len(rows))
        print("END TAGS (hand):", dict(e))
        print("FILTER_DONE", flush=True)
        return 0

    hand_files = sorted(a.corpus.glob("*__hand__*.pt"))


    grip_files = sorted(set(a.corpus.glob("*iphone*__*.pt"))
                        | set(a.corpus.glob("*__gripper__*.pt")))
    if a.embodiment == "hand":
        grip_files = []
    elif a.embodiment == "gripper":
        hand_files = []
    rng = random.Random(a.seed)
    if a.mode == "names":
        files = []
        for n in (s.strip() for s in a.names.split(",") if s.strip()):
            files += sorted(a.corpus.glob(f"{n}*.pt"))[:1]
        out_name = "named_scores.json"
    elif a.mode == "sample":
        files = rng.sample(hand_files, min(a.n_hand, len(hand_files)))
        files += rng.sample(grip_files, min(a.n_grip, len(grip_files)))
        out_name = f"sample_scores_seed{a.seed}.json"
    else:
        files = hand_files + grip_files
        out_name = "full_corpus_scores.json"
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            if not 0 <= i < n:
                ap.error(f"--shard {a.shard}: need 0 <= I < N")


            files = files[i::n]
            tag = "" if a.embodiment == "all" else f"{a.embodiment}_"
            out_name = f"full_corpus_scores_{tag}shard{i}of{n}.json"
    if a.viz and a.viz_dir:
        a.viz_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        name = f.name[:-3]
        rec = torch.load(f, weights_only=False)
        rgb = np.asarray(rec["rgb"])
        K, dm, dW, dH = load_depth_K(rec, rgb)
        project_uv = lambda p, K=K, dW=dW, dH=dH: \
            project_point_to_uv(p, K, dW, dH)
        is_hand = "__hand" in name
        cache = None
        traj = np.asarray(rec["traj_m"], np.float64)
        traj = traj[0] if traj.ndim == 3 else traj
        pmask = None if a.no_person_mask else person_mask_for(rgb[0])
        trim = episode_trim.get("__".join(name.split("__")[:3])) \
            if is_hand else None
        row = score_record(name, traj, is_hand=is_hand, depth_map=dm,
                           project_uv=project_uv, cache=cache,
                           bad_pose_names=bad_pose, scene_caps=scene_caps,
                           person_mask=pmask, hand_trim=trim,
                           p=DEFAULT_PARAMS)


        row["camera"] = rec.get("camera")
        rows.append(row)
        if a.viz and a.viz_dir:
            render(rgb, traj, project_uv, row,
                   a.viz_dir / f"ff_{row['label']}_{name[:42]}.png")
        print(f"{row['label']:9s} {row['embodiment'][:4]}"
              f" occ={row['occlusion']:13s} end={row['end']:15s}"
              f" grip={row['grip_i']:2d} start={row['start_i']:2d}"
              f"(+{row['start_shift']}) ae={row['end_i']:2d}"
              f" sup={row['sup_len']:.2f}m {name[:46]}", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / out_name).write_text(json.dumps(rows, indent=1))
    c = collections.Counter(r["label"] for r in rows)
    e = collections.Counter(r["end"] for r in rows if r["embodiment"] == "hand")
    print("TALLY:", dict(c), "of", len(rows))
    print("END TAGS (hand):", dict(e))
    print("FILTER_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
