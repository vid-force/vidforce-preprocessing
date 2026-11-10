#!/usr/bin/env python3
"""Select the sequences: availability and quality gate, split, pairing.

Every recording of the window inventory is joined with the outputs of the earlier stages. A recording becomes a
sequence when it passes every condition below; the first one it fails is recorded as its stage.

  unbuilt          the release lacks an input the record builder needs (calibration, poses, annotation, landmarks)
  reject           quality filter: contact occluded or outside the image, less than half of the span in view, or an
                   implausibly long path
  ambiguous        quality filter: borderline visibility or an uncertain end of the span (not used)
  no_intrinsics    no intrinsics or no object window (a human recording without a gripper recording of the same
                   scene and camera)
  no_first_frame   no first-frame package
  not_visible      first frame: contact outside the image, less than half of the span inside the image, or the
                   object window occluded
  excluded         window of the held-out scene with corrupted tracking (not an evaluation episode)
  clean            a sequence

Split. An episode listed in ``splits/eval_episodes.json`` belongs to the evaluation set (``val``) with its partition,
the other windows of the held-out scene (corrupted tracking) are excluded, and every other episode is training data.
Pairing. A human sequence is paired with the gripper sequence of its episode, from the same camera where one exists
and from the other camera otherwise; the partner supplies its measured force as reference.

  python scripts/data/select_sequences.py --inventory <window_inventory.json> --corpus <records> \
      --filter <record_filter.json> --conditioning <conditioning.json> --first-frame <fields dir> \
      --out <sequences.json> [--check splits/eval_sequences.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vidforce.data.objects import episode_object
from vidforce.data.splits import load_eval_episodes, split_of

CAMERAS = ("iphone_1", "iphone_2")
STAGES = ("unbuilt", "reject", "ambiguous", "no_intrinsics", "no_first_frame",
          "not_visible", "excluded", "clean")
CHECK_KEYS = ("scene", "trial", "window", "episode", "embodiment", "camera",
              "split", "partition", "action", "prompt", "measured_force",
              "paired_gripper", "filter_label", "visibility", "start_i", "end_i",
              "span_length_m", "contact_uv", "object_window")


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def universe(inventory: dict) -> dict[str, dict]:
    """Every recording of the release: one per episode, embodiment and available camera."""
    records = {}
    for key, t in inventory["trials"].items():
        scene, trial = key.split("|")
        for emb in ("gripper", "hand"):
            for window in t[f"{emb}_windows"]:
                episode = f"{scene}__{trial}__{window}"
                for camera in CAMERAS:
                    if camera in t[f"{emb}_views"]:
                        records[f"{episode}__{emb}__{camera}"] = {
                            "scene": scene, "trial": trial, "window": window,
                            "episode": episode, "embodiment": emb, "camera": camera}
    return records


def built_records(corpus: pathlib.Path) -> set[str]:
    built = set()
    for manifest in sorted(corpus.glob("corpus_manifest_shard*of*.json")):
        for row in load(manifest):
            if row["status"] in ("ok", "cached"):
                built.add(row["record"])
    return built


def stage_of(name: str, split: str, built, verdicts, conditioning, index, fields) -> str:
    """The first condition a recording fails, or ``clean``."""
    if name not in built:
        return "unbuilt"
    label = (verdicts.get(name) or {}).get("label")
    if label == "REJECT":
        return "reject"
    if label == "AMBIGUOUS":
        return "ambiguous"
    if label != "PASS":
        return "unbuilt"
    c = conditioning.get(name)
    if c is None or (c.get("K_source") or "none").split("(")[0] == "none" \
            or not (c.get("spans") or {}).get("object"):
        return "no_intrinsics"
    idx = index.get(name)
    if idx is None or idx["status"] != "ok":
        return "no_first_frame"
    if not idx["record_valid"] \
            or (fields.get(name) or {}).get("object_visible_frac") is None:
        return "not_visible"
    if split not in ("train", "val"):
        return "excluded"
    return "clean"


def select(inventory, built, verdicts, conditioning, index, fields, eval_episodes):
    """``(records, sequences)``: every recording with its stage, and the clean ones with their fields."""
    records = universe(inventory)
    clean_gripper = collections.defaultdict(list)
    prompts = collections.defaultdict(list)
    for name, r in records.items():
        r["split"], r["partition"] = split_of(r["episode"], eval_episodes)
        r["stage"] = stage_of(name, r["split"], built, verdicts, conditioning,
                              index, fields)
        if r["stage"] == "clean" and r["embodiment"] == "gripper":
            clean_gripper[r["episode"]].append(name)
        if name in conditioning:
            prompts[r["episode"]].append(conditioning[name].get("prompt_text"))

    sequences = {}
    for name, r in sorted(records.items()):
        if r["stage"] != "clean":
            continue
        v, c, f = verdicts[name], conditioning[name], fields.get(name) or {}
        partner = None
        if r["embodiment"] == "hand":
            grippers = clean_gripper.get(r["episode"], [])
            same = f"{r['episode']}__gripper__{r['camera']}"
            partner = same if same in grippers else (grippers[0] if grippers else None)
        obj = episode_object(prompts[r["episode"]])
        sequences[name] = {
            "scene": r["scene"], "trial": r["trial"], "window": r["window"],
            "episode": r["episode"], "embodiment": r["embodiment"],
            "camera": r["camera"], "split": r["split"], "partition": r["partition"],
            "object": obj, "object_source": "prompt" if obj else None,
            "action": (f.get("labels") or {}).get("action"),
            "prompt": c.get("prompt_text"),
            "measured_force": bool(f.get("force_valid")),
            "paired_gripper": partner,
            "filter_label": v["label"], "visibility": v["occlusion"],
            "start_i": v["start_i"], "end_i": v["end_i"],
            "span_length_m": v.get("sup_len"),
            "contact_uv": c.get("contact_uv"),
            "object_window": c.get("bbox"),
        }
    return records, sequences


def check(sequences: dict, reference: dict) -> list[str]:
    """Differences between the recomputed evaluation sequences and a stored file."""
    mine = {n: s for n, s in sequences.items() if s["split"] == "val"}
    problems = [f"only in the stored file: {n}" for n in sorted(set(reference) - set(mine))]
    problems += [f"only recomputed: {n}" for n in sorted(set(mine) - set(reference))]
    for name in sorted(set(reference) & set(mine)):
        ref, got = reference[name], mine[name]
        keys: list[str] = [*CHECK_KEYS]
        if ref.get("object_source") == "prompt":
            keys.append("object")
        for k in keys:
            if ref.get(k) != got.get(k):
                problems.append(f"{name}: {k}: stored={ref.get(k)!r} recomputed={got.get(k)!r}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=pathlib.Path, required=True,
                    help="window inventory written by make_raw_window_inventory.py")
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="record directory of build_full_corpus.py (its build manifests are read)")
    ap.add_argument("--filter", type=pathlib.Path, required=True,
                    help="verdict file written by make_filter_registry.py")
    ap.add_argument("--conditioning", type=pathlib.Path, required=True,
                    help="conditioning file written by make_conditioning.py")
    ap.add_argument("--first-frame", type=pathlib.Path, required=True,
                    help="directory written by snapshot_first_image_fields.py")
    ap.add_argument("--eval-episodes", type=pathlib.Path,
                    default=pathlib.Path("splits/eval_episodes.json"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--check", type=pathlib.Path, default=None,
                    help="compare the recomputed evaluation sequences with this file")
    a = ap.parse_args(argv)

    index = {r["name"]: r for r in load(a.first_frame / "index_first_image.json")["rows"]}
    records, sequences = select(
        load(a.inventory), built_records(a.corpus), load(a.filter)["records"],
        load(a.conditioning)["records"], index,
        load(a.first_frame / "first_image_v1_exo_fields.json"),
        load_eval_episodes(a.eval_episodes))

    stages = collections.Counter(r["stage"] for r in records.values())
    by_split = collections.Counter(
        (s["split"], s["embodiment"]) for s in sequences.values())
    print(f"recordings: {len(records)}")
    for st in STAGES:
        print(f"  {st:15s} {stages.get(st, 0)}")
    for (split, emb), n in sorted(by_split.items()):
        print(f"sequences {split:5s} {emb:8s} {n}")

    if a.out is not None:
        meta = {"n_recordings": len(records),
                "stages": {st: stages.get(st, 0) for st in STAGES},
                "n_sequences": len(sequences),
                "by_split": {f"{s}/{e}": n for (s, e), n in sorted(by_split.items())}}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"meta": meta, "records": sequences}, indent=1))
        print("WROTE", a.out)
    if a.check is not None:
        problems = check(sequences, load(a.check)["records"])
        for p in problems[:50]:
            print("  DIFF", p)
        print("CHECK_FAILED" if problems else f"CHECK_OK: evaluation sequences match {a.check}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
