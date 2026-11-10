"""Tests of the evaluation-set files, the split rule, the object names and the selection gate."""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib

from vidforce.data.objects import episode_object, prompt_class
from vidforce.data.splits import load_eval_episodes, split_of

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"


def _load_script(name):
    path = ROOT / "scripts" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sequences():
    return json.loads((SPLITS / "eval_sequences.json").read_text())["records"]


def test_eval_set_counts():
    seqs = _sequences()
    assert len(seqs) == 99
    by = collections.Counter((s["partition"], s["embodiment"]) for s in seqs.values())
    assert by == {("unseen_scene", "gripper"): 23, ("unseen_scene", "hand"): 24,
                  ("held_out_interaction", "gripper"): 25, ("held_out_interaction", "hand"): 27}
    assert all(s["split"] == "val" and s["filter_label"] == "PASS" for s in seqs.values())
    assert all(s["measured_force"] == (s["embodiment"] == "gripper") for s in seqs.values())
    assert (SPLITS / "eval_sequences.txt").read_text().split() == sorted(seqs)


def test_names_match_fields():
    for name, s in _sequences().items():
        assert name == f"{s['episode']}__{s['embodiment']}__{s['camera']}"
        assert s["episode"] == f"{s['scene']}__{s['trial']}__{s['window']}"
        assert 0 <= s["start_i"] < s["end_i"] <= 15


def test_partitions_follow_the_held_out_episodes():
    episodes = load_eval_episodes(SPLITS / "eval_episodes.json")
    assert collections.Counter(episodes.values()) == {"unseen_scene": 22, "held_out_interaction": 28}
    assert all((ep.split("__")[0] == "livingroom_1") == (part == "unseen_scene")
               for ep, part in episodes.items())
    seqs = _sequences()
    for s in seqs.values():
        assert split_of(s["episode"], episodes) == ("val", s["partition"])
    assert len({s["episode"] for s in seqs.values()}) == 49
    spans = json.loads((SPLITS / "eval_human_spans.json").read_text())
    assert set(spans) <= set(episodes) and len(spans) == 33


def test_split_rule():
    episodes = {"livingroom_1__1-7__window_0": "unseen_scene",
                "kitchen_7__1-9__window_3": "held_out_interaction"}
    assert split_of("livingroom_1__1-7__window_0", episodes) == ("val", "unseen_scene")
    assert split_of("kitchen_7__1-9__window_3", episodes) == ("val", "held_out_interaction")
    assert split_of("livingroom_1__1-7__window_8", episodes) == ("excluded", None)
    assert split_of("kitchen_7__1-9__window_4", episodes) == ("train", None)


def test_pairing_of_the_eval_set():
    seqs = _sequences()
    paired = 0
    for s in seqs.values():
        partner = s["paired_gripper"]
        if s["embodiment"] == "gripper":
            assert partner is None
            continue
        if partner is None:
            assert not any(t["episode"] == s["episode"] and t["embodiment"] == "gripper"
                           for t in seqs.values())
            continue
        paired += 1
        p = seqs[partner]
        assert p["embodiment"] == "gripper" and p["episode"] == s["episode"]
        same = f"{s['episode']}__gripper__{s['camera']}"
        assert partner == same or same not in seqs
    assert paired == 40


def test_object_names():
    assert prompt_class("open")[0] is None
    assert episode_object(["open", "open upper drawer of cabinet next to sink"]) == "drawer"
    assert episode_object(["open white toilet lid"]) == "other"
    assert episode_object(["open silver fridge door"]) == "refrigerator door"
    assert episode_object(["close", "open"]) is None
    seqs = _sequences().values()
    assert collections.Counter(s["object"] for s in seqs) == {
        "drawer": 40, "cupboard door": 33, "oven door": 9, "refrigerator door": 9, "other": 8}
    assert collections.Counter(s["object"] for s in seqs if s["measured_force"]) == {
        "drawer": 20, "cupboard door": 18, "oven door": 3, "refrigerator door": 4, "other": 3}
    assert all(s["object_source"] in ("prompt", "visual check") for s in seqs)


def test_selection_gate_split_and_pairing():
    sel = _load_script("select_sequences")
    both = ["iphone_1", "iphone_2"]
    inventory = {"trials": {
        "kitchen_7|1-9": {"gripper_windows": ["window_3", "window_4"], "hand_windows": ["window_3", "window_4"],
                          "gripper_views": both, "hand_views": both},
        "livingroom_1|1-7": {"gripper_windows": ["window_8"], "hand_windows": [],
                             "gripper_views": ["iphone_1"], "hand_views": []}}}
    e3, e4 = "kitchen_7__1-9__window_3", "kitchen_7__1-9__window_4"
    l8 = "livingroom_1__1-7__window_8__gripper__iphone_1"
    names = sorted(sel.universe(inventory))
    assert len(names) == 9 and l8 in names

    built = set(names) - {f"{e4}__gripper__iphone_1"}
    verdicts = {n: {"label": "PASS", "occlusion": "visible", "start_i": 0, "end_i": 15, "sup_len": 0.4}
                for n in names}
    verdicts[f"{e3}__gripper__iphone_2"]["label"] = "REJECT"

    def cond(name, k="stored_depth_K"):
        prompt = "open upper drawer" if "__gripper__" in name else "open"
        return {"K_source": k, "prompt_text": prompt, "contact_uv": [320.0, 240.0],
                "bbox": {"center_uv": [320.0, 240.0], "half_extents_px": [128.0, 128.0]},
                "spans": {"object": {"box": [192, 112, 448, 368]}} if k != "none" else None}

    conditioning = {n: cond(n) for n in names}
    conditioning[f"{e4}__hand__iphone_1"] = cond(f"{e4}__hand__iphone_1", "none")
    index = {n: {"name": n, "status": "ok", "record_valid": True} for n in names}
    index[f"{e4}__hand__iphone_2"]["record_valid"] = False
    fields = {n: {"object_visible_frac": 0.8, "force_valid": "__gripper__" in n, "labels": {"action": "open"}}
              for n in names}

    records, sequences = sel.select(inventory, built, verdicts, conditioning, index, fields,
                                    {e3: "held_out_interaction"})
    stage = {n: r["stage"] for n, r in records.items()}
    assert stage[f"{e3}__gripper__iphone_2"] == "reject"
    assert stage[f"{e4}__gripper__iphone_1"] == "unbuilt"
    assert stage[f"{e4}__hand__iphone_1"] == "no_intrinsics"
    assert stage[f"{e4}__hand__iphone_2"] == "not_visible"
    assert stage[l8] == "excluded"
    assert sorted(sequences) == [f"{e3}__gripper__iphone_1", f"{e3}__hand__iphone_1",
                                 f"{e3}__hand__iphone_2", f"{e4}__gripper__iphone_2"]
    assert sequences[f"{e3}__hand__iphone_1"]["paired_gripper"] == f"{e3}__gripper__iphone_1"
    assert sequences[f"{e3}__hand__iphone_2"]["paired_gripper"] == f"{e3}__gripper__iphone_1"
    assert sequences[f"{e3}__gripper__iphone_1"]["split"] == "val"
    assert sequences[f"{e3}__gripper__iphone_1"]["partition"] == "held_out_interaction"
    assert sequences[f"{e4}__gripper__iphone_2"]["split"] == "train"
    assert sequences[f"{e3}__hand__iphone_1"]["object"] == "drawer"
    assert sequences[f"{e3}__hand__iphone_1"]["measured_force"] is False
    assert sel.check(sequences, {n: s for n, s in sequences.items() if s["split"] == "val"}) == []
    broken = {n: dict(s) for n, s in sequences.items() if s["split"] == "val"}
    broken[f"{e3}__hand__iphone_2"]["paired_gripper"] = None
    assert len(sel.check(sequences, broken)) == 1
