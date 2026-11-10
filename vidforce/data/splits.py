"""Evaluation split.

Splits are fixed per episode (scene, session, interaction window) before any filtering. ``splits/eval_episodes.json``
lists the held-out episodes in two partitions: ``unseen_scene`` (the living-room scene, which is held out as a whole)
and ``held_out_interaction`` (a stratified set of episodes from ten other scenes). Every other episode belongs to the
training split, except the remaining windows of the held-out scene (corrupted tracking), which are excluded.
"""
from __future__ import annotations

import json
import pathlib

HELD_OUT_SCENE = "livingroom_1"
PARTITIONS = ("unseen_scene", "held_out_interaction")


def load_eval_episodes(path) -> dict[str, str]:
    """``{episode: partition}`` of the held-out episodes."""
    blob = json.loads(pathlib.Path(path).read_text())
    out: dict[str, str] = {}
    for partition in PARTITIONS:
        for episode in blob[partition]:
            if episode in out:
                raise ValueError(f"episode listed twice: {episode}")
            out[episode] = partition
    return out


def split_of(episode: str, eval_episodes: dict[str, str]) -> tuple[str, str | None]:
    """``(split, partition)`` of an episode; split is ``val``, ``train`` or ``excluded``."""
    if episode in eval_episodes:
        return "val", eval_episodes[episode]
    if episode.split("__")[0] == HELD_OUT_SCENE:
        return "excluded", None
    return "train", None
