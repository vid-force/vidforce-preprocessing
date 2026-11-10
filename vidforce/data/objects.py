"""Object names for per-object reporting.

The Hoi! release labels the mechanism of an articulated part, not the object. An episode is named by the part that the
prompts of its recordings mention: every descriptive prompt is matched against ``RULES`` in order, and the episode takes
the majority name. Episodes without a descriptive prompt have no name from this rule.
"""
from __future__ import annotations

import collections
import re

RULES = [
    (r"toilet", "other:toilet"),
    (r"chest lid|\blid\b", "other:lid"),
    (r"cabinet.*dishwasher|microwave cabinet", "cupboard"),
    (r"dishwasher", "other:dishwasher"),
    (r"microwave", "other:microwave"),
    (r"washing machine|washer|dryer", "other:appliance"),
    (r"fridge|refrigerator|freezer", "fridge"),
    (r"\boven\b", "oven"),
    (r"\bwindow\b", "window"),
    (r"\bdrawer|pull-out|pull out", "drawer"),
    (r"cupboard|cabinet|closet|wardrobe|locker|shelf door|\bdoor\b|\bdoors\b", "cupboard"),
]

DISPLAY = {
    "drawer": "drawer",
    "cupboard": "cupboard door",
    "window": "window",
    "oven": "oven door",
    "fridge": "refrigerator door",
    "other": "other",
}


def prompt_class(p: str):
    p = (p or "").strip().lower()
    if p in ("", "open", "close"):
        return None, None
    for pat, cls in RULES:
        if re.search(pat, p):
            return cls, pat
    return "unknown:unmatched", None


def episode_object(prompts) -> str | None:
    """Name of the object of an episode from the prompts of its recordings, or None."""
    descriptive = sorted({(p or "").strip() for p in prompts
                          if (p or "").strip().lower() not in ("", "open", "close")})
    votes = collections.Counter()
    for p in descriptive:
        c, _ = prompt_class(p)
        if c:
            votes[c] += 1
    if not votes:
        return None
    return DISPLAY.get(votes.most_common(1)[0][0].split(":")[0])
