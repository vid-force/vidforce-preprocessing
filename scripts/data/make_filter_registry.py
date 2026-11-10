#!/usr/bin/env python3
"""Per-record filter verdicts from the merged filter scores.

``filter_records.py --mode merge`` writes one score row per record. This script applies the optional manual
adjudications and writes the verdict file that the conditioning and selection stages read:

  records  {record: embodiment, camera, label (PASS / AMBIGUOUS / REJECT), occlusion tag, depth-ray gap [m],
            start / end tags, the articulation span (grip_i, start_i .. end_i; waypoint indices, inclusive),
            its path length sup_len [m], override}

Adjudication file (optional): ``{"force_pass": {record: reason}, "trim": {episode: {"start_i": int, "end_i": int}}}``.
``force_pass`` sets the label of a record to PASS, ``trim`` sets the span of the human recordings of an episode; an
adjudicated record carries the kind of adjudication in its ``override`` field.

  python scripts/data/make_filter_registry.py --scores <scores>/full_corpus_scores.json --out <out>/record_filter.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

REC_FIELDS = ("embodiment", "label", "occlusion", "gap", "bad_pose",
              "start", "end", "grip_i", "end_i", "start_i", "start_shift",
              "sup_len", "override")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=pathlib.Path, required=True,
                    help="merged scores written by filter_records.py --mode merge")
    ap.add_argument("--overrides", type=pathlib.Path, default=None,
                    help="manual adjudications (optional)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)

    raw = a.scores.read_bytes()
    rows = json.loads(raw)
    ov_raw = b""
    if a.overrides and a.overrides.exists():
        ov_raw = a.overrides.read_bytes()
        ov = json.loads(ov_raw)
        for r in rows:
            key = r["name"]
            if key in ov.get("force_pass", {}):
                r["label"] = "PASS"
                r["override"] = "force_pass"
            if r["embodiment"] == "hand":
                for ep, t in ov.get("trim", {}).items():
                    if key.startswith(f"{ep}__hand"):
                        r["start_i"] = int(t["start_i"])
                        r["end_i"] = int(t.get("end_i", r["end_i"]))
                        r["start_shift"] = r["start_i"] - r["grip_i"]
                        r["override"] = "trim"
    records = {}
    for r in rows:
        records[r["name"]] = {**{k: r.get(k) for k in REC_FIELDS},
                              "camera": r.get("camera")}

    labels = collections.Counter(r["label"] for r in records.values())
    out = {
        "meta": {
            "scores_file": a.scores.name,
            "scores_sha256": hashlib.sha256(raw).hexdigest(),
            "overrides_sha256":
                hashlib.sha256(ov_raw).hexdigest() if ov_raw else None,
            "n_records": len(records),
            "labels": dict(sorted(labels.items())),
        },
        "records": dict(sorted(records.items())),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"records: {len(records)} {dict(labels)}")
    print("WROTE", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
