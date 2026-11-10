# VidForce — preprocessing pipeline and evaluation-set identifiers (anonymous review copy)

Supplementary code for the ICLR 2027 submission *VidForce: Predicting Force-Aware Articulation by Observing Humans*.
It contains

- the preprocessing code that turns the public Hoi! release into the sequences used in the paper (Appendix B): the
  record of every recording (frames, ground-truth contact trajectory, measured force, depth and intrinsics), the
  quality filter, the first-frame contact and object window, the first-frame checks, and the selection, split and
  pairing of the sequences;
- the identifiers of the paper's evaluation set (Eval): its 99 sequences with their partition, pairing and
  first-frame inputs.

The model, the training and evaluation code, the extraction of the model inputs (including the re-framed windows of
Eval′) and the checkpoints are not part of this copy; they are released upon acceptance.

## Contents

| path | contents |
|---|---|
| `scripts/data/` | the pipeline stages; order and commands in `docs/PIPELINE.md` |
| `vidforce/data/hoi/` | readers of the Hoi! release: calibration, pose, wrench and trigger streams, confirmed interaction windows, object annotations |
| `vidforce/data/` | record builders (`corpus_build.py` gripper, `hand_records.py` human), force targets (`force_stats.py`), quality filter and span trimming (`filtering.py`), contact and object window (`conditioning.py`, `crops.py`), first-frame checks (`visibility.py`), split and object names (`splits.py`, `objects.py`) |
| `vidforce/perception/depth.py` | wrapper of Depth Anything 3 (depth and intrinsics) |
| `splits/` | the evaluation set (below) |
| `tests/` | CPU tests of the filter, the record containers and the selection |

## The evaluation set (`splits/`)

A *sequence* is one camera view of one interaction in one embodiment, named
`{scene}__{session}__{window}__{embodiment}__{camera}` with embodiment `gripper | hand` and camera
`iphone_1 | iphone_2`. An *episode* is `{scene}__{session}__{window}`; its recordings from both cameras show the same
interaction, and its gripper recordings share one force measurement.

| file | contents |
|---|---|
| `eval_sequences.json` | the 99 Eval sequences with the fields below |
| `eval_sequences.txt` | their names, one per line |
| `eval_episodes.json` | the 50 held-out episodes: 22 of the unseen living-room scene (`unseen_scene`) and 28 from ten other scenes (`held_out_interaction`). Splits are fixed per episode before filtering; 49 of the 50 episodes contribute at least one sequence |
| `eval_human_spans.json` | initial spans of the human recordings of 33 held-out episodes, an input of the filter (`docs/PIPELINE.md`) |

Counts: 99 sequences = 48 gripper (measured force) + 51 human; 47 from the unseen scene (23 gripper, 24 human) and 52
held-out interactions (25 gripper, 27 human); 40 human sequences have a paired gripper sequence in Eval.

The training split consists of the sequences of all other episodes that pass the same pipeline (630 sequences in total,
531 for training); the other windows of the held-out scene, whose tracking is corrupted, are excluded. This copy lists
the evaluation set only; the list of training sequences is part of the release upon acceptance.

Fields of a sequence in `eval_sequences.json`:

| field | meaning |
|---|---|
| `scene`, `trial`, `window`, `episode`, `embodiment`, `camera` | identity (`trial` is the session) |
| `split`, `partition` | `val`; `unseen_scene` or `held_out_interaction` |
| `object`, `object_source` | object name used for per-object reporting: the part that the prompts of the episode mention (`prompt`), or a visual check where no prompt names it (`visual check`) |
| `action`, `prompt` | `open` / `close`; the dataset prompt of the recording |
| `measured_force` | true for gripper sequences |
| `paired_gripper` | for a human sequence, the gripper sequence of the same episode whose measured force serves as its reference (same camera where one exists) |
| `filter_label`, `visibility` | verdict of the quality filter and its visibility tag |
| `start_i`, `end_i`, `span_length_m` | the articulation span in waypoint indices (16 waypoints, inclusive) and its path length |
| `contact_uv` | first-frame contact in pixels of the 640-pixel frame |
| `object_window` | centre and half extents (pixels) of the object window |

## Running

`docs/PIPELINE.md` lists the stages with their commands and describes the filter and the selection rules. The raw Hoi!
release (zip archives and its metadata files) is obtained from the dataset's page. Stages that estimate depth or
person masks need a GPU; Depth Anything 3 must be importable (`requirements.txt`).

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests -q
```

The last stage compares its recomputed evaluation sequences with the stored file:

```bash
python scripts/data/select_sequences.py ... --check splits/eval_sequences.json
```
