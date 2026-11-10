# Pipeline: Hoi! release → sequences

Run from the repository root with `PYTHONPATH=.`. `$HOI_ROOT` is the raw Hoi! release, `$OUT` an output directory.
Record naming: `{scene}__{session}__{window}__{embodiment}__{camera}` (embodiment `gripper | hand`, camera
`iphone_1 | iphone_2`); an episode is `{scene}__{session}__{window}`.

## Stages

| # | stage | command | output |
|---|---|---|---|
| 1 | window inventory | `python scripts/data/make_raw_window_inventory.py --root $HOI_ROOT --out $OUT/window_inventory.json --date <YYYY-MM-DD>` | every confirmed interaction window of the release and the cameras that recorded it |
| 2 | records | `python scripts/data/build_full_corpus.py --root $HOI_ROOT --out $OUT/records --inventory $OUT/window_inventory.json [--shard i/n]` | one record per recording (16 frames) and a build manifest per shard; a recording that lacks an input is listed there with the reason |
| 3 | filter scores | `python scripts/data/filter_records.py --corpus $OUT/records --out-dir $OUT/filter_scores --mode full --embodiment gripper --shard i/n`, then `--mode merge`; `python scripts/data/make_scene_caps.py <merged gripper scores> $OUT/filter_scores/scene_caps.json`; then `--mode full --embodiment hand --shard i/n --scene-caps $OUT/filter_scores/scene_caps.json --episode-trim splits/eval_human_spans.json` and `--mode merge` | one score row per record (`full_corpus_scores.json`) |
| 4 | verdicts | `python scripts/data/make_filter_registry.py --scores $OUT/filter_scores/full_corpus_scores.json --out $OUT/record_filter.json` | label, visibility tag and articulation span per record |
| 5 | conditioning | `python scripts/data/make_conditioning.py --corpus $OUT/records --filter $OUT/record_filter.json --out $OUT/conditioning.json` | intrinsics, first-frame contact, object and scene window per record |
| 6 | first frame | `python scripts/data/build_first_image_corpus.py --sidecar $OUT/conditioning.json --corpus $OUT/records --out-dir $OUT/first_frame [--shard i --num-shards n]` (then `--manifest`, `--validate N`), `python scripts/data/snapshot_first_image_fields.py --pkg-dir $OUT/first_frame --out-dir $OUT/first_frame_fields` | first-frame package per record (contact frame, windows, force targets, first-frame checks) and its scalar fields |
| 7 | selection | `python scripts/data/select_sequences.py --inventory $OUT/window_inventory.json --corpus $OUT/records --filter $OUT/record_filter.json --conditioning $OUT/conditioning.json --first-frame $OUT/first_frame_fields --out $OUT/sequences.json --check splits/eval_sequences.json` | the sequences with split, partition and pairing; `--check` compares the evaluation sequences with the stored file |
| 8 | final records | `python scripts/data/build_full_corpus.py --root $HOI_ROOT --out $OUT/records_32f --inventory $OUT/window_inventory.json --names-file <sequence names> --num-frames 32 --depth-source da3 --depth-frames all` | the records of the sequences with 32 frames and Depth Anything 3 depth and intrinsics for every frame |

`scripts/data/run_corpus_and_filter.sh` runs stages 2 and 3 in this order on one multi-GPU machine. For the evaluation
set, `<sequence names>` in stage 8 is `splits/eval_sequences.txt`.

## Records (stage 2)

A record holds the frames of one recording from one third-person camera, sampled uniformly over the interaction, the
contact trajectory in the camera frame at 16 waypoints with their times, and the camera pose.

- Gripper (`vidforce/data/corpus_build.py`): the trajectory is the tool point of the instrumented gripper, from the
  pose of the tracking device mounted on it and the calibrated device-to-tool transform. The record stores the
  compensated and filtered force of the force–torque sensor at the waypoints and as a dense stream, the contact flag
  of the trigger, the peak force over the contact samples, the articulation axis, and Depth Anything 3 depth with
  intrinsics.
- Human (`vidforce/data/hand_records.py`): the trajectory is the thumb tip of the manipulating hand from the hand
  tracking of the head-worn device, mapped into the camera frame. Human recordings carry no force.

Force targets (`vidforce/data/force_stats.py`, computed in stage 6): the peak is the maximum force magnitude over the
contact samples; the 32-sample profile is the Savitzky–Golay-smoothed magnitude of the contact phase on a uniform grid
over the span, rescaled so that its maximum equals the peak. Torque targets use the same span and the same reduction,
applied to the magnitude of the torque logged in the sensor frame.

## Quality filter (stage 3, `vidforce/data/filtering.py`)

The filter keeps the recordings in which the interaction can be seen. Its central test is the depth-ray gap of a
trajectory point: the depth of the point minus the depth image at its projection. Up to 0.20 m the point is visible, up
to 0.45 m borderline, beyond that occluded.

Gripper records. The gap is the median over the first four waypoints. A record is rejected when the contact is
occluded or outside the image, or when at least three waypoints fall on a person (Mask R-CNN mask of the first frame)
with a median gap above 0.20 m. A borderline gap or missing depth makes it ambiguous. The span is the whole window, as
delimited by the release.

Human records. The span is trimmed to the part in which the object moves.
- Start: the grip onset, where the hand slows down after its approach (`grip_index`), moved past a residual approach
  that ends in a decelerating dip (`refine_contact_start`).
- End: the earliest of the kinematic end (the hand settles or reverses), the stop of the object, and a per-scene cap on
  the path length (the median path length of the passing gripper records of the scene, stage 3), cut earlier when the
  hand rests and then walks away (`refine_release_end`).
- Visibility: the first waypoint of the span with a visible gap becomes the start of the supervised span. A record is
  rejected when no waypoint is visible or in the image, when the start lies inside the person mask, or when the visible
  part covers less than half of the path of the span. It is ambiguous when the gap is only borderline, the end of the
  span is uncertain, or fewer than four waypoints remain.

Both. A span longer than 2.5 m is rejected as corrupted tracking.

Initial human spans. The stop of the object is read from tracked scene points near the hand. These tracks come from
the extraction of the model inputs, which is not part of this copy, so the filter here receives no tracks and that cue
is inactive. For 33 held-out episodes, the grip onset and end from the first pass, which used the tracks, are stored in
`splits/eval_human_spans.json`; `--episode-trim` starts from them, and the refinements and the visibility test above
are applied on top. The spans of the evaluation sequences as used in the paper are the `start_i` and `end_i` fields of
`splits/eval_sequences.json`.

## Conditioning and first frame (stages 5 and 6)

Intrinsics at image resolution come from Depth Anything 3 on the first frame of the gripper record; a human record
takes the median intrinsics of the gripper records of the same scene and camera and is not used when there are none.
The first frame is the contact frame: the first trigger contact of a gripper record and the start of the span of a
human record. The contact pixel is the projection of the first point of the span. The object window is a square crop
around the contact with half side 1.3 times the largest extent of the projected span from the contact, at least 0.4
times the longer image side in total, clamped to the image; the scene window is 1.5 times wider
(`vidforce/data/conditioning.py`, `vidforce/data/crops.py`).

First-frame checks (`vidforce/data/visibility.py`): the contact lies inside the image; at least half of the waypoints
of the span project inside the image; and, of the depth pixels of the object window that are not more than 0.35 m
behind the contact, at least 30 % lie within 0.35 m of the contact depth (the others are occluders in front of it). A
window without enough valid depth fails.

## Selection (stage 7, `scripts/data/select_sequences.py`)

A recording becomes a sequence when it was built, has the filter label PASS, has intrinsics and an object window,
passes the first-frame checks, and belongs to the training or the evaluation split. Of the 1,434 recordings of the
release, 630 remain: 531 training and 99 evaluation sequences (Appendix B of the paper gives the numbers per step).

Split. Splits are fixed per episode before filtering. `splits/eval_episodes.json` lists the held-out episodes; the
other windows of the held-out scene, whose tracking is corrupted, are excluded; all remaining episodes are training
data. Filtering removes recordings but never moves an episode between splits.

Pairing. A human sequence is paired with the gripper sequence of its episode, from the same camera where one exists
and otherwise from the other camera. The peak and normalized profile of the partner are its force reference.

## Reproducibility

Stages 2, 3 and 8 run Depth Anything 3 and stage 3 a person detector; their outputs depend on the hardware and library
versions in the last digits. `splits/eval_sequences.json` stores the outcome used in the paper.
