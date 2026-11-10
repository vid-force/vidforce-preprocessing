#!/bin/bash
# Records and filter scores for the whole release on one multi-GPU machine.
#   1. build the record of every recording of the window inventory
#   2. score the gripper records and derive the per-scene path caps from them
#   3. score the human records with these caps, merge all scores
#
#   HOI_ROOT=<raw Hoi! release> OUT_ROOT=<output directory> INVENTORY=<window_inventory.json> NGPU=8 \
#       bash scripts/data/run_corpus_and_filter.sh
set -u
HOI_ROOT=${HOI_ROOT:?set HOI_ROOT (raw Hoi! release)}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT (output directory)}
INVENTORY=${INVENTORY:?set INVENTORY (window inventory JSON)}
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
CORPUS=${CORPUS:-$OUT_ROOT/records}
SCORES=${SCORES:-$OUT_ROOT/filter_scores}
HUMAN_SPANS=${HUMAN_SPANS:-$CODE/splits/eval_human_spans.json}
NGPU=${NGPU:-8}
cd "$CODE"
export PYTHONPATH=.
mkdir -p "$CORPUS" "$SCORES"

run_shards () {  # run_shards <label> <cmd...>: NGPU background shards, one per GPU
    local label=$1; shift
    local pids=() rc=0
    for g in $(seq 0 $((NGPU - 1))); do
        CUDA_VISIBLE_DEVICES=$g "$@" --shard "$g/$NGPU" \
            > "$SCORES/${label}_$g.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            rc=1
            echo "${label}_SHARD_${i}_FAILED"; tail -5 "$SCORES/${label}_$i.log"
        fi
    done
    return $rc
}

echo "=== 1: build records ==="
run_shards build python scripts/data/build_full_corpus.py \
    --root "$HOI_ROOT" --out "$CORPUS" --inventory "$INVENTORY" || { echo FAILED; exit 1; }
grep -h "DONE" "$SCORES"/build_*.log

echo "=== 2: score gripper records, per-scene caps ==="
SCORE_ARGS=(--corpus "$CORPUS" --out-dir "$SCORES" --mode full)
run_shards gripper python scripts/data/filter_records.py "${SCORE_ARGS[@]}" \
    --embodiment gripper || { echo FAILED; exit 1; }
python scripts/data/filter_records.py --corpus "$CORPUS" --out-dir "$SCORES" \
    --mode merge > "$SCORES/merge_gripper.log" 2>&1
cp "$SCORES/full_corpus_scores.json" "$SCORES/gripper_scores.json"
python scripts/data/make_scene_caps.py "$SCORES/gripper_scores.json" \
    "$SCORES/scene_caps.json" || { echo FAILED; exit 1; }

echo "=== 3: score human records, merge ==="
run_shards hand python scripts/data/filter_records.py "${SCORE_ARGS[@]}" \
    --embodiment hand --scene-caps "$SCORES/scene_caps.json" \
    --episode-trim "$HUMAN_SPANS" || { echo FAILED; exit 1; }
python scripts/data/filter_records.py --corpus "$CORPUS" --out-dir "$SCORES" --mode merge
echo ALL_DONE
