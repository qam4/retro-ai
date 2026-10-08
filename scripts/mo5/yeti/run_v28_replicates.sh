#!/usr/bin/env bash
# Three replicates of v24's config on clean HEAD, then score each with the referee.
#
# Sequential ON PURPOSE. Three concurrent runs would contend for CPU with 8 subprocess
# envs each, changing throughput and therefore the very run-to-run variation being
# measured. ~2.9h training + ~11min referee per replicate, so ~9.5h total.
#
# The referee runs POST HOC (no --watch). v26's --watch referee died 10s in on a
# not-yet-created snapshots dir and that run trained unrefereed; a second attempt
# stamped its idle timer once per batch and dropped 24 of 60 snapshots. Both bugs are
# fixed in f16c85a, but scoring after the fact cannot hit either and costs nothing.
#
# Deliberately NOT `set -e`: if one replicate dies the other two still run, and the
# exit codes are in the log.
set -u

# The repo root, from this script's own location, not a hardcoded home directory.
cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=python:build/ci-linux
export RETRO_AI_ROM_DIR=roms

echo "HEAD $(git rev-parse --short HEAD)   dirty: $(git status --porcelain | wc -l) path(s)"
git status --porcelain
echo

for r in a b c; do
  CFG=experiments/003-yeti/configs/yeti_curriculum_l4_v28${r}_v24repeat_6m.yaml
  OUT=output/mo5/yeti/training/yeti_curriculum_l4_v28${r}_v24repeat_6m

  echo "========== REPLICATE ${r} TRAIN START $(date -Is)"
  python3 -u scripts/mo5/yeti/train_checkpoint_curriculum.py --config "$CFG"
  echo "========== REPLICATE ${r} TRAIN EXIT $? $(date -Is)"

  echo "========== REPLICATE ${r} REFEREE START $(date -Is)"
  python3 -u scripts/mo5/yeti/keep_best_sweep.py \
    --snapshots-dir "${OUT}/snapshots" \
    --episodes 30 \
    --device cpu \
    --level 4 \
    --profile yeti_fruit_level4 \
    --fruits-total 1 \
    --start-state output/mo5/yeti/level4/level4_start.sav \
    --stall-threshold 40 \
    --max-steps 1500
  echo "========== REPLICATE ${r} REFEREE EXIT $? $(date -Is)"
done

echo "========== ALL THREE DONE $(date -Is)"
