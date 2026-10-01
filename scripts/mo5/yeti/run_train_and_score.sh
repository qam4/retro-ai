#!/usr/bin/env bash
# Train one config to completion, then score every snapshot with the referee.
#
# Usage: run_train_and_score.sh <config.yaml> [<config.yaml> ...]
#
# Replaces the copy-pasted body of run_v28_replicates.sh. The output directory is read
# FROM THE CONFIG rather than passed in, so the two can never disagree.
#
# The referee runs POST HOC (no --watch). A --watch referee died 10s into v26 on a
# not-yet-created snapshots dir and that run trained unrefereed; a second attempt stamped
# its idle timer once per batch and dropped 24 of 60 snapshots. Both bugs are fixed in
# f16c85a, but scoring after the fact cannot hit either and costs nothing.
#
# Deliberately NOT `set -e`: if one config dies the rest still run, and the exit codes
# are in the log.
set -u

cd /home/ec2-user/src/fred/retro-ai || exit 1
export PYTHONPATH=python:build/ci-linux
export RETRO_AI_ROM_DIR=roms

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <config.yaml> [<config.yaml> ...]" >&2
  exit 2
fi

echo "HEAD $(git rev-parse --short HEAD)   dirty: $(git status --porcelain | wc -l) path(s)"
git status --porcelain
echo

for CFG in "$@"; do
  if [ ! -f "$CFG" ]; then
    echo "========== MISSING CONFIG $CFG -- skipped"
    continue
  fi
  OUT=$(python3 -c "
from retro_ai.training.run_config import RunConfig
print(RunConfig.from_yaml('$CFG').training.output)
")
  if [ -z "$OUT" ]; then
    echo "========== COULD NOT READ output FROM $CFG -- skipped"
    continue
  fi

  echo "========== $CFG TRAIN START $(date -Is)   -> $OUT"
  python3 -u scripts/mo5/yeti/train_checkpoint_curriculum.py --config "$CFG"
  echo "========== $CFG TRAIN EXIT $? $(date -Is)"

  echo "========== $CFG REFEREE START $(date -Is)"
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
  echo "========== $CFG REFEREE EXIT $? $(date -Is)"
done

echo "========== ALL DONE $(date -Is)"
