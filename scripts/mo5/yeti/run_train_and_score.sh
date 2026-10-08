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

# The repo root, from this script's own location (scripts/mo5/yeti/), not a hardcoded
# home directory: the repo is public and that path carried the user's name.
cd "$(dirname "$0")/../../.." || exit 1
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
  # EVERYTHING the referee needs comes from the config, never from this script, so the
  # referee scores the run on the level, view and reach test it trained with. This used
  # to hardcode L4 (`--level 4 --profile yeti_fruit_level4 --fruits-total 1` and L4's
  # start state), which would have scored any other level's run as L4 without an error.
  # Values are shlex-quoted before eval, so a path with spaces cannot split.
  VARS=$(python3 - "$CFG" <<'PYEOF'
import shlex
import sys

from retro_ai.training.run_config import RunConfig

c = RunConfig.from_yaml(sys.argv[1])
if c.curriculum is None:
    raise SystemExit("config has no curriculum section")
vals = {
    "OUT": c.training.output,
    "RESIZE_MODE": c.env.resize_mode,
    "REACH_MODE": c.curriculum.waypoint_reach_mode,
    "PROFILE": c.env.profile,
    "LEVEL": int((c.reward.params or {}).get("level", 1)),
    "FRUITS_TOTAL": c.curriculum.fruits_total,
    "START_STATE": c.curriculum.start_state or "",
    "STALL": c.env.stall_threshold,
    "MAXSTEPS": c.env.max_steps,
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PYEOF
)
  if [ -z "$VARS" ]; then
    echo "========== COULD NOT READ the referee settings FROM $CFG -- skipped"
    continue
  fi
  eval "$VARS"
  REF_ARGS=(--level "$LEVEL" --profile "$PROFILE" --fruits-total "$FRUITS_TOTAL"
            --stall-threshold "$STALL" --max-steps "$MAXSTEPS")
  # L1 starts from a game reset, not a save-state.
  if [ -n "$START_STATE" ]; then
    REF_ARGS+=(--start-state "$START_STATE")
  fi

  echo "========== $CFG TRAIN START $(date -Is)   -> $OUT"
  python3 -u scripts/mo5/yeti/train_checkpoint_curriculum.py --config "$CFG"
  echo "========== $CFG TRAIN EXIT $? $(date -Is)"

  echo "========== $CFG REFEREE START $(date -Is)   resize_mode=$RESIZE_MODE reach_mode=$REACH_MODE ${REF_ARGS[*]}"
  python3 -u scripts/mo5/yeti/keep_best_sweep.py \
    --snapshots-dir "${OUT}/snapshots" \
    --resize-mode "$RESIZE_MODE" \
    --reach-mode "$REACH_MODE" \
    --episodes 30 \
    --device cpu \
    "${REF_ARGS[@]}"
  echo "========== $CFG REFEREE EXIT $? $(date -Is)"
done

echo "========== ALL DONE $(date -Is)"
