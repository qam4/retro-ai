#!/usr/bin/env bash
# v13's stated success criterion (config header + commit e080485): judge it as a
# snapshot DISTRIBUTION via keep_best_sweep against v6's n=150 sweep
# (mean 6.37, median 7.88, max 10.00, >=9.5 in 8/150, <1.0 in 10/150).
#
# Two sweeps, sequentially (8 cores; each eval is its own subprocess):
#   1. v13's 150 snapshots.
#   2. v6's 150 snapshots RE-SWEPT on current code, because dc8e5a0 changed the
#      Rope1/Spring/Step anchors and therefore the route-depth stick. v6's
#      published row was measured with the old anchors, so it is not
#      apples-to-apples with v13. Written to best_restick/ so v6's historic
#      sweep_state.json is preserved.
#
# Eval settings match those recorded in v6's _eval.json: 12 episodes, 1 fruit,
# level 4, the level-4 start state. stall/max-steps follow v13's training env.
set -uo pipefail
# Repo root is three levels up from experiments/003-yeti/runs/. These scripts lived in
# debug/ until 4fde329, where ".." WAS the repo root; moving them silently broke every
# relative path below until this was fixed.
cd "$(dirname "$0")/../../.."

TRAIN=output/mo5/yeti/training
COMMON=(--episodes 12 --profile yeti_fruit_level4 --fruits-total 1 --level 4
        --start-state output/mo5/yeti/level4/level4_start.sav
        --stall-threshold 40 --max-steps 1500 --device cpu)

echo "=============================================================="
echo "SWEEP 1/2: v13 (the run under test)"
echo "=============================================================="
python3 scripts/mo5/yeti/keep_best_sweep.py \
  --snapshots-dir "$TRAIN/yeti_curriculum_l4_v13_anchors_15m/snapshots" \
  "${COMMON[@]}"
echo "sweep 1 exit: $?"

echo
echo "=============================================================="
echo "SWEEP 2/2: v6 re-swept on CURRENT code (same-stick control)"
echo "=============================================================="
python3 scripts/mo5/yeti/keep_best_sweep.py \
  --snapshots-dir "$TRAIN/yeti_curriculum_l4_v6_anchorfix_15m/snapshots" \
  --best-dir "$TRAIN/yeti_curriculum_l4_v6_anchorfix_15m/best_restick" \
  "${COMMON[@]}"
echo "sweep 2 exit: $?"

echo
echo "ALL SWEEPS DONE"
