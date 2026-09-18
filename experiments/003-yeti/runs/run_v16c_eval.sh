#!/usr/bin/env bash
# Put v16c on the SAME stick as the v6/v13 champions, so "are we back to earlier
# levels?" can be answered with one number instead of two incompatible ones.
#
# What was incompatible: v6/v13 are reported as mean rung 8.52 / 8.49 of 13 with
# >= rung 10 in 226/300 episodes -- a CHAMPION snapshot (best of 150) evaluated over 300
# from-reset episodes. v16c has only ever been reported as reset_reach, a running EMA
# during training. Champion-vs-EMA flatters the champion, so the gap was unmeasurable.
#
# Two stages, matching debug/run_v13_sweeps.sh and debug/run_champion_reeval.sh exactly
# (their COMMON arrays are copied verbatim -- do not "tidy" them, the comparison is the
# whole point):
#   1. keep_best_sweep over v16c's snapshots at 12 episodes -> picks a champion.
#   2. eval_from_reset at 300 episodes, stochastic, on that champion.
#
# ONE UNFAIRNESS, IN V16C'S DISFAVOUR, LEFT DELIBERATELY: v16c is a 6M run and has 60
# snapshots against v6/v13's 150, so its champion is the best of 60 rather than the best
# of 150. A weaker selection is the honest comparison for a shorter run; inflating it by
# snapshotting more often would not make the policy better.
set -uo pipefail
# Repo root is three levels up from experiments/003-yeti/runs/. These scripts lived in
# debug/ until 4fde329, where ".." WAS the repo root; moving them silently broke every
# relative path below until this was fixed.
cd "$(dirname "$0")/../../.."

TRAIN=output/mo5/yeti/training
# Takes a run directory so the SAME stick can be applied to any run; defaults to the 6M
# arm it was written for. The whole point is that these numbers are comparable, so the
# eval settings below must not be edited per-run -- pass a different run instead.
RUN="${1:-$TRAIN/yeti_curriculum_l4_v16c_payonchange_cold_6m}"
TAG="$(basename "$RUN")"

SWEEP=(--episodes 12 --profile yeti_fruit_level4 --fruits-total 1 --level 4
       --start-state output/mo5/yeti/level4/level4_start.sav
       --stall-threshold 40 --max-steps 1500 --device cpu)
EVAL=(--episodes 300 --stochastic --profile yeti_fruit_level4 --fruits-total 1
      --level 4 --start-state output/mo5/yeti/level4/level4_start.sav
      --stall-threshold 40 --max-steps 1500)

echo "=============================================================="
echo "STAGE 1/2: keep_best_sweep over $TAG ($(ls "$RUN/snapshots"/*.zip 2>/dev/null | wc -l) snapshots, n=12)"
echo "=============================================================="
python3 scripts/mo5/yeti/keep_best_sweep.py \
  --snapshots-dir "$RUN/snapshots" "${SWEEP[@]}"
echo "sweep exit: $?"

CHAMP="$RUN/best/best_model.zip"
if [ ! -f "$CHAMP" ]; then
  echo "no champion at $CHAMP -- cannot re-eval"
  exit 1
fi

echo
echo "=============================================================="
echo "STAGE 2/2: re-eval the champion at n=300 (same stick as v6/v13)"
echo "=============================================================="
python3 scripts/mo5/yeti/eval_from_reset.py --model "$CHAMP" \
  --out "output/monitor/champ300_${TAG}.json" "${EVAL[@]}"
echo "eval exit: $?"
echo
echo "V16C EVAL DONE"
