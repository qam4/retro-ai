#!/usr/bin/env bash
# Put ANY run's champion on the shared stick, so the numbers are comparable.
#
#   run_champion_eval.sh <run_dir>                 sweep the snapshots, then eval the winner
#   run_champion_eval.sh <run_dir> <champion.zip>  skip the sweep, eval that champion
#
# THE POINT IS THE SETTINGS, NOT THE SCRIPT. Every champion figure quoted in
# level4_notes.md came from these exact flags -- 300 episodes, stochastic, level 4,
# the level-4 start state, stall 40, max 1500. Do NOT edit them per-run: that is what
# makes v6 8.52/13, v13 8.49/13 and v16c 8.49/13 mean the same thing. Pass a different
# run instead.
#
# WHY STOCHASTIC, AND WHY 300. Deterministic eval on a fixed start state is ONE
# trajectory replayed N times (determinism verified: 0/27 mismatches in the SN3 replay
# work), so it measures nothing about robustness. And 12 episodes is a keep-best TRIGGER
# resolution, not a measurement: v13 and v6-restick both read rung 10.00/13 at n=12 and
# separated only at n=300 (63.3% vs 75.3% at rung 10). Re-eval a champion at 200-300
# before building on it.
#
# Consolidates run_v16c_eval.sh (parameterised, sweep+eval) and run_champion_reeval.sh
# (hardcoded v13/v6, eval only, and writing to the since-deleted debug/). The second
# form below covers what that one did, including champions outside `best/` -- v6's lives
# in `best_restick/` because its historic sweep was preserved when it was re-swept on
# current code.
set -uo pipefail
cd "$(dirname "$0")/../../.."

RUN="${1:?usage: run_champion_eval.sh <run_dir> [champion.zip]}"
CHAMP_IN="${2:-}"
TAG="$(basename "$RUN")"

SWEEP=(--episodes 12 --profile yeti_fruit_level4 --fruits-total 1 --level 4
       --start-state output/mo5/yeti/level4/level4_start.sav
       --stall-threshold 40 --max-steps 1500 --device cpu)
EVAL=(--episodes 300 --stochastic --profile yeti_fruit_level4 --fruits-total 1
      --level 4 --start-state output/mo5/yeti/level4/level4_start.sav
      --stall-threshold 40 --max-steps 1500)

if [ -n "$CHAMP_IN" ]; then
  CHAMP="$CHAMP_IN"
  TAG="${TAG}_$(basename "$(dirname "$CHAMP_IN")")"
  echo "skipping the sweep, evaluating: $CHAMP"
else
  n=$(ls "$RUN/snapshots"/*.zip 2>/dev/null | wc -l)
  echo "=============================================================="
  echo "STAGE 1/2: keep_best_sweep over $TAG ($n snapshots, n=12)"
  echo "=============================================================="
  python3 scripts/mo5/yeti/keep_best_sweep.py \
    --snapshots-dir "$RUN/snapshots" "${SWEEP[@]}"
  echo "sweep exit: $?"
  CHAMP="$RUN/best/best_model.zip"
fi

if [ ! -f "$CHAMP" ]; then
  echo "no champion at $CHAMP -- nothing to evaluate"
  exit 1
fi

echo
echo "=============================================================="
echo "STAGE 2/2: eval the champion at n=300, stochastic, from reset"
echo "=============================================================="
python3 scripts/mo5/yeti/eval_from_reset.py --model "$CHAMP" \
  --out "output/monitor/champ300_${TAG}.json" "${EVAL[@]}"
echo "eval exit: $?"
echo
echo "CHAMPION EVAL DONE: $TAG"
