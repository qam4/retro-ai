#!/usr/bin/env bash
# Re-eval both champions at n=300, per level4_notes.md: "12 episodes is a
# TRIGGER resolution, not a measurement; it cannot separate the top snapshots
# from each other... Always re-eval a champion at 200-300 before building on it."
#
# Both v13 and v6-restick produced a champion reading rung 10.00/13 at n=12, so
# n=12 cannot tell them apart. This measures both on the same stick at n=300.
set -uo pipefail
# Repo root is three levels up from experiments/003-yeti/runs/. These scripts lived in
# debug/ until 4fde329, where ".." WAS the repo root; moving them silently broke every
# relative path below until this was fixed.
cd "$(dirname "$0")/../../.."

TRAIN=output/mo5/yeti/training
COMMON=(--episodes 300 --stochastic --profile yeti_fruit_level4 --fruits-total 1
        --level 4 --start-state output/mo5/yeti/level4/level4_start.sav
        --stall-threshold 40 --max-steps 1500)

for tag in v13 v6restick; do
  case "$tag" in
    v13)      M="$TRAIN/yeti_curriculum_l4_v13_anchors_15m/best/best_model.zip" ;;
    v6restick) M="$TRAIN/yeti_curriculum_l4_v6_anchorfix_15m/best_restick/best_model.zip" ;;
  esac
  echo "=============================================================="
  echo "RE-EVAL n=300: $tag  ($M)"
  echo "=============================================================="
  python3 scripts/mo5/yeti/eval_from_reset.py --model "$M" \
    --out "debug/l4_champ300_${tag}.json" "${COMMON[@]}"
  echo "$tag exit: $?"
  echo
done
echo "ALL RE-EVALS DONE"
