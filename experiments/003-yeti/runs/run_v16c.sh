#!/usr/bin/env bash
# v16c: the RECONCILIATION arm for the v16a/v16b A/B.
#
#   v16a  mark_airborne false                            -> Spring 0.61, Step 0.15
#   v16b  mark_airborne true                             -> Spring 0.04, Step 0.01
#   v16c  mark_airborne true + pay_on_target_change true  -> this run
#
# v16b's collapse was traced offline to the landing payment being deleted: marking
# mid-air changes the waypoint list on a frame the reward returns early from, so the
# change is first seen on the LANDING frame, and the list-change guard skips exactly
# that frame -- the one the D2 freeze has banked the whole 61-step trampoline arc into.
# Replaying one trajectory through both variants: floor-9 arrival +3.200 -> 0.000, 29/29
# episodes. With pay_on_target_change it is +3.920, and only 3 frames of 359 move.
#
# The question here is whether that repairs the TRAINING collapse, not just the frame.
# Read index 8 (Spring) and index 9 (Step) of reset_reach against v16a's 0.61 / 0.15.
#
# GATED SMOKE. The reward change was already pinned frame-by-frame offline (tests
# test_reward_airborne_marking.py, and debug/l4_spring_trace.py over 40 episodes), so the
# smoke is not here to check reward semantics -- it is here to catch a config or plumbing
# error before spending ~3h. A cold run at 100k is only at route[4]: 2-3/30 (measured on
# both v16 arms), so the gate is deliberately weak: it fails a run that produces nothing.
set -uo pipefail
# Repo root is three levels up from experiments/003-yeti/runs/. These scripts lived in
# debug/ until 4fde329, where ".." WAS the repo root; moving them silently broke every
# relative path below until this was fixed.
cd "$(dirname "$0")/../../.."

CFG=experiments/003-yeti/configs/yeti_curriculum_l4_v16c_payonchange_cold_6m.yaml
OUT=output/mo5/yeti/training/yeti_curriculum_l4_v16c_payonchange_cold_6m

echo "=============================================================="
echo "SMOKE 100k (gate: route points >= 2, v16a had 3 / v16b had 2)"
echo "=============================================================="
python3 -u scripts/mo5/yeti/smoke_train.py --config "$CFG" \
  --timesteps 100000 --min-chain 2
rc=$?
if [ $rc -ne 0 ]; then
  echo "SMOKE FAILED (exit $rc) -- NOT starting the 6M run"
  exit $rc
fi

echo
echo "=============================================================="
echo "FULL RUN 6M   ($(date -u +%H:%M:%S) UTC)   expect ~2h50m"
echo "=============================================================="
# DO NOT clear $OUT here. kiro-monitor writes status.json and output.log into it (per
# the training-runs steering), so an `rm -rf "$OUT"` in this script deletes the monitor's
# own files mid-run -- it would have fired the moment the smoke above passed. The caller
# clears the directory BEFORE launching the monitor instead.
python3 -u scripts/mo5/yeti/train_checkpoint_curriculum.py --config "$CFG"
echo "v16c exit: $?"
