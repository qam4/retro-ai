#!/usr/bin/env bash
# Two-arm A/B on ONE lever: reward.params.mark_airborne.
#
#   v16a  mark_airborne: false  -- control, marking after the airborne return (old)
#   v16b  mark_airborne: true   -- the fix, a jump landing can be marked
#
# Cold, 6M, seed 42, everything else identical. Run SEQUENTIALLY: 8 cores and each arm
# uses num_envs 8, so running them together would halve each one's throughput and make
# the wall-clock numbers meaningless.
set -uo pipefail
cd "$(dirname "$0")/.."

for arm in v16a_markground_cold_6m v16b_markair_cold_6m; do
  echo "=============================================================="
  echo "ARM: $arm   ($(date -u +%H:%M:%S) UTC)"
  echo "=============================================================="
  rm -rf "output/mo5/yeti/training/yeti_curriculum_l4_${arm}"
  python3 -u scripts/mo5/yeti/train_checkpoint_curriculum.py \
    --config "experiments/003-yeti/configs/yeti_curriculum_l4_${arm}.yaml"
  echo "$arm exit: $?"
  echo
done
echo "BOTH ARMS DONE"
