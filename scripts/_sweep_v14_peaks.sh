#!/bin/bash
set -e
D=output/mo5/yeti/training/yeti_curriculum_v14_aggscore/snapshots
OUT=output/mo5/yeti/eval/sweep_v14_peaks
mkdir -p "$OUT"
# Snapshots where the training diag showed princess-from-reset peaks
# (the round-million sweep landed between these and saw 0%).
for s in 4000000 6250000 12750000 18750000 19250000 19750000; do
  echo "===== snapshot $s ====="
  env RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux python3 \
    scripts/eval_from_reset.py --model "$D/model_${s}_steps.zip" \
    --episodes 300 --stochastic --out "$OUT/snap_${s}.json" 2>&1 | tail -13
done
echo "===== SWEEP DONE ====="
