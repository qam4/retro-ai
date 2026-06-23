#!/bin/bash
set -e
D=output/mo5/yeti/training/yeti_curriculum_v13_long/snapshots
OUT=output/mo5/yeti/eval/sweep_v13
mkdir -p "$OUT"
for s in 5750000 6000000 6250000 10000000 15000000; do
  echo "===== snapshot $s ====="
  env RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux python3 \
    scripts/eval_from_reset.py --model "$D/model_${s}_steps.zip" \
    --episodes 300 --stochastic --out "$OUT/snap_${s}.json" 2>&1 | tail -13
done
echo "===== SWEEP DONE ====="
