#!/bin/bash
set -e
D=output/mo5/yeti/training/yeti_curriculum_v14_aggscore/snapshots
OUT=output/mo5/yeti/eval/sweep_v14
mkdir -p "$OUT"
for s in 5000000 8000000 11000000 13000000 15000000 17000000 19000000 20000000; do
  echo "===== snapshot $s ====="
  env RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux python3 \
    scripts/eval_from_reset.py --model "$D/model_${s}_steps.zip" \
    --episodes 300 --stochastic --out "$OUT/snap_${s}.json" 2>&1 | tail -13
done
echo "===== SWEEP DONE ====="
