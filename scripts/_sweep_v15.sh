#!/bin/bash
set -e
D=output/mo5/yeti/training/yeti_curriculum_v15_phase2
OUT=output/mo5/yeti/eval/sweep_v15
mkdir -p "$OUT"
for M in "$D/snapshots/model_3000000_steps.zip" \
         "$D/snapshots/model_4000000_steps.zip" \
         "$D/snapshots/model_4500000_steps.zip" \
         "$D/snapshots/model_4750000_steps.zip" \
         "$D/final_model.zip"; do
  echo "===== $M ====="
  env RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux python3 \
    scripts/eval_from_reset.py --model "$M" \
    --episodes 300 --stochastic --out "$OUT/$(basename $M).json" 2>&1 | tail -3
done
echo "===== SWEEP DONE ====="
