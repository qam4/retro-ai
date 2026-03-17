# Experiment 002: Validation Log

**Companion to:** `002-systematic-agent-investigation.md`
**Purpose:** Record raw results for each phase as they complete.

## Phase 0: Random Baseline

**Command:**
```bash
RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \
  python3.9 scripts/exp002_runner.py phase0
```

**Results:**
- Status: DONE ✅
- Mean reward: 10.0 ± 8.3
- Best reward: 26.0
- Mean episode length: 232
- Episodes: 10
- Note: RAM score starts at 3 after startup sequence (480 warmup frames). Reward delta is correct.

## Phase 1: Vanilla PPO (No Extras)

**Command:**
```bash
RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \
  python3.9 scripts/exp002_runner.py phase1
```

**Config:** PPO, lr=3e-4, batch_size=64, joystick [3,3,2], memory reward, 4 envs, GPU+FP16, no RND/DrQ/sticky

**Smoke test (500 steps):**
- Status: PASS ✅
- No errors, episodes completing

**Training (100k steps):**
- Rolling reward peaked ~7 at 6k steps, settled ~5.5 by 100k
- Wall clock: ~34 min

**100k final eval (3 episodes):**
- Mean reward: 4.0 ± 0.0
- Beats random baseline? NO (4.0 < 10.0)
- Video reviewed: YES — agent fires and moves, destroyed a few satellites
- Note: worse than random. This is our "bad baseline" to improve on.

## Phase 2: PPO + RND (coefficient 0.1)

**50k checkpoint:**
- Mean reward: —
- vs Phase 1: —

**100k final eval:**
- Mean reward: —
- vs Phase 1: —

## Phase 3: PPO + DrQ + Sticky Actions

**100k final eval:**
- Mean reward: —
- vs previous best: —

## Phase 4: Tuned PPO Hyperparameters

**100k final eval:**
- Mean reward: —
- vs previous best: —

## Phase 5: DQN + PER

**100k final eval:**
- Mean reward: —
- vs previous best: —

## Phase 6: SimPLe

**100k final eval:**
- Mean reward: —
- vs previous best: —
