# Experiment 002: Systematic Agent Investigation

**Date:** 2026-03-16
**Status:** Planning
**Game:** Satellite Attack (Videopac/Odyssey²)
**Budget:** 100,000 timesteps per config

## Motivation

Experiment 001 showed that none of the four trained agents learned meaningful
gameplay. DQN collapsed to a single action, PPO variants learned to move but
not fire, and training rewards were inflated by RND intrinsic bonuses. SimPLe,
which is designed for 100k-step Atari, scored only 1.0 in eval.

We went too fast — built the whole pipeline and ran all configs without
validating each layer. This experiment takes a systematic approach: isolate
variables, validate each component, and only add complexity when the simpler
version works.

## Methodology

**Principle:** One variable at a time. Each phase builds on a validated
baseline. If a phase doesn't improve results, we investigate before moving on.

**Validation criteria for each phase:**
- Smoke test (500 steps): agent runs without errors, actions are diverse
- Checkpoint review at 25k and 50k: reward should be trending up
- If reward is flat or near-zero at 50k, stop and investigate — don't waste
  the remaining 50k steps
- Final eval (3 episodes, deterministic): agent should score > random baseline
- Video review: visually confirm the agent moves AND fires

**Random baseline:** Before any training, measure what random actions score
in 10 episodes. This is the floor — any trained agent must beat it.

## Phase 0: Establish Random Baseline

**Goal:** Measure what random play scores, so we have a floor to compare against.

**Method:**
- Run 10 episodes with random actions (no model)
- Record mean reward, best reward, episode length
- Record one video for visual reference

**Success:** We have a number. Any trained agent must beat it.

## Phase 1: Vanilla PPO (No Extras)

**Goal:** Verify the basic training loop works — PPO can learn from score
delta reward without any exploration aids.

**Config:** Minimal PPO
- `algorithm: PPO`, `reward_mode: memory`, `action_mode: joystick`
- `intrinsic_reward.enabled: false` (no RND)
- `augmentation: false` (no DrQ)
- `sticky_actions: 0` (no stochasticity)
- `num_envs: 4`, `total_timesteps: 100000`
- Default PPO hyperparameters

**Checkpoints:**
- 500 steps: smoke test — runs, diverse actions
- 25k steps: is reward trending up from random baseline?
- 50k steps: is reward meaningfully above random? If not, stop.
- 100k steps: final eval with video

**Success criteria:** Mean eval reward > random baseline. Agent visibly
moves and fires in video.

**If it fails:** The problem is in the reward signal, action space, or
environment setup — not in the exploration aids. Debug before proceeding.

## Phase 2: Add Exploration (RND)

**Goal:** Test whether RND intrinsic reward helps or hurts.

**Config:** Phase 1 + RND
- Same as Phase 1, plus `intrinsic_reward.enabled: true`
- Test with `coefficient: 0.1` first (not 1.0 — that overwhelmed the signal
  in experiment 001)

**Checkpoints:** Same as Phase 1.

**Success criteria:** Eval reward ≥ Phase 1. If lower, RND is hurting —
the coefficient is too high or the bonus is drowning the game score.

**If it fails:** Try `coefficient: 0.01`. If still worse, drop RND entirely.

## Phase 3: Add Augmentation (DrQ) and Sticky Actions

**Goal:** Test whether data augmentation and action stochasticity improve
sample efficiency.

**Config:** Best of Phase 1/2 + DrQ + sticky actions
- `augmentation: true`, `sticky_actions: 0.25`

**Success criteria:** Eval reward ≥ previous best.

## Phase 4: PPO Hyperparameter Tuning

**Goal:** See if tuned PPO hyperparameters help within the 100k budget.

**Config:** Best of Phase 1-3 + tuned hyperparameters
- `ent_coef: 0.05`, `n_steps: 512`, `learning_rate: 0.001`

**Success criteria:** Eval reward > previous best.

## Phase 5: DQN + PER

**Goal:** Test whether off-policy DQN with prioritized replay is more
sample-efficient than PPO at 100k steps.

**Config:**
- `algorithm: DQN`, `num_envs: 1`
- `prioritized_replay: true` (verify PER actually loads — experiment 001
  logged `prioritized_replay_unavailable`)
- Use best exploration settings from Phase 2-3

**Pre-check:** Verify `sb3-contrib` PER actually works before training.
The experiment 001 training log showed it fell back to standard replay.

**Success criteria:** Eval reward > best PPO variant.

## Phase 6: SimPLe World Model

**Goal:** Test whether the world model amplifies learning within 100k real
steps.

**Pre-check:** Before full training, verify:
- WorldModel forward pass produces reasonable outputs (not NaN/constant)
- DreamEnv generates diverse synthetic transitions
- PPO can learn on DreamEnv (even if poorly)

**Config:** Best PPO settings + `simple.enabled: true`

**Success criteria:** Eval reward > best non-SimPLe variant.

**If it fails:** The world model may not be learning a useful representation.
Check validation MSE — if it's high, the model isn't predicting well enough
to generate useful synthetic data.

## Monitoring Protocol

For every training run:

1. **Smoke test (500 steps):** Verify no crashes, check action distribution
2. **25k checkpoint:** Read metrics.csv, check reward trend
3. **50k checkpoint:** If mean reward ≈ random baseline, stop and investigate
4. **100k final:** Eval with video, compare to previous phases

Log all intermediate results in this document as we go.

## Results

*(To be filled in as experiments run)*

### Phase 0: Random Baseline
- Status: Not started
- Mean reward: —
- Best reward: —
- Mean episode length: —

### Phase 1: Vanilla PPO
- Status: Not started

### Phase 2: PPO + RND
- Status: Not started

### Phase 3: PPO + RND + DrQ + Sticky
- Status: Not started

### Phase 4: Tuned PPO
- Status: Not started

### Phase 5: DQN + PER
- Status: Not started

### Phase 6: SimPLe
- Status: Not started
