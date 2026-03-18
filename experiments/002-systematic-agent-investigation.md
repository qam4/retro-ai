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

## Investigation: Why Phase 1 Failed

### Key Data Points

| Metric | Random Agent | Trained PPO (eval) |
|--------|-------------|-------------------|
| Mean reward | 6.2 ± 5.5 | 3.0 ± 0.0 |
| Mean episode length | 192 | 139 |
| Fire rate | ~50% (random) | 64.7% |
| Unique actions | 18 (all) | 14 |
| Behavior | Diverse, stochastic | Fixed sequence, identical across seeds |

### Reward Signal Analysis

The memory-based reward for Satellite Attack is extremely sparse:
- Only 1.68% of steps produce non-zero reward (1 reward per ~60 steps)
- Reward values: +1 (small satellite), +3 (medium), +10 (large/rare)
- No negative rewards — death is signaled by episode termination only
- Episodes end via `done_when_score_drops` (score resets on death)

### Why Course Automobile Worked but Satellite Attack Doesn't

Course Automobile (the game that trained successfully) has fundamentally
different reward dynamics:
- **Dense reward**: Score increases continuously while holding Up — every
  frame the car moves forward generates reward
- **Simple strategy**: Hold Up + dodge = score. One dominant action (Up)
  is immediately rewarded
- **Clear episode structure**: 2-minute timer, predictable length

Satellite Attack requires:
- **Coordinated multi-step actions**: Move to align with target, THEN fire
- **Sparse reward**: Only get +1/+3/+10 when a shot actually hits
- **No intermediate signal**: No reward for aiming, approaching targets,
  or firing (only for hitting)

### The Core Problem

At 100k steps with ~60-step reward sparsity, the agent sees roughly
100k/60 ≈ 1,667 non-zero reward signals total. Split across 4 envs and
~470 episodes, that's ~3.5 reward events per episode. PPO needs to
credit-assign these sparse rewards back to the correct actions in a
[3,3,2] multi-discrete space — that's 18 possible actions per step.

The signal-to-noise ratio is simply too low for PPO to learn a meaningful
policy in 100k steps. The training rolling reward hovering around 5-7
(≈ random baseline) confirms the agent never found a strategy better
than random.

### The Deterministic Eval Collapse

The trained model scores 3.0 with zero variance across all seeds because:
- PPO's stochastic policy during training explores somewhat (rolling
  reward ~6.2 ≈ random)
- But the deterministic policy (argmax) collapses to a fixed action
  sequence that happens to score 3.0
- The policy never learned which actions cause reward — it just learned
  a default behavior

### Comparison with Experiment 001

| Run | Algorithm | Intrinsic | Eval Reward | Notes |
|-----|-----------|-----------|-------------|-------|
| Exp 001 RND+DrQ | PPO | RND 1.0 | 1.0 | RND dominated signal |
| Exp 001 Tuned PPO | PPO | RND 1.0 | 1.0–4.0 | Same problem |
| Exp 001 DQN+PER | DQN | RND 1.0 | 13.0 | Collapsed to 1 action |
| Exp 002 Phase 1 | PPO | None | 3.0 | Fixed sequence |

Removing RND didn't help — the underlying problem is reward sparsity,
not intrinsic reward interference.

### Recommendations

**Option A: Reward shaping (most promising for 100k budget)**
- Add a small survival bonus (+0.01/step) so the agent learns to stay alive
- Add a firing bonus (+0.1 when fire is pressed) so the agent discovers
  that firing is important
- Keep the score delta as the primary reward
- Risk: shaped rewards can create degenerate policies (e.g. fire constantly)

**Option B: Increase budget to 500k–1M steps**
- Course Automobile needed 500k steps with dense reward
- Satellite Attack with sparse reward likely needs 1M+ steps
- But 100k should still show a learning signal if the reward is right

**Option C: Simplify the action space**
- Use discrete (18 actions) instead of multi-discrete [3,3,2]
- DQN with discrete actions scored 13.0 in exp 001 (best result so far)
- Discrete space is easier for value-based methods

**Recommended next step:** Try Option A (reward shaping) with a small
survival bonus and firing bonus, keeping 100k budget. If we see a clear
learning signal (reward trending up, not flat), that validates the approach
and we can scale to more steps.

## Results

*(To be filled in as experiments run)*

### Phase 0: Random Baseline
- Status: **Complete**
- Mean reward: 6.2 ± 5.5
- Best reward: 17.0
- Worst reward: 1.0
- Mean episode length: 192 ± 74
- Action distribution: uniform across all 18 actions (as expected)
- Notes: High variance — some episodes score 1 (quick death), others 14–17
  (lucky hits). The floor is ~6 reward. Any trained agent must consistently
  beat this.

### Phase 1: Vanilla PPO
- Status: **Complete**
- Training: 100k steps, 473 episodes, 1769s wall clock
- Training mean reward: 6.2 ± 6.1 (best: 43.0)
- Rolling reward trend: 4.3 → peaked 7.5 at ~75k → declined to 6.2 at 100k
- Eval (3 episodes, deterministic): reward 3.0 ± 0.0, length 139
- All 3 eval episodes: identical reward (3.0) and length (139) — zero variance
- Verdict: **FAIL.** Eval reward (3.0) is below random baseline (6.2).
  The agent learned a fixed deterministic policy that scores worse than random.
  Training rolling reward briefly exceeded random (~7.5 at 75k) but the
  deterministic policy doesn't generalize. The identical eval episodes suggest
  the agent converged to a single action sequence regardless of seed.
  Possible causes: reward signal too sparse, policy collapsed to a narrow
  behavior, or the environment's stochasticity (random_noop_max=30) isn't
  enough to diversify training.

### Phase 1b: PPO + Shaped Reward (survival bonus + time limit)
- Status: **Complete** (first run — re-run pending with improved instrumentation)
- Config: `survival_bonus=0.01/step`, `max_episode_steps=900`
- Training: 100k steps, 431 episodes, ~27 min wall clock
- Rolling reward trend: 6.0 → peaked ~10.4 at 92k → settled ~9.5 at 100k
- Eval (3 episodes, deterministic): reward 4.0 ± 0.0, length 141
- Verdict: **Mixed.** Training reward well above random but eval still collapses.

#### Reward Decomposition (retroactive analysis from metrics CSV)

| Metric | Phase 1 (vanilla) | Phase 1b (shaped) |
|--------|-------------------|-------------------|
| Total reward (mean) | 6.05 | 9.27 |
| Survival bonus component | 0 | 2.35 |
| Game score component | 6.05 | 6.92 |
| Episode length (mean) | 214 | 235 |
| Episodes | 1,404 | 431 |

The shaped reward adds ~2.35/episode from survival bonus. The actual game
score improvement is only +0.87 over vanilla — modest but real.

#### Learning Progression (game score component by quartile)

| Quartile | Game Score | Survival | Total | Ep Length |
|----------|-----------|----------|-------|-----------|
| Q1 (early) | 5.89 | 2.12 | 8.01 | 212 |
| Q2 | 7.31 | 2.34 | 9.65 | 234 |
| Q3 | 7.51 | 2.54 | 10.05 | 254 |
| Q4 (late) | 6.98 | 2.39 | 9.37 | 239 |

The agent briefly learns to score above random (Q2-Q3: 7.3-7.5 vs random
6.2) but regresses in Q4. The policy improves then drifts — not stable.

#### Key Observations
- 900-step time limit almost never triggers (1/431 episodes hit it).
  Episodes end via `done_when_score_drops` well before 900 steps.
- Min episode length = 98 steps — agent always scores something before dying.
- The deterministic eval collapse (4.0 ± 0.0) is the same pattern as Phase 1
  (3.0 ± 0.0): stochastic training policy explores and scores, but argmax
  collapses to a fixed low-scoring sequence.

### Phase 1b-v2: Re-run with Improved Instrumentation
- Status: **Complete**
- Training: 100k steps, 443 episodes, ~27 min wall clock
- Rolling reward trend: 2.0 → peaked ~9.1 at 95k → settled ~8.9 at 100k
- Eval (deterministic): reward 16.0 ± 0.0, length 515
- Eval (stochastic): reward 4.3 ± 4.7, length 251
- Verdict: **Deterministic policy beats random baseline (16.0 vs 6.2).**
  Stochastic policy is worse than random — the learned distribution is
  noisy but the mode (argmax) found a good fixed sequence.

#### Score Recording Bug (discovered during investigation)

The metrics CSV showed `score=3` for all 443 episodes. Investigation revealed
two bugs in the score reporting pipeline (not affecting training):

1. **ThreadedVecEnv overwrote terminal info**: `step_wait()` did
   `info.update(reset_info)` on episode end, replacing the terminal step's
   `score` with the new episode's post-startup score (always 3 because the
   startup sequence runs ~120 noop frames). Fixed by storing `terminal_info`
   as a separate key (matching SB3's SubprocVecEnv convention).

2. **Info JSON reported post-death score**: `make_info_json()` read the live
   score, but `done_when_score_drops` fires after the score has already
   dropped to 0. Fixed by tracking `peak_score_` (high-water mark) and
   reporting that instead.

3. **Hardcoded Course Automobile defaults removed**: `wire_memory_reward_system()`
   and `read_current_score()` had fallback logic reading IntRAM[54..55] when
   no score addresses were configured. `parse_timer_params()` defaulted timer
   addresses to 65/66. All game profiles now must configure their own addresses
   explicitly — no silent fallback to wrong addresses.

The reward computation (MemoryRewardSystem) was never affected — it uses its
own `read_score()` path which was always correct. All prior training results
remain valid.

#### Code Changes Made (session 2026-03-18)
- `src/videopac_rl.cpp`: Added `make_info_json()` / `make_info_json_error()`
  helpers. Info JSON `"score"` field now reports `peak_score_` (high-water
  mark, reset on `reset()`). Removed hardcoded Course Automobile defaults
  from `wire_memory_reward_system()`, `read_current_score()`, and
  `parse_timer_params()`. `read_current_score()` returns -1 when no score
  addresses are configured (omits score from JSON).
- `python/retro_ai/wrappers/threaded_vec_env.py`: Fixed `step_wait()` to
  preserve terminal step info under `terminal_info` key instead of
  overwriting it with reset info.
- `python/retro_ai/training/callbacks.py`: Added `rolling_ep_len` to the
  periodic training log line. Fixed `record_episode` to read from
  `terminal_info` (SB3 VecEnv stores terminal step info there after
  auto-reset) so the game score at death is captured, not the post-reset
  score of 0.
- Build: PGO profile data cleared (stale after code changes). All 38 C++
  tests pass. Both games smoke-tested.

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
