# Experiment 003: Yeti (MO5) Training

## Game
Yeti (1984, Loriciels) — Thomson MO5 platform game (Donkey Kong clone).
4 floors, 4 fruits to collect, reach princess at top. Snowballs roll down.
Jumping over snowballs gives +10 score (unlimited). Fruits give +10/+20/+30/+40 by floor.

## The Core Problem
Every RL approach converges to **snowball farming** on floor 1 — the agent
learns to jump over snowballs for +10 each, never climbing higher. This is
a local optimum that's hard to escape because the reward is dense and risk-free.

---

## Approaches Tried

### 1. PPO + Score Reward + Survival Bonus
- **Runs**: ppo_500k, ppo_2M, ppo_5M, ppo_10M
- **Result**: Score ~20, stays on floor 1, farms snowball jumps
- **Insight**: Survival bonus keeps it alive, score reward reinforces jumping

### 2. Height Reward Shaping
- **Milestone height** (ppo_5M_height, height_1.0, height_v2): Agent climbs to
  floor 2, comes back down for fruits, gets stuck
- **Delta(Y)** (ppo_5M_deltaY): Agent just jumps constantly (±reward oscillation)
- **Thresholded delta(Y)** (ppo_5M_thresh, thresh_100): Filters jumping, catches
  climbing. Coeff=10 works, coeff=100 causes oscillation/death
- **Insight**: Height reward is counterproductive — agent needs to go both up AND
  down to collect all 4 fruits before reaching princess

### 3. Higher Entropy (ppo_10M_highent)
- ent_coef=0.05 instead of 0.01
- **Result**: Similar to baseline, no improvement
- **Insight**: More randomness doesn't help when the local optimum is so attractive

### 4. IMPALA CNN at Full Resolution (ppo_5M_impala_fullres)
- 320×200 input, 4-block IMPALA ResNet (4.5M params)
- **Result**: Same snowball farming, reward ~23, plateaus by 500k steps
- **Insight**: Visual resolution isn't the bottleneck — reward landscape is
- **Bug found**: config merge ignored `resize: null` from game profile (fixed)

### 5. RND Intrinsic Exploration
- **v1** (ppo_5M_rnd, coeff=1.0): Score 224, ep_len 1005 — much better survival
  but still snowball farming. RND made it a better farmer, not a climber.
- **v2** (ppo_5M_rnd_v2, coeff=5.0, ent=0.05): Similar curves to v1
- **v3** (ppo_10M_rnd_v3, coeff=1.0, 10M steps): Peaked at mean 77.6 around
  episode 15k, then regressed to ~50. Never consistently discovered climbing.
- **Insight**: RND makes repetitive states boring, but snowball jumping gives
  +10 extrinsic reward each time, which outweighs the novelty penalty

### 6. Go-Explore Phase 1 (Exploration)
- Random actions + save state teleportation, no neural network
- **Result**: Mapped entire game in 8 minutes. 1134 cells across all 5 floors.
  Best score 500. All floors discovered including princess area.
- **Key**: Save states allow teleporting to any discovered position and exploring
  from there. No need to survive the journey.
- **Bug found**: Crayon's AudioSystem.cycle_counter not serialized, causing
  save/restore non-determinism after ~182 frames. Fixed by serializing full
  audio state.

### 7. Go-Explore Phase 2 — Random Starting States
- PPO trained starting from random archive save states (all floors)
- **Result**: Score 10 from game start. Agent learned to play from floor 4 but
  can't get there from floor 1.
- **Insight**: Starting from random positions doesn't teach the agent to chain
  the journey from start to finish.

### 8. Go-Explore Phase 2 — Backward Curriculum
- Start from floor 4, advance to floor 3, 2, 1, 0 as performance improves
- **Result**: Score 20-30 from game start. Curriculum advanced through all stages
  but agent forgot earlier stages (catastrophic forgetting).
- **Problems identified**:
  - Advance threshold too low (5.0) — advances before mastering each stage
  - No mixing of stages — forgets old skills
  - Stage advancement per-env not global (8 envs advance independently)
  - Single policy for visually different floors

---

### 9. Checkpoint Curriculum Ablation
Goal: figure out which knob in the checkpoint-curriculum training actually
matters — so far the curriculum design has been mostly guesswork. Run
`train_checkpoint_curriculum.py` five times with one knob changed at a
time, everything else held equal (5M steps, 8 envs, `fruit_bonus` reward,
seed 42, yeti_fruit profile).

Configs live in `experiments/003-yeti/configs/`:

| ID | reset | frontier | earlier | seed archive  | stall | Hypothesis                                   |
|----|-------|----------|---------|---------------|-------|----------------------------------------------|
| A  | 1.0   | 0.0      | 0.0     | —             | 15    | baseline; no curriculum → snowball farming   |
| B  | 0.0   | 1.0      | 0.0     | go_explore_fruit | 15 | no reset practice → agent forgets how to start |
| C  | 0.4   | 0.4      | 0.2     | go_explore_fruit | 15 | main hypothesis — balanced mix + seed works  |
| D  | 0.4   | 0.4      | 0.2     | —             | 15    | same as C but no seed → is the seed critical? |
| E  | 0.8   | 0.15     | 0.05    | go_explore_fruit | 10 | reset-heavy with occasional frontier starts  |

Two config bugs caught during smoke-testing:

- **Profile name.** The configs originally referenced
  `mo5_yeti_training` (a filename), but `GameProfileRegistry.load()`
  resolves by the `name:` field inside the YAML, not the filename.
  Correct profile is `yeti_fruit`.
- **Seed archive format.** The original plan pointed at
  `go_explore_v8/archive.pkl`, whose `cell_key = (y_bucket, x_bucket,
  score_bucket)`. The seeding code in `train_checkpoint_curriculum.py`
  expects `cell_key[2]` to be `fruits_remaining`. Silently dropped all
  but 5 cells. Switched to `go_explore_fruit/archive.pkl`, which does
  use the `fruits_remaining` format (32 cells at CP1, 29 at CP2,
  nothing at CP3/CP4).

Both fixes are in commit `b5a3ffd`.

**Results — last 20% of episodes, fraction of reset starts that reached ≥2 fruits:**

| Run | reset / frontier / earlier | seeded | reach ≥ 2 | reach ≥ 3 |
|-----|---------------------------:|:------:|----------:|----------:|
| A   | 1.0 / 0.0 / 0.0            |   —    |    1.4%   |    0.0%   |
| B   | 0.0 / 1.0 / 0.0            |   ✓    |   collapsed — frontier trap (see below) |
| C   | 0.4 / 0.4 / 0.2            |   ✓    |   79.1%   |    0.0%   |
| D   | 0.4 / 0.4 / 0.2            |   —    |    3.1%   |    0.0%   |
| E   | 0.8 / 0.15 / 0.05          |   ✓    |   80.9%   |    0.0%   |

**Headline: the Go-Explore seed archive is the load-bearing ingredient.**

- Seeded runs (C, E) reach fruit 2 from reset ~80% of the time.
- Unseeded runs (A, D) reach fruit 2 <5% of the time — D with the
  balanced mix is barely better than A with no curriculum at all.
- The reset/frontier mix matters much less than seeding. 40% reset (C)
  and 80% reset (E) give essentially the same result.

**B's frontier-only collapse.** The one run where the curriculum went
degenerate. The checkpoint buffer accumulated a single CP4 save
(`fruits_rem=0`, 4 fruits collected) very early, and since
`frontier_fraction=1.0` means "always pick the highest level with ≥1
cell", every subsequent episode loaded that one state. Inspection of
the state showed a snowball already adjacent to the player with no time
to react — agent dies within ~100 frames, episode ends, reload, repeat.
4.9M 1-step episodes in 5M steps, fps collapsed 6× over the run due to
repeated reset/load overhead. Not a failure of the "frontier-only"
idea per se — a failure of *unfiltered* frontier selection. A quality
signal on frontier saves (e.g. require the agent to have survived N
frames from the save in past attempts) would fix it. (TODO)

**The wall at fruit 3.** No run collected fruit 3 even once in 5M
steps, including C and E despite 40-15% of episodes starting from a
CP2 state. The seed archive (`go_explore_fruit/archive.pkl`, 85 cells)
caps at 2 fruits collected, so there's no bootstrap, and PPO can't
discover fruit 3 from CP2 on its own under the current reward. Two
plausible next moves: (1) extend the Go-Explore archive to include
CP3/CP4 states, and re-run with a better seed; (2) notice that fruit 3
is geometrically harder (higher floor, more snowballs, longer path
from CP2) and deserves its own investigation.

**Reward gap at the princess.** Observed while debugging B: when the
agent reaches the princess, the game increments level and repopulates
`fruits_remaining` from 0 back to 4. The `fruit_bonus` reward formula
only fires on `curr_fruits < prev_fruits`, so princess = 0 reward. Even
if the curriculum could push the agent to fruit 4, it has no incentive
to actually *touch* the princess rather than just dying. A new reward
`fruit_princess_bonus` that detects level-complete (fruits jump up,
lives preserved, bonus reset) and pays a one-shot bonus has been added
(commit `705cd7d`) — not used in this ablation, available for future
runs.

**Tooling produced during the ablation:**

- `python/retro_ai/training/episode_metrics.py` — pure aggregation
  function over episode rows. Returns a flat `{tag: scalar}` dict.
- `python/retro_ai/training/callbacks.py::EpisodeMetricsCallback` —
  SB3 callback that pulls episodes from the `EpisodeLogger`'s ring
  buffer and pushes aggregates to TB via `self.logger.record`. Wired
  into all three training scripts.
- `scripts/episodes_to_tb.py` — one-shot replay of `episodes.csv` into
  a TB event file, using the same aggregator so tag schemes don't drift.
  Useful for runs that finished before the callback existed (used to
  visualize A/B/C/D/E).

The ablation runs' TB replays live at
`output/mo5/yeti/training/ablation_*/tb_replay/`.

---

## Technical Findings

### Emulator Determinism
- MO5 emulator (Crayon) is deterministic from a cold start
- Save/restore had non-determinism due to unserialized AudioSystem state
- Fixed: serialize cycle_counter, cycles_since_toggle, prev_sample, dac_sample,
  dac_active, write_pos, read_pos, toggle_count, porta_toggle_count
- Also added MasterClock state and cassette cycle state to save format (v3)
- Remaining: first reset (live startup) differs slightly from cached restore
  (cosmetic framebuffer difference, doesn't affect game logic)

### Performance
- MO5/Crayon: ~1900 emu_fps at 84×84 with 8 envs, ~800 at 320×200
- Videopac: ~250 emu_fps at 84×84 with 8 envs (VDC rendering is expensive)
- Skip-render optimization: run_frame(false) for intermediate frame_skip frames
- Go-Explore Phase 1: ~2000 fps (no neural network, pure emulator speed)

### Config Merge Bug
- `resize: null` in game profile was ignored because merge logic didn't track
  explicit keys in GameProfile. Fixed by adding `_explicit_keys` tracking.

---

## Next Steps
- Fix Phase 2 backward curriculum: mix 50% current frontier + 50% earlier stages
- Increase advance threshold to require mastery before progressing
- Global stage tracking across all envs
- Consider: is backward curriculum the right approach, or should we use
  the discovered trajectories as demonstrations for imitation learning?
