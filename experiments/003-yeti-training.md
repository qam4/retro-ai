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
