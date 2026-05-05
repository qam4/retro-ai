# Experiment 003: Yeti (MO5) Training

## Game
Yeti (1984, Loriciels) — Thomson MO5 platform game (Donkey Kong clone).
4 floors, 4 fruits to collect on the way up, reach the princess at the top.
Snowballs roll down. Jumping over a snowball gives +10 score (unlimited).
Fruits give +10/+20/+30/+40 by floor. Level completes on princess-touch;
the game then restarts with all fruits repopulated on the same layout.

## The Core Problem

Every RL approach plateaus in the early-game. The observed mode has been
**snowball farming** on floor 1 — repeatedly jump snowballs for +10 score
and never climb. Score stays around 20-30. This note tracks what we've
tried, what disk evidence there is, and what the current-best understanding
of the failure mode is.

**Terminology used throughout:** CP0 = game reset (0 fruits collected),
CP1 = 1 fruit collected, … CP4 = 4 fruits collected (next step is
princess). "Per-segment success" = probability of advancing at least one
CP when the episode *starts* at a given CP. "End-to-end chain" = probability
of reaching a CP within the same episode that started at CP0.

A note on evidence: pre-Apr-2026 runs don't have `episodes.csv` — only a
`final_model.zip` and a checkpoint buffer (where applicable). Claims
about those runs come from old logs / prior conversations and are marked
"narrative" vs "verified".

---

## Approaches Tried

### 1. PPO + Score Reward + Survival Bonus  *(narrative)*
- **Runs**: `ppo_500k`, `ppo_2M`, `ppo_5M`, `ppo_10M` (config.yaml era)
- **Result**: Score ~20, stays on floor 1, farms snowball jumps
- **Insight**: Survival bonus keeps the agent alive, score reward reinforces
  jumping. Local optimum is dense and safe, so PPO stays there.

### 2. Height Reward Shaping  *(narrative)*
- **Milestone height** (`ppo_5M_height`, `_1.0`, `_v2`): Agent climbs to
  floor 2, returns for fruits, gets stuck
- **Delta(Y)** (`ppo_5M_deltaY`): Agent jumps constantly (±reward oscillation)
- **Thresholded delta(Y)** (`ppo_5M_thresh`, `_100`): Filters jumping,
  catches climbing. Coeff=10 works, coeff=100 causes oscillation/death
- **Insight**: Height reward is counterproductive — the game requires both
  up AND down motion to collect fruits before reaching the princess.

### 3. Higher Entropy (ppo_10M_highent)  *(narrative)*
- `ent_coef=0.05` instead of 0.01
- **Result**: Similar to baseline
- **Insight**: More action randomness doesn't escape the snowball-farming
  basin when the reward gradient points into it.

### 4. IMPALA CNN at Full Resolution (ppo_5M_impala_fullres)  *(narrative)*
- 320×200 input, 4-block IMPALA ResNet (~4.5M params)
- **Result**: Same snowball farming, reward ~23, plateaus by 500k steps
- **Insight**: Visual resolution isn't the bottleneck. The reward landscape
  is.
- **Bug found**: config merge ignored `resize: null` from game profile
  (fixed by tracking explicit keys in GameProfile).

### 5. RND Intrinsic Exploration  *(narrative)*
- **v1** (`ppo_5M_rnd`, coeff=1.0): Score 224, ep_len 1005 — much better
  survival but still snowball farming. RND made it a better farmer.
- **v2** (`ppo_5M_rnd_v2`, coeff=5.0, ent=0.05): Similar to v1
- **v3** (`ppo_10M_rnd_v3`, coeff=1.0, 10M steps): Peaked at mean 77.6
  around episode 15k, regressed to ~50. Never consistently discovered
  climbing.
- **Insight**: RND makes repeated states boring, but +10 per snowball
  jump outweighs the novelty penalty.

### 6. Go-Explore Phase 1 (state-space mapping)  *(partially verified)*
- Random actions + save-state teleportation, no neural network.
- **`go_explore_v8`** (verified): 1134 cells, cell_key =
  (y_bucket, x_bucket, score_bucket). Scores 0..500. All 5 y-buckets
  covered (top of screen = y_bucket 4 = floor 4 vicinity). No direct
  evidence the princess was ever *touched* in this archive — we checked
  `0x0A23` as a level counter today and it isn't; no other level byte
  known yet. So "reached the princess area" is the conservative
  formulation; "touched the princess" is unverified.
- **`go_explore_fruit`** (verified): 85 cells, cell_key =
  (y_bucket, x_bucket, fruits_remaining). Distribution:
  24 @ CP0, 32 @ CP1, 29 @ CP2. **Zero cells at CP3 or CP4.** This is
  the archive the ablation used for seeding.
- Earlier `go_explore_v2..v7`, `go_explore_fruit_v2`,
  `go_explore_phase2_smoke` dirs exist on disk but have no `archive.pkl`
  — incomplete or unstarted runs.
- **Key capability**: save states let you teleport to any discovered
  position and explore from there. No need to survive the journey.
- **Bug found**: Crayon's AudioSystem state wasn't serialized, causing
  save/restore non-determinism after ~182 frames. Fixed by serializing
  the full audio state (cycle_counter, cycles_since_toggle, prev_sample,
  dac_sample, dac_active, write_pos, read_pos, toggle_count,
  porta_toggle_count). Also added MasterClock state and cassette cycle
  state to the save format (v3).

### 7. Go-Explore Phase 2 — Random Starting States  *(narrative)*
- `go_explore_phase2` dir on disk has `final_model.zip` + TB events,
  no episodes.csv (pre-config-driven). Narrative from prior sessions:
- PPO trained starting from random archive save states (all floors).
- **Result**: Score 10 from game start. Agent could play from a floor-4
  start but couldn't chain up from a reset start.
- **Insight**: Random starts don't teach the agent to chain — each start
  position is a separate sub-problem, and they don't cohere into a
  single policy that knows the journey.

### 8. Go-Explore Phase 2 — Backward Curriculum  *(narrative)*
- `go_explore_phase2_v2`, `_v3` on disk with final_model.zip only.
- Narrative: Start from floor 4, advance to 3, 2, 1, 0 as performance
  crossed a reward threshold.
- **Result**: Score 20-30 from game start.
- **Problems identified at the time**: advance threshold too low
  (advanced before mastering each stage); no mixing of stages →
  forgetting; per-env (not global) stage advancement; single policy for
  visually different floors.
- *Today's reading*: the "catastrophic forgetting" framing may have
  overstated what was actually happening. The ablation's D run (no seed,
  balanced mix) shows the same external symptom — 3% CP0→CP2 chain rate
  — without any forgetting dynamics. A lower-CP state simply gets very
  little training signal if the frontier keeps advancing. Some of what
  looked like forgetting was probably just "never learned this segment".

### 8.5. Checkpoint Curriculum (pre-ablation)  *(verified — buffers only)*

Between the phase-2 work and the ablation, we iterated on a different
curriculum design: a **checkpoint curriculum** where the agent starts
most episodes from game reset and a subset from saved states captured
whenever it previously reached CP1, CP2, etc. See
`scripts/train_checkpoint_curriculum.py`.

Three runs on disk with `checkpoints.pkl`:

| Run                    | CP0 | CP1 saves | CP2 saves | CP3 saves | CP4 saves |
|------------------------|----:|----------:|----------:|----------:|----------:|
| `curriculum_vanilla_v2` |  0  |    19,006 |       260 |         0 |         0 |
| `curriculum_v4`        |  0  |    45,407 |    22,584 |         2 |         0 |
| `curriculum_v5` (+10M, resumed from v4) | 0 | 67,411 | 22,602 | 2 | **1** |

The ratio `CP2_saves / CP1_saves` is a proxy for "once the agent
reached CP1 in a reset chain, how often did it continue to CP2?":

- vanilla_v2: 260/19006 ≈ **1.4%**
- v4: 22584/45407 ≈ **49.7%**
- v5: near-flat after v4 — only 18 new CP2 saves in 10M additional steps

Live per-segment success as reported in the training logs was
`[0→1:≥90%, 1→2:0%, 2→3:0%, 3→4:0%]` for all of these. **Meaning: the
agent never learned to succeed when starting at CP1 or higher.** The
CP2/CP3/CP4 saves in v4 and v5 came from *reset chains* where PPO
happened to reach those checkpoints as a side effect of a successful
reset episode, not from targeted learning of the segments.

v5's single CP4 save is a curiosity — one end-to-end reset chain
reached CP4 in 10M steps. We haven't verified whether that save is a
viable state or an unrecoverable one (snowball-adjacent, like the CP4
save that poisoned ablation B).

### 9. Checkpoint Curriculum Ablation  *(verified)*

Goal: figure out which knob in the checkpoint curriculum actually does
the work. Five runs of `train_checkpoint_curriculum.py`, one knob
changed at a time, everything else held equal (5M steps, 8 envs,
`fruit_bonus` reward, seed 42, `yeti_fruit` profile).

Configs in `experiments/003-yeti/configs/`:

| ID | reset | frontier | earlier | seed archive        | stall |
|----|------:|---------:|--------:|---------------------|------:|
| A  |   1.0 |      0.0 |     0.0 | —                   |    15 |
| B  |   0.0 |      1.0 |     0.0 | `go_explore_fruit`  |    15 |
| C  |   0.4 |      0.4 |     0.2 | `go_explore_fruit`  |    15 |
| D  |   0.4 |      0.4 |     0.2 | —                   |    15 |
| E  |   0.8 |     0.15 |    0.05 | `go_explore_fruit`  |    10 |

**Two config bugs caught during smoke-testing** (fixed in `b5a3ffd`):

1. **Profile name.** Configs originally referenced `mo5_yeti_training`
   (a filename). `GameProfileRegistry.load()` matches on the YAML's
   `name:` field. The correct profile name is `yeti_fruit`.
2. **Seed archive format.** The plan pointed at `go_explore_v8/archive.pkl`,
   whose `cell_key[2]` is `score_bucket` (not `fruits_remaining` as the
   seeding code assumes). Silently dropped nearly all cells. Switched
   to `go_explore_fruit/archive.pkl` (24@CP0, 32@CP1, 29@CP2,
   0@CP3/CP4).

**Per-segment results — last 20% of episodes, fraction that advanced
at least one CP from their start level:**

| Run | start=CP0 (n)   | start=CP1 (n) | start=CP2 (n)   | notes              |
|-----|-----------------|---------------|-----------------|--------------------|
| A   | **98.3%** (2578)| —             | —               | no curriculum      |
| C   | 98.3% (2360)    | **76.4%** (518) | 0% (1988)     | seeded, balanced   |
| D   | 95.9% (2116)    | 2.9% (238)    | 0% (269)        | unseeded, balanced |
| E   | 99.5% (2719)    | 75.0% (80)    | 0% (529)        | seeded, reset-heavy |
| B   | —               | —             | —               | collapsed (see below) |

**The per-segment picture cuts cleanly:**

- **CP0 → CP1**: ~98% across *all* runs, including the no-curriculum
  baseline A. **The curriculum doesn't do anything for this segment.**
  Snowball-farming is NOT about "can't reach fruit 1"; agents learn
  that fine given enough time. All the prior "score ~20" results were
  plateau-at-one-fruit + occasional snowball jumps, not a total failure
  to ever collect any fruit.
- **CP1 → CP2**: 75-76% seeded, ~3% unseeded. **This is where the
  curriculum earns its keep, but only if seeded.** Without CP1 seeds,
  the agent barely ever gets to practice the segment from a CP1 start
  — unseeded D only logged 238 CP1-start episodes vs seeded C's 518
  despite identical start-distribution fractions. CP1 saves have to
  accumulate organically from reset chains and that's slow.
- **CP2 → CP3**: **0% for every run.** The agent never collected fruit
  3 even once in 5M steps. The seed archive has zero CP2 cells — wait,
  it has 29 CP2 cells. The issue is different: the agent starts at CP2
  and dies/stalls before reaching fruit 3. Fruit 3 must be a genuinely
  harder segment to solve via PPO-from-scratch, or the current reward
  isn't pushing toward it.

**B's pathology.** Same shape as what was tentatively called
"catastrophic forgetting" in Phase 2 backward curriculum. Here, the
checkpoint buffer accumulated a single CP4 save very early (4 fruits
collected, snowball already adjacent to the player — unwinnable).
With `frontier_fraction=1.0`, every subsequent episode loaded that one
state, the agent died within ~100 frames, episode ended, reload,
repeat. 4.9M 1-step episodes in 5M training steps; fps decayed 6×
because of reset/load overhead. Not a failure of the "frontier only"
curriculum idea — a failure of **unfiltered** frontier selection.
Fix: add a quality signal to frontier cells (e.g., require past
survival ≥ N frames from the save), or require ≥ K cells at a level
before using it as frontier. TODO.

**Reward gap at the princess.** When the agent reaches the princess,
the game repopulates `fruits_remaining` from 0 back to 4 for the new
level. `fruit_bonus` only pays on `curr_fruits < prev_fruits`, so
princess = 0 reward. Even if training could push the agent past CP4,
there's currently no incentive to actually *touch* the princess rather
than die. Added `fruit_princess_bonus` in commit `705cd7d` (detects
level-complete via `fruits↑ + bonus↑ + lives_preserved`, pays using
`prev_bonus` so fast finishes pay more). Not used by the ablation
runs — they didn't get close enough for it to matter.

**Tooling produced during the ablation:**

- `python/retro_ai/training/episode_metrics.py` — pure
  `aggregate(rows, max_level) -> {tag: scalar}` function.
- `python/retro_ai/training/callbacks.py::EpisodeMetricsCallback` —
  SB3 callback; wired into all three training scripts; writes the
  per-segment metrics above to TB alongside default SB3 tags.
- `scripts/episodes_to_tb.py` — one-shot replay of `episodes.csv`
  into a TB event dir, sharing the same aggregator so tag schemes
  don't drift. Used to visualize A/B/C/D/E.

### 10. Go-Explore from validated CP2 seeds  *(verified)*

First half of the closed-loop plan from approach 9's followup: use
Go-Explore to push past the CP2→CP3 wall, starting from CP2 save-states
rather than game reset. Implemented via
`scripts/go_explore.py --seed-archive ... --seed-min-cp 2`, which loads
the given archive, filters it through the state validator, and adds
the viable cells to the exploration archive before the main loop starts.

Two prerequisites added for this experiment:

- **State validator** (`python/retro_ai/training/state_validator.py` +
  `scripts/filter_archive.py`). Rule: load state, noop probe, reject if
  bonus==0 at load OR bonus doesn't drop by min_drop=2 over 30 frames.
  Unit-tested, spot-checked against B's known-doomed CP4 and
  `go_explore_fruit` cells. The curriculum's inline `_validate_checkpoint`
  was refactored to delegate to this module, so training and offline
  filtering now share one rule.
- **Go-Explore `--seed-archive`** (commit `f833790`). Loads a prior
  archive.pkl, optionally filters by CP level, runs the validator,
  adds survivors to the in-memory `CellArchive`.

**Run**:
- 5M exploration steps, seed `go_explore_fruit/archive.pkl`,
  `--seed-min-cp 2`.
- 85 seed cells → 56 filtered by CP level (CP0/CP1) → 16 rejected by
  validator (frozen / bonus=0) → **13 cells seeded** (all CP2).
- Output: `output/mo5/yeti/go_explore_from_cp2/`.

**Result**: 103 cells discovered total. 5 y-buckets covered (including
11 cells at y_bucket 4, the princess area). Best score 470.
**Zero CP3 or CP4 cells.** Fruit 3 was never collected.

Breakdown of the final archive:

| fruits_remaining | interpretation | cells |
|:---:|---|:---:|
| 4 | CP0 (no fruits collected) | 34 |
| 3 | CP1 | 37 |
| 2 | CP2 | 32 |
| 1 | CP3 | 0 |
| 0 | CP4 | 0 |

The CP0 and CP1 cells are new — they accumulated as dying exploration
attempts (re)populated them, even though seeding only started at CP2.
So random-action search from CP2 didn't push up to CP3; it mostly
died back down to CP0/CP1.

**What this rules out**:

- "Just give Go-Explore CP2 seeds and it will find CP3." 5M steps,
  13 viable starting points, all of Go-Explore's weighting heuristics,
  no CP3. The closed-loop plan as written in approach 9's followups
  is blocked at this step.

**What this doesn't rule out**:

- Much longer Go-Explore (20-50M steps) might eventually hit CP3 by
  chance. Random-action search is theoretically complete; we just ran
  out of patience.
- Different Go-Explore knobs (sticky_prob, cell resolution, death
  detection threshold) might help — we used the defaults that worked
  at lower CPs.
- The 2 CP3 saves and 1 CP4 save that `curriculum_v5` accumulated (via
  PPO reset chains over ~20M steps) haven't been validated yet. If any
  are usable, they could seed a CP3 curriculum without needing
  Go-Explore at CP3 at all.

---

## Where we stand after all this

Reframed in plain terms:

- **CP0 → CP1** is learnable by plain PPO given enough training. Agents
  across every approach reach fruit 1 with high reliability. "Snowball
  farming" isn't "agent can't find fruit 1" — it's "agent collects fruit 1,
  then spends the rest of the episode on floor 1 racking up +10 snowball
  jumps".
- **CP1 → CP2** is learnable by PPO, but only if it gets direct exposure
  to CP1-start episodes. Reset-only training almost never gives PPO that
  exposure (the agent has to reach CP1 on its own, and only ~3% of CP1
  episodes under reset-only go on to CP2 — so the signal stays sparse).
  The checkpoint curriculum fixes that by injecting CP1-start episodes
  directly, but only if there are CP1 save-states in its "bag" to sample
  from. The bag starts empty unless it's pre-populated; the ablation
  showed a pre-populated bag (32 CP1 saves, from Go-Explore) pushes CP1→CP2
  success to ~76%, while an empty bag leaves it near baseline (~3%).
- **CP2 → CP3** is the current wall. It's 0% across every run we have —
  including the two ablation runs with 29 CP2 saves in the bag and ~2000
  CP2-start episodes of practice. More CP2 practice isn't helping. Either
  the CP2 save-states are bad starting points (some we've inspected are
  literally unwinnable — snowball adjacent at load time), or CP2→CP3 is a
  harder task than CP1→CP2 for some other reason (longer climb, more
  snowballs, or a reward signal that doesn't push hard enough toward
  fruit 3). **Approach 10 tried random-action Go-Explore from 13 validated
  CP2 seeds for 5M steps and still found zero CP3 cells**, which narrows
  the cause: random actions can't cross the boundary either.
- **CP3 → CP4** — we have two CP3 saves total across all runs (from
  curriculum_v4 and v5, over ~20M combined training steps). No reliable
  data on whether CP3→CP4 is solvable.
- **CP4 → princess** — we have one CP4 save (curriculum_v5) and one in
  ablation B's doomed state. We don't know whether either captures a
  viable starting position. We don't have a confirmed princess-touch yet.

**One infrastructure gap matters more than anything else:** we don't have a
way to tell whether a given save-state is a *viable* starting point.
The existing `_validate_checkpoint` runs 20 noop frames and passes the
save if the bonus counter changed — that passed B's unwinnable CP4.
Everything downstream of "use save N as a curriculum seed" is on shaky
ground until we can filter out doomed states.

---

## Plan: reaching the princess

The plan is a single loop, run once by hand to test the assumption, then
productionized if it works. Each iteration pushes the frontier up one
checkpoint.

**Status**: partially scouted during this session. We found a usable
signal for state validation (see below) and sketched what the validator
and the loop should look like, but neither is committed code yet.

### Validator design (provisional — not yet committed)

A state is **unusable** if either of these is true with noop actions:

- `bonus == 0` at load time (no time left).
- `bonus` does not decrement over ~30 noop frames (game clock is
  frozen, meaning the agent has no effective control — typically
  mid-death-animation or similar).

Both conditions were spot-checked against:
- **Known-doomed** state: ablation B's CP4 save. Bonus=767 at load,
  drops by 1 over 30 frames → flagged.
- **Visually-problematic** CP1 states from `go_explore_fruit`: 6 cells
  where the player loads on top of a snowball. All 6 show
  bonus-frozen. Correctly flagged.
- **Visually-clean-but-still-doomed** states: 3 cells where the player
  loads standing still, not obviously threatened, but bonus is 0 at
  load and lives is lost after ~240 frames of noop with the character
  drifting upward off-screen. Mechanism unknown (maybe a time-up
  penalty); operationally correct to reject them.

Applying this provisional rule to `go_explore_fruit/archive.pkl` (85
cells): ≈55% viable (47/85). CP0 71% viable, CP1 53%, CP2 45%.

Open concerns: we haven't tested the rule at scale, and haven't tried
to find false positives (states the rule rejects but a trained policy
could actually play). If the rule over-rejects, refine it. The rule is
not set in stone.

### Render-after-load bug (separate issue)

While dumping frames for manual review we noticed the full-size
(320×200) framebuffer from a `load_state`-restored env is missing the
HUD (score + bonus text area) compared to a cold-reset render. The 84×84
downsampled observation doesn't make this visible, but it suggests the
render path or some state isn't fully restored on load. Worth a
separate C++-focused session to investigate — doesn't block the plan
here.

### Planned next steps

1. **Finish the validator.**
   - New module `python/retro_ai/training/state_validator.py` with the
     two-condition bonus-drop rule above.
   - Unit tests against known-good (fresh reset at various frame
     counts) and known-bad (B's CP4, the 9 frozen CP1 cells).
   - Thin CLI `scripts/filter_archive.py` that reads an archive.pkl
     and writes a filtered archive.pkl alongside.

2. **Closed-loop experiment (PPO ↔ Go-Explore).**
   - Add `--seed-states` flag to `scripts/go_explore.py` so Go-Explore
     can teleport to a random save-state instead of cold-starting.
   - Run the filter over every seed archive we have. Produce a
     validated CP2 seed pool.
   - Run Go-Explore from the validated CP2 pool, target CP3. Validate
     any CP3 states it finds, add them to the archive.
   - Re-run ablation C's config (10M steps) with the enriched + filtered
     archive. Measure CP2→CP3 and CP3→CP4.
   - Iterate the same recipe for CP3→CP4→princess.

### Followups not on the critical path

- **Quality-filter the curriculum's frontier selection.** B's pathology
  happens when one bad save at the highest CP becomes the entire
  frontier sample. Require ≥ K validated saves at a level before it's
  usable as frontier, or weight-sample proportional to past survival.
- **HUD-after-load render bug** (see above).
- **Characterize the "bonus=0 → lose-a-life after ~240 frames"
  behavior.** Validator correctly rejects those states regardless of
  mechanism, but we shouldn't have a mystery in our game model.
- **Multi-seed confirmation of the ablation.** C vs D with 3-5 different
  random seeds, to confirm the 76% vs 3% gap isn't seed-42 noise.
- **Reconcile `train_segment.py`'s 0% CP1→CP2 result** with the
  ablation's 76% on the same segment.
- **Directory reorg**: `output/mo5/yeti/` into
  `training/ | exploration/ | smoke/ | eval/`.

---

## Technical Findings

### Emulator Determinism
- MO5 emulator (Crayon) is deterministic from a cold start.
- Save/restore was non-deterministic until the AudioSystem + MasterClock
  + cassette cycle state were all serialized. Save format v3 now
  round-trips correctly.
- Remaining: the first reset (live startup) differs slightly from a
  cached restore (cosmetic framebuffer difference, doesn't affect game
  logic).

### Performance
- MO5/Crayon: ~1900 emu_fps at 84×84 with 8 envs, ~800 at 320×200.
- Videopac: ~250 emu_fps at 84×84 with 8 envs (VDC rendering is
  expensive).
- Skip-render optimization: `run_frame(false)` for intermediate
  frame_skip frames.
- Go-Explore Phase 1: ~2000 fps (no neural network, pure emulator
  speed).

### Config Merge Bug
- `resize: null` in a game profile was ignored because the merge logic
  didn't track explicit keys. Fixed by adding `_explicit_keys` tracking
  on `GameProfile`.
