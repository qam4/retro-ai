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

### 11. Go-Explore with richer cell keys  *(verified)*

Revisiting approach 10 after realising the cell-key scheme was
collapsing meaningfully distinct game states into one bucket.

Two fixes to `scripts/go_explore.py`:

- **Cell-key grid rework**: y-buckets are 32 px tall anchored at the
  bottom of the screen (one bucket per floor); x-buckets are 8 px in
  game-x space (1 sprite-width, 10 buckets). The previous grid had
  30-px y-buckets anchored at the top, which mid-jump states from
  floor 3 to hit the "floor 4" bucket. Replacing with bottom-anchored
  32-px buckets resolves this.
- **Cell key third element: frozenset of collected fruit-floors**
  instead of the `fruits_remaining` count. A state "collected fruit 1
  only" and a state "collected fruit 3 only" now occupy different
  cells instead of collapsing into the same "1 fruit collected"
  bucket. Per-fruit presence is read directly from 4 RAM addresses
  identified via CP0..CP4 diff (fruit on floor n: 0x2FAD/0x2F00/
  0x2E68/0x2DD8 — non-zero when sprite is present, zero when
  collected).
- **`fruits_order` list persisted per-cell** in the archive: the
  chronological sequence of floor numbers as fruits were picked.
  Not part of the key (two physical histories ending in the same
  fruit-set collapse to one cell) — kept only for analysis.

Commit: `2c62c72`.

**Run** (output/mo5/yeti/go_explore_v9): 5M steps, fresh start (no
seed), 41 minutes wall time.

**Result: 432 cells discovered**, a 5× jump from prior runs. Breakdown:

| fruits-collected set                 | cells |
|:-------------------------------------|------:|
| none                                 |    41 |
| singletons {1}, {2}, {3}, {4}        | 35, 37, 38, 21 |
| pairs {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4} | 37, 31, 5, 33, 7, 41 |
| triples {1,2,3}, {1,2,4}, {1,3,4}, {2,3,4}     | 32, 23, 19, 23 |
| all four (CP4)                       |     9 |

Notable findings:

- **All 15 non-empty fruit-subsets found.** {1,4} and {2,4} are rare
  (5 and 7 cells), suggesting those pairs need specific navigation
  random actions rarely produce.
- **9 CP4 cells — all 4 fruits collected.** First time any run has
  captured this state. Scores 200-230 (consistent with fruits + a
  handful of snowball jumps). Validated: 6/9 viable.
- **`fruits_order` for the 9 CP4 cells is uniformly `[1, 2, 4, 3]`.**
  Random actions found ONE path to all 4 fruits, and it isn't the
  "natural" floor-order. Worth remembering when interpreting learned
  policies later.
- **Zero cells at y_bucket 4 with all fruits collected.** None of the
  CP4 states are "at the princess with all fruits"; they're
  somewhere else on the map. The agent didn't touch the princess —
  `fruits_order` length never exceeded 4, which would have been the
  tell (collected-then-re-appeared after level-complete).
- The agent hit all 5 y-buckets including 30 cells in the princess
  area (y_bucket 4), just never with a complete fruit set.

Why this unblocked us (approach 10 got zero CP3 cells from 5M steps
with the old scheme):

With the old key, many distinct game states collapsed into the same
cell — e.g. "collected fruit 2, on floor 3" and "collected nothing,
on floor 3" shared a cell, and only one got saved. Go-Explore's
teleport-and-extend loop needs distinct cells to make progress: if
saving a "more progress" state overwrites the "less progress"
state at the same key, the frontier can't accumulate. The richer key
lets every incremental subset-of-fruits become its own starting
point, and the chain of short random walks compounds.

**Viability breakdown of the 432 cells** (via state_validator):
CP0 32/41, CP1 88/131, CP2 81/154, CP3 43/97, CP4 6/9. Higher CPs
have more frozen states (random actions save more often at risky
moments), but every level has viable seeds.

This archive is the first we've had that covers every CP (including
CP4) with validated save-states from a single source.

### 12. Segment training on v9 seeds — CP1→CP2  *(verified)*

First clean test of per-segment training (one fresh policy, trained
only on CP_N starts). Prior attempts (segment_1to2, _v2, _v3) used
curriculum_v5's 100 CP1 states, all clustered at the one spot fruit 1
gets collected. Approach 11's v9 archive gives us 88 validated CP1
states spread across the map — the diverse-seed pool that should let
per-segment training actually learn.

Config: `experiments/003-yeti/configs/segment_1to2_v4.yaml` (fresh
PPO policy, 5M steps, fruit_bonus reward, 5 settle frames after
load_state). Seeds extracted from v9 via `scripts/extract_seeds.py`.

**Result: 41% CP1→CP2 success in the last 20%, learning curve still
climbing.**

Progression over 10 training bins:

| bin | step     | CP1→CP2 |
|----:|---------:|--------:|
| 0   | 489k     |  7.7%   |
| 1   | 942k     |  8.8%   |
| 2   | 1.3M     |  6.3%   |
| 3   | 1.8M     | 11.8%   |
| 4   | 2.3M     | 13.0%   |
| 5   | 2.9M     | 24.9%   |
| 6   | 3.4M     | 35.5%   |
| 7   | 3.9M     | 31.9%   |
| 8   | 4.4M     | 38.7%   |
| 9   | 5.0M     | 40.9%   |

Comparison to prior attempts on the same segment:
- `segment_1to2` (v5 seeds, no settle):       ~0%
- `segment_1to2_v2` (v5 seeds, no settle):    peaked 44%, ended lower
- `segment_1to2_v3` (v5 seeds, with settle):  peaked 15%, collapsed to 1.9%
- **`segment_1to2_v4` (v9 seeds, with settle): 41%, monotonically rising**
- reference: shared-policy ablation C hit 76% on this segment

So per-segment training isn't broken — we just needed diverse seeds.
With v9's map-spread CP1 pool it learns cleanly, and 5M steps is
probably not enough (curve still rising).

**Quirk:** 3 of v9's 88 "CP1" seeds actually read as fruits_remaining=2
(CP2) after load+5-frame-settle — the agent drifted into a fruit sprite
during the settle. Those 3 seeds produced 8% of episodes with
start_level=2. Not a bug in anything we're measuring (the CP1→CP2
percentage looks only at start_level=1 rows) but worth knowing. Could
harden ``extract_seeds.py`` to re-validate states after the same
settle procedure ``train_segment.py`` uses, and drop ones whose CP
changes.

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

Approach 10 (Go-Explore from CP2) confirmed a hard fact: random-action
search does **not** cross CP2→CP3, even from 13 validated CP2 seeds in
5M steps. The "closed-loop PPO ↔ Go-Explore" plan in its original form
is therefore blocked — Go-Explore isn't going to hand us CP3 states for
the seeding step.

The live question is what alternative **does** work. The most promising
lead is one we already have evidence for but haven't followed up on:

**Per-segment training (one fresh agent per CP segment).**

Why per-segment is worth a serious look:

- Each agent has a single, narrow task: "start at CP_N, reach CP_N+1".
  Smaller credit-assignment problem than a shared policy that has to
  behave correctly at every floor.
- As Agent_N gets good, its own play produces more (and probably
  higher-quality) CP_N+1 saves. Those feed Agent_{N+1}'s starts.
- Doesn't need Go-Explore at all for the core loop — each agent
  generates the starts for the next one.
- The `fruit_princess_bonus` reward (already implemented) makes the
  last segment, CP4→princess, actually pay reward.

But we already *tried* per-segment training once (`segment_1to2`) and it
got 0% success on CP1→CP2 — the exact same segment the shared-policy
ablation C hit 76% on. Before we invest in per-segment training end to
end, we need to understand why `segment_1to2` failed.

### Investigation plan

1. **Validate v5's 100 CP1 states** through
   `python/retro_ai/training/state_validator.py`. `segment_1to2` used
   those states as its only starts — if a large fraction are frozen
   (the pattern we saw in `go_explore_fruit`), that alone could
   explain the 0% success. If so, segment training is fine; we just
   fed it bad data.
2. **Re-run `segment_1to2` with validated starts.** Point
   `train_segment.py` at the filtered checkpoints.pkl (either via
   `scripts/filter_archive.py` applied to a converted archive, or by
   extending the segment script to call the validator on load).
   5M steps, same config otherwise. If it now hits non-trivial
   success, per-segment training is viable. If it stays at 0%, the
   fresh-agent-per-segment design itself is broken and we stay with
   shared-policy.

This investigation is cheap (validator is seconds; a 5M segment run
is ~40 min) and gives a clean fork in the road.

### If per-segment works (segmented pipeline)

- **Agent_0→1**: train from CP0 starts. Already known to work in any
  PPO run (98% in ablation A).
- **Agent_1→2**: train from validated CP1 starts produced by Agent_0→1
  or (cheaper for now) by the existing Go-Explore archive. Re-uses the
  fix from step 2 above.
- **Agent_2→3**: train from validated CP2 starts. The CP2 starts we
  have today (29 from go_explore_fruit, 13 validated) are small but
  real; Agent_1→2's own play should produce more. This is the segment
  where everything has failed before; a dedicated per-segment agent
  is our best shot at cracking it.
- **Agent_3→4**: train from validated CP3 starts. v5 gave us 2 CP3
  states that validate today; Agent_2→3's play should produce more.
- **Agent_4→princess**: train from validated CP4 starts with
  `fruit_princess_bonus`. v5 has 1 validated CP4 state; same
  expectation (Agent_3→4's play generates more).

### If per-segment fails

Stay with shared-policy, and invest in whatever helps CP2→CP3 inside
the shared-policy frame. The main levers there:

- Multi-seed runs of the ablation, to confirm the 76%/3% gap replicates.
- Reward shaping that biases toward fruit 3 specifically (differential
  per-fruit reward, or a floor-reach bonus).
- Much longer runs — v5 reached CP3 twice and CP4 once in ~20M steps
  of reset-chain play, so raw time alone might push CP3 success off 0%.

### Followups not on the critical path

- **Quality-filter the curriculum's frontier selection** (B's pathology).
- **HUD-after-load render bug** (separate C++ session).
- **Characterize the "bonus=0 → lose-a-life after ~240 frames"
  behavior.**
- **Multi-seed confirmation of the ablation.**
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

### 13. Validator rebuild + C++ load_state bug  *(verified)*

Two bugs surfaced while investigating why per-segment CP2→CP3 training
produces so many "short failure" episodes.

#### 13.1. C++ bonus-stall detector leaked across load_state

`MO5Interface::load_state` restored emulator memory but did not reset
the reward-wrapper fields `previous_bonus_` / `bonus_stall_count_` /
`previous_lives_` / `previous_y_` / `previous_fruits_remaining_`.
Those counters accumulate across episodes, so loading a save-state
between episodes in a training env inherited whatever state the
wrapper had at the end of the previous episode.

Concrete demonstration (scripts/diagnose_load_done.py before the fix,
same save-state in three contexts):

| Context                                    | first_done_frame |
|--------------------------------------------|-----------------:|
| `reset(seed=0)` → load → probe             |                6 |
| `reset(seed=0)` → 1000 frames → load → probe|                1 |
| load → probe; then load again → probe      |          6, then 1 |

After fix (load_state now mirrors reset for the trackers): all three
contexts give first_done at frame 5, deterministic.

Why this mattered:
- Training envs that load checkpoints (curriculum, segment, go-explore)
  were getting `done=True` spuriously in the first ~10 frames of each
  new episode whenever the previous episode ended in a bonus stall
  (i.e. death) — which is most of them, at segments where the agent
  dies often. Episodes ended immediately, counted as failures, agent
  never had a chance to act.
- Explains some of the "short episode" noise we'd been glossing
  over — many of those episodes really were just dead-on-arrival
  because of stale trackers, not because the state was bad.

Commit: `40a8d9e` (`fix(mo5): reset per-episode death trackers on
load_state`). Regression test pinned in
`tests/python/test_mo5_load_state_resets_death_trackers.py`: loading
the same state in two different contexts must give the same
`first_done_frame`.

#### 13.2. Python validator was a separate death rule

The validator (from approach 10) was doing its own "is this state
alive?" check: bonus must drop by `min_drop=2` over `probe_frames=30`.
That rule is not the same as C++'s ("bonus unchanged for 10 consecutive
frames → `done=True`"). Pathology: a state where bonus ticks twice
right after load and then freezes meets the validator's drop=2 rule
(pass) and C++'s consecutive-10-unchanged rule (fail within ~19 frames).

Hard case we traced: a CP2 seed saved from segment_2to3_v2's
episodes.csv (`debug/short_cp2_episodes/episode_0_state.pkl`). Player
is mid-jump into a snowball; jump resolves post-load, snowball hits,
bonus freezes. Validator said "viable" (+2 drop over 30 frames), C++
fired done at frame 19. Training env saw the state pass validation,
loaded it, killed it on frame 19, counted it as a short failed
episode. 34% of segment_2to3_v2's episodes looked like this.

**Rebuild:** the validator now delegates to the env's `done` signal.
Load, 5 settle noops, then 120 probe noops; if env returns
`done=True` during the probe, reject. Same death rule as training,
by construction.

#### 13.3. Calibrating probe_frames from v9

To pick probe_frames, ran every cell in the v9 archive (432 cells)
through a 500-frame noop probe and recorded when done fires:

| first_done_frame bucket | cells | fraction |
|:-----------------------:|:-----:|:--------:|
| 1–10                    | 165   | 38%      |
| 11–20                   | 34    | 8%       |
| 21–30                   | 23    | 5%       |
| 31–60                   | 19    | 4%       |
| 61–120                  | 43    | 10%      |
| 121–200                 | 59    | 14%      |
| 201–300                 | 3     | 1%       |
| 301–500                 | 5     | 1%       |
| survived 500            | 81    | 19%      |

Not bimodal — the distribution has a long tail. But eyeballing the
first 8 cells in each bucket as videos (`dump_probe_videos.py`):

- **0–120**: unplayable. Agent is already-dying, landing on a
  snowball, or mid-fall with nowhere to land. One exception at
  `cp1_i113` in 11–20 (could have gone down a ladder). One at
  `cp2_i044` in 61–120 (done at frame 118, playable).
- **121–200 and beyond**: playable. A snowball is arriving but from
  a distance any trained policy would have time to respond to.

Probe cutoff: **120 frames**. Rejects 284/432 (66%) of v9 cells — all
confirmed unplayable under video review, with 1 known false-negative
(`cp2_i044`). Zero false-positives confirmed.

Tooling used to calibrate (kept in `scripts/`, indexed in
`scripts/README.md`):
- `probe_archive_done_frames.py` — the sweep that produced the
  table above.
- `dump_probe_frames.py` / `dump_probe_videos.py` — per-bucket PNG /
  MP4 dumps for eyeballing.

Commit: `abdeea0` (`refactor(state_validator): delegate to env done
signal`). Four callers updated to match: `extract_seeds.py`,
`filter_archive.py`, `train_checkpoint_curriculum.py`, `go_explore.py`.

#### 13.4. What this unblocks

Prior seeded-training runs (approach 12's `segment_1to2_v4`,
approach 11's follow-ups) used the old validator's
"viable" states. After 13.1+13.2 it's likely a meaningful fraction
of those states were actually unplayable — training saw short failed
episodes, aggregation numbers (CP1→CP2 = 41%) were diluted.

Next empirical steps will tell us how much these bugs were costing
us. The concrete experiments left open:

1. **Re-validate v9 archive** with the new validator (probe=120).
   Produces a smaller, higher-quality seed set.
2. **Re-run `segment_1to2_v4`** on the re-validated seeds. Compare
   CP1→CP2 against the previous 41%. Expect higher (both fewer
   short-dies from stale trackers AND fewer bad seeds).
3. **Run `segment_2to3` from scratch** with the re-validated seeds,
   this time with neither bug obscuring the signal.

### 14. segment_2to3_v3 results — the policy doesn't climb  *(verified)*

First clean CP2→CP3 run with both the C++ `load_state` bug and the
validator/C++ drift fixed (approach 13). Re-ran segment_2to3 from
scratch with:

- 48 re-validated CP2 seeds from v9 (down from 81 unfiltered), all
  video-confirmed playable under noop.
- 5M steps, 8 envs, `fruit_bonus` reward, `segment_1to2_v4`
  hyperparameters.
- Fresh PPO policy, no warm-start.

Config: `experiments/003-yeti/configs/segment_2to3_v3.yaml`. Run
directory: `output/mo5/yeti/training/segment_2to3_v3`. Commit
`a1bc35e`.

**Headline result: 3.88% CP2→CP3 over the whole run, 3.30% in the
last 20%.** Effectively the same as prior attempts (v1 ≈ v2 ≈ 3.7%).
The two fixes in approach 13 were real but did not move the needle
on this segment. The wall is in the learning problem, not in our
tooling.

Per-collected-set breakdown (last 20%, pure seeds only):

| fruits already collected | success rate  | n     |
|:------------------------:|:-------------:|:-----:|
| {1, 2}                   | 2.8%          | 3527  |
| {1, 3}                   | 2.3%          | 1906  |
| {2, 3}                   | 5.9%          | 1324  |
| {2, 4}                   | 0.0%          | 180   |
| {3, 4}                   | 4.0%          | 1721  |
| {1, 2, 4}                | 100%          | 181   |
| {2, 3, 4}                | 100%          | 192   |

The two 100% rows are the 2 "drift seeds" that pick up a 3rd fruit
in the 5-frame settle — they aren't real CP2 starts. Real CP2
subsets all hover 0-6%; no "easy" subset and no obvious "hard"
subset dominating the zero.

**Where the policy actually goes.** For each pure-CP2 episode,
compute the starting game-floor (from start_y) and the final
game-floor (from final_y, with y ≥ 30 to exclude the y-up-off-screen
death-animation frames), then look at `delta_floor`:

| delta_floor | n     |
|:-----------:|:-----:|
| -2          | 166   |
| -1          | 1611  |
| 0           | 6526  |
| +1          | 351   |
| +2          | 4     |

**75% of episodes die on the same floor they started.** 21% fall to
a lower floor. Only **4.1% ever climb a floor during the episode.**
Per start floor:

| start_floor (game) | % up  | % down | % same | CP3 rate |
|:------------------:|:-----:|:------:|:------:|:--------:|
| 1 (spawn)          | 9.6%  |  0.0%  | 90.4%  |  1.43%   |
| 2                  | 3.1%  | 40.7%  | 56.3%  |  2.60%   |
| 3                  | 0.5%  | 14.9%  | 84.6%  |  0.00%   |
| 4 (top)            | 0.7%  | 25.1%  | 74.2%  | 12.83%   |

Reading across:
- From spawn the agent rarely climbs (9.6%). When it does, it sometimes
  hits CP3, but 1.4% is barely above chance.
- From floor 2, it mostly falls off (40.7%) or dies in place (56%).
  Climbing to floor 3 almost never happens.
- From floor 3 it never climbs to floor 4. It either stays (85%) or
  falls (15%).
- **From floor 4 — the top — it hits CP3 12.8% of the time**, and that
  success is mostly "fall onto a remaining fruit", since only 0.7%
  move up (there's nowhere to go up to).

Anecdotal confirmation from 12 sample rollouts
(`scripts/rollout_policy_from_seeds.py` against v3's final_model):
policy mostly jumps left/right in place, eventually falls to a lower
floor or walks onto a snowball. It doesn't seek ladders. Same pattern
from diverse seed positions.

**What this rules in/out.**

- The per-segment pipeline itself is fine. Seeds load correctly, the
  agent gets meaningful reward for the 3 fruits it does collect, and
  the old flakiness (stale stall counter → spurious short episodes)
  is gone.
- The task is what's hard: from any CP2 state, the next fruit is
  usually a floor away and behind a ladder. PPO with `fruit_bonus`
  doesn't have an exploration signal that biases toward climbing —
  from the agent's view "go up a ladder" looks like "walk to a
  specific spot, press up, wait 30 frames, repeat". No reward
  gradient points there until the fruit is in reach.
- "Snowball jumping" that we previously worried about isn't even on
  the table yet — the policy isn't jumping over snowballs because
  it's not trying to cross them. It's standing around on its
  starting floor until a snowball finds it.

**Next lever.** Either much longer training (the v5 shared-policy
chain reached CP3 twice and CP4 once in ~20M combined steps, so the
signal is there but sparse), or a new mechanism that pushes the
policy to climb. Options on the table, in increasing invasiveness:

1. **Train longer.** Rerun segment_2to3_v3 for 20-40M steps. Cheap
   to set up; expensive in wall time. If CP3 rate creeps up
   monotonically, shaping isn't needed.
2. **Per-episode floor-novelty bonus.** Give +1 the first time the
   agent reaches a y-bucket it hasn't been to this episode. Less
   noisy than the old per-frame `delta(y)` reward (which caused
   jumping oscillation). Encourages "touch a new floor", doesn't
   reward repeated bouncing in place.
3. **Directional distance-to-remaining-fruit reward.** Read which
   fruits remain, compute (dx, dy) to the nearest one, reward
   reductions. Denser signal. Risk: could over-specialise or get
   stuck at a wall.
4. **Demonstrations / BC init.** Record expert play (or a scripted
   climbing policy), behaviour-clone into the PPO init. Bypasses
   the exploration problem for the cost of building a teacher.

Deciding which to try next.

### 14.1. "The policy climbs to its target floor, not further"

Follow-up question on approach 14: if CP1→CP2 works (41%), the
policy must be climbing — so why can't CP2→CP3 learn climbing too?
Hypothesis: the CP1→CP2 policy doesn't learn "climbing is good",
it learns "climb from floor 1 to floor 2 because fruit 2 is there".
Out of distribution, no climbing.

Testing that with the same per-floor analysis we ran on v3, but
against `segment_1to2_v4`'s episodes.csv (last 20%, 8762 CP1-start
episodes):

| start_floor (game) | % up   | % down | % same | CP2 rate |
|:------------------:|:------:|:------:|:------:|:--------:|
| 1 (spawn)          | 27.6%  |  0.0%  | 72.4%  | 76.70%   |
| 2                  |  2.6%  |  9.9%  | 87.5%  |  5.89%   |
| 3                  |  0.2%  | 66.0%  | 33.8%  |  4.20%   |
| 4                  |  0.0%  | 37.2%  | 62.8%  |  0.00%   |

Comparison against `segment_2to3_v3` (same table from approach 14):

| start_floor (game) | v4 up  | v3 up  |
|:------------------:|:------:|:------:|
| 1 (spawn)          | 27.6%  |  9.6%  |
| 2                  |  2.6%  |  3.1%  |
| 3                  |  0.2%  |  0.5%  |
| 4                  |  0.0%  |  0.7%  |

Reading:
- The CP1→CP2 policy can climb — but *only* from spawn (27.6%).
  It was trained on CP1 seeds spread across all floors of the map
  (25 on floor 1, 22 on floor 2, 17 on floor 3, 18 on floor 4), so
  this is not an out-of-distribution effect. It's "from floor 1
  the learned trajectory points up; from floor 2 onwards it
  doesn't".
- When started above floor 1, the CP1→CP2 policy behaves almost
  identically to the CP2→CP3 policy. They converge on the same
  "sit here or fall" pattern whenever started above floor 1.
- "Afraid to climb higher" fits the pattern: climbing is risky
  (fall + snowball), and the policy only adopts the risky action
  where a reliable reward gradient points above it. On floor 1
  that gradient exists (fruit 2 is right there). On floor 2 and
  above, the gradient is "try to find a remaining fruit somewhere
  on this floor" which mostly fails and never teaches climbing.

**Implication for next steps.** The options from approach 14 map
onto this more precisely now:

1. **Train longer.** Gives PPO more chances to stumble onto higher
   floors via random exploration. Evidence from curriculum_v5 says
   it's possible (2 CP3 and 1 CP4 states found in 20M reset-chain
   steps), but vanishingly rare.
2. **Per-episode floor-novelty bonus.** Directly rewards "climb to
   a floor you haven't been to yet this episode". Addresses the
   specific deficiency: the agent doesn't know climbing-above-fruit
   is valuable.
3. **Distance-to-remaining-fruit.** Dense gradient toward the
   specific remaining target. Solves the "which way should I
   climb?" question. Risk: fruit 1/2 are on low floors; if any
   seed has those remaining, the gradient points DOWN, not up.
4. **BC init.** Demonstrations teach the climbing skill directly.
