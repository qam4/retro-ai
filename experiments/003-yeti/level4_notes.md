# Yeti Level 4 — run history and wall diagnosis

L4 had no notes file until now. Its narrative lived only in the config headers
(`configs/yeti_curriculum_l4_v1..v5*.yaml`) and in run logs, which is why probe
results kept being re-derived or lost. Record findings HERE.

## Status

Princess: **0**, across v1..v6 (6 runs, ~85M steps). The from-reset chain is now
solid to rung 10 (`Low1`, floor 12) and stops dead there.

| run | lever | outcome |
|---|---|---|
| v1 | cold, phase-2 settings | 0 fruits in 3000 reset episodes (route unwinnable: no phase gate) |
| v2 | `waypoints_after_fruit` phase gate | fruit 93%, wall at `Step` |
| v3 | warm-start from v2 | pools poisoned (inherited `reached_next` credit) |
| v4 | `admit_requires_survival` | deepest rung 12; `Step` reach 0.85, `Low1_launch` 0.54 (but see the tol-6 caveat: that 0.54 was counting a stalled climb as an arrival) |
| v5 | phase-1 temperature (`n_steps` 512→16, no `target_kl`) | regressed everything; deepest rung 10 |
| v6 | **reward-milestone anchor fix** (`Fr1`, `Rope1`; 3 launch pads dropped) | **best run.** rung-10 from-reset reach 0.06 → 0.66 EMA (peak 0.86 @8.76M); `Low1` touches 1,247 → 23,730; deepest rung 12 |

### v6 champion, measured at n=300 (2026-08-24)

`best/best_model.zip` = `model_14600000_steps.zip`, 300 stochastic from-reset
episodes, level-4 geometry, stall 40 / max_steps 1500:

```
reached >= rung 10:  254/300  (84.7%)
reached >= rung 11:    0/300  ( 0.0%)     <-- the wall, exactly zero
mean rung: 9.38 / 13
1 fruit:  289/300 (96.3%)
princess:   0/300 ( 0.0%)
```

Two things to take from this. First, rung 10 at **84.7%** is the real v6 gain — v4's
from-reset reach for the same rung was 0.06. The anchor fix worked. Second, rung 11
is not "rare", it is **0/300**: a hard barrier, not a low-probability crossing. Do
not model it as something more training time will smooth out.

> **SUPERSEDED IN PART (2026-09-11).** The "hard barrier, not a low-probability
> crossing" reading no longer holds as a general claim. It is still exactly what v6's
> champion measures (0/300, and re-swept on current code its champion is again 0/300).
> But v13's champion reaches **rung 12 in 1/300** from reset — it credits `Low2` and
> `Lhi_down_bot`, i.e. it crosses rope 2. So rung 11 is a ~0.3% crossing for that
> policy, not a wall at zero — confirmed independently at 3/1071 in a from-reset hunt
> (pooled 4/1371 = 0.29%), with all three crossings caught on video and using pose 15,
> the leftward rope carry. See "v13 RESULT" and the pose-15 section below. Rung 12 had
> already been touched during v6's TRAINING (see the v6 row in the table above); what is
> new is seeing it from reset, repeatedly, and knowing the manoeuvre it uses.

Note the 12-episode sweep score for this same snapshot was `rung 10.00`, versus
9.38 at n=300. 12 episodes is a *trigger* resolution, not a measurement; it cannot
separate the top snapshots from each other (eight of them scored 9.4–10.0) and it
cannot tell 0% from ~4%. Always re-eval a champion at 200–300 before building on it.

### `keep_best_sweep` scoring — was broken, now fixed

Historic `best/best_model.zip` under v4 and v5 is **meaningless**: the sweep scored
`princess + 1e-3 * reach_top`, and on a one-fruit level `reach_top` = P(collect the
fruit), which saturates at 1.0 for most snapshots. Ties broke to the EARLIEST
snapshot, so v4 "kept" 100k and v5 kept 1M — both near-untrained. Eval was blind to
the 13 route rungs between fruit and princess.

Fixed by tracking route depth through the rollout harness
(`yeti_rollout.rollout_episode(track_waypoints=...)` → `max_rung`, `n_rungs`,
`reached_points`) and scoring
`princess + 1e-2 * (mean_rung / n_rungs) + 1e-5 * reach_top`.

Re-sweeping v6's 150 snapshots produced **94 distinct scores** where the old
formula produced 1–2. Ranking (12 eps each):

| step | mean rung |
|---|---|
| 14,600,000 | 10.00 |
| 2,600,000 | 9.92 |
| 13,400,000 | 9.92 |
| 1,200,000 | 9.83 |
| 8,600,000 | 9.83 |
| 5,400,000 | 9.75 |

mean 6.37, median 7.88; 8/150 at rung >= 9.5; **10/150 below rung 1.0** (collapsed).

The good snapshots are **scattered** across the whole run (1.2M, 2.6M, 5.4M, 8.6M,
13.4M, 14.6M), not clustered at the end. So v6 is not a policy that improves and
then degrades — it oscillates between the rung-10 ceiling and collapse for all 15M.
That is the gate instability (see "The gate locks out full pools at the wall"), and
the frozen-snapshot evals independently agree with the training EMAs about which
phases are the bad ones.

## v5 was not a valid test of its own hypothesis

The hypothesis (see its config header) was that L4 had never received the
high-temperature phase-1 exploration that unlocked L1, and that supplying it would
jump the plateau. Two reasons the run cannot answer that:

1. **Confounded staging.** v4 was staged weights-only (`l4_v3_weights_only/`
   contains just `final_model.zip`) and rebuilt pools from empty. v5 was staged
   with weights **and** v4's `checkpoints.pkl`. So temperature and initial pools
   both changed.
2. **Feedback through the gate.** High-temperature updates degraded from-reset
   reach, which shut `gate_waypoints` on the deep points, which cut deep starts
   4-5x (`Low1_launch` 4488 → 917, `Step` 3689 → 1027). Temperature was not an
   independent variable.

## The gate locks out full pools at the wall

Both v4 and v5 set `gate_waypoints: true`, `reach_threshold: 0.15`. A waypoint is
eligible as a start only if `wp_reach_ema >= 0.15`. Measured (`episodes.csv`,
`start_key`), across BOTH runs:

```
Low1          pool 100   from-reset reach 0.00 (best ever 0.02 v4 / 0.06 v5)   starts 0
Low2_launch   pool 100   reach 0.00                                            starts 0
Lhi_up_top / Hi1_launch  pools >0, reach 0.00                                  starts 0
```

100 admitted, survival-gated seeds at each of two points, never once loaded. The
gate asks for 15% from-reset reach before permitting practice at the very point
whose practice would produce that reach. Same mechanism L3 diagnosed (v15 notes,
"gate oscillation") and prescribed hysteresis for; never implemented. On L4 it
deadlocks outright rather than oscillating, because deep reach never approaches
0.15.

Note `Low1_launch` itself WAS heavily practised (4488 starts in v4) — see below
for why that practice was worth less than it looks.

## WHERE THE FROM-RESET CHAIN ACTUALLY BREAKS (read this before picking a wall)

v4's route table reads, in route order:

```
Step_launch   0.89
Step          0.85
Lclimb3_top   0.06      <- looks like the collapse
Low1_launch   0.54      <- but this is FURTHER along and HIGHER
Low1          0.00      <- the real break
Low2_launch   0.00      (650 captures, all from seeded episodes)
Low2          0.00      (1 capture, ever)
```

Reach is non-monotonic along a linear route, so one of those numbers is wrong.
**MEASURED 2026-08-24: the agent stalls 4px below the platform. `Low1_launch` counts
that stall as an arrival.**

`debug/l4_climb_height.py --model 14000000 --episodes 30` measures the MINIMUM y
(highest point) each from-reset episode reaches. A minimum over a whole episode needs
no tolerance window, so unlike every waypoint-box number it cannot be an artifact of
the 4-frame detection stride:

```
min-y per episode          y82 n=15   y84 n=12   y108/140/172 n=1 each
topped the ladder (<=78)   0/30
on the ladder column       30/30 episodes, median 38 steps there
best height ON the ladder  y82 n=15  y86 n=7  y90 n=3  y92 n=1  y146 n=3  y172 n=1
```

27/30 episodes climb `Lclimb3` to y82-84 and stop. **NONE of 30 ever reaches y78, the
f11 standing height.** The agent is on the ladder column in every single episode and
spends ~38 steps there.

That resolves the non-monotonicity, and the honest number is the LOW one:

* `Lclimb3_top` needs y in [76,80]. The agent never gets there. **0.03 is correct.**
* `Low1_launch` accepts y in [72,84]. The agent stalls at y82, inside the box, so it
  fires. **0.40-0.54 is the stall being scored as an arrival.**

It also re-reads the pool composition recorded under Q3 below: the 43/100
`Low1_launch` seeds at y82 in ladder pose are not "contamination", **they are the
stall point itself**. v4 spent 4488 practice episodes at that waypoint, 43% of them
starting from exactly where the agent gets stuck, and the other 57% (at y78) from
states it only ever occupies when seeded there.

**THE WALL IS THE LAST STEP OF THE `Lclimb3` CLIMB: y82 -> y78.** Every other L4
finding in this file — f11 -> f12, ROPE 2, pool purity, seed determinism — is above
it and was measured on a region the agent does not reach from reset.

Corroborating, from `debug/l4_f11_arrival_vs_pool.py --model 14000000 --episodes 60
--settle 1` (v4's 14M snapshot, whose route table reads `Low1_launch` 0.40):

```
from-reset: STOOD ON f11 in 1/60  (2%)
  the single arrival was x_ram 66, y78, pose 8  -- i.e. ON THE LADDER at platform
  height, at the ladder's x; not walking on the platform
  of that 1, episode later stood on f12:  0
  of that 1, Low1 reach box fired:        1     -> box fired without a landing
scripted winnability   arrivals 1/1 (n=1, meaningless)   pool 11/20 (55%)
```

2% agrees with `Lclimb3_top`'s 0.03-0.06 and refutes `Low1_launch`'s 0.40-0.54.
The mechanism is box width, in the opposite direction to what this file previously
claimed:

* `Lclimb3_top` is a LADDER waypoint -> tol 2 -> box x_ram 64..68, y 76..80.
  That is the platform-standing position. **Honest.**
* `Low1_launch` is a JUMP waypoint -> tol 6 -> box x_ram 54..66, y **72..84**,
  pose set including ladder. The `Lclimb3` ladder is at x_ram 66, so a state
  PART-WAY UP the ladder at y82 sits inside this box. **It fires without the agent
  ever reaching floor 11.**

So the agent gets part-way up the ladder in ~40-54% of from-reset episodes and onto
the platform in ~2-3%.

**The honest ordering of L4's walls:**

1. **f10 (`Step`) -> f11, the `Lclimb3` ladder climb, is THE wall.** Reach falls
   0.85 -> 0.02. Everything above it — the `Low1_launch`, `Low1` and `Low2_launch`
   pools, 100 states each — was built ENTIRELY from seeded episodes. That is why
   4488 practice episodes at `Low1_launch` never transferred: they drill a region
   the agent cannot reach from reset.
2. f11 -> f12 (`Low1`) and f12 -> f13 (ROPE 2) are LATER walls. Neither can be
   measured from reset until (1) is solved. Work on them is premature.

Corollary: the `Low1_launch` pool being 43% mid-ladder states (see Q3 below) is not
merely a capture-tolerance bug, it is a SYMPTOM — at that waypoint the agent is
usually on the ladder, because it hardly ever makes the platform.

Two earlier claims in this file were wrong and are retracted: that `Lclimb3_top`
0.06 was a detection artifact, and that f11 -> f12 was the wall. A session's worth of
probing (Q0, Q3, ROPE 2) was spent above the real break.

The `Low1` box also over-fires: it registered a reach in an episode that never stood
on f12. Both boxes need auditing, not just `Low1_launch`.

## THE WALLS (measured 2026-08-24)

Ran `scripts/mo5/yeti/diag/l4_low_route_probe.py --n 20 --skip-poses` and
`debug/l4_low1_jump_bruteforce.py --pool Low1_launch --n-seeds 12` against v4's
pools. Both scripts already existed (written 2026-08-19/20) with explicit decision
rules; their results had never been recorded.

Geometry: `f11` P12 y78 x_ram 60-78 → `f12` P11 y70 x_ram 44-56 → `f13` P10 y70
x_ram 0-30.

### Wall A — f11 → f12 (`Low1`), a plain jump: FEASIBLE, pool half-unusable

```
scripted walk-left-then-JUMP-LEFT      3/20  (15%)
best single fixed phase (12/24)        4/20  (20%)
solvable by SOME phase                10/20  (50%)
NOOP lifetime  min 29  median 49  max 81   survived 150 frames: 0/20
```

At training granularity (frame_skip 4, `l4_low1_jump_bruteforce`, 12 seeds):
best single plan 4/12 (33%), solved by some plan 5/12 (42%). Broken down by the
seed's own state — the decisive cut:

```
x_ram 62 y78 pose 4   1/1 solvable
x_ram 65 y78 pose 4   2/2 solvable
x_ram 65 y78 pose 5   2/2 solvable
x_ram 66 y82 pose 8   0/7 solvable   <- LADDER pose, BELOW f11
```

**Every seed genuinely standing on f11 is solvable (5/5). Every seed in ladder pose
at y82 is not (0/7).** 7 of the 12 sampled seeds are the latter, consistent with the
figure recorded in `l4_low1_jump_bruteforce.py`'s docstring from an earlier
measurement of the whole pool (45/100 in ladder pose 8, 43 of them at y82). The jump
waypoint tolerance is 6, so the `Low1_launch` box at (60, 78) spans y 72..84 and
admits x_ram 66 / y82 — a mid-ladder position 4px below the platform — and pose 8
is in `SEED_POSES`. So the "jump-off pad" pool is ~half mid-climb states from which
the intended manoeuvre is impossible without first finishing the climb, under a
~49-frame death clock (nothing survives standing still).

=> Wall A is **not a hard skill**. It is a capture-tolerance defect: half the
practice at this doorstep is practice at an impossible task. Fix is in capture, not
in the policy.

### Wall B — f12 → f13 (`Low2`), a ROPE modelled as a jump: 0/20, and the platform is safe

```
scripted walk-left-then-JUMP-LEFT      0/20  (0%)   16/20 DIED, 4 timeout
best single fixed phase                0/20  (0%)
solvable by SOME phase                 0/20  (0%)
NOOP lifetime  min 150  median 150  max 150   survived 150 frames: 20/20
```

Zero hazard pressure (every seed survives 150 frames of NOOP) and zero success
across every phase, with 80% of attempts fatal. This is not phase sensitivity —
it is the wrong manoeuvre. `LEVEL4` models f12→f13 as a plain `jump_edge`.

**CAVEAT, do not over-read:** `plan_rope` hardcodes `period=24` and sweeps
`i % 24`, so "all 24 phases" only covers the phase space if the rope's period
divides 24. Rope 1's carry lasts ~30+ frames in trace. Wall B is therefore
established as *not a jump* and *not hazard-limited*; it is NOT established as
infeasible. Re-probe with the measured rope period before concluding that.

**Correction to "the platform is safe": the LANDING platform carries snowballs.**
The NOOP-survival probe above measures the seed's *departure* platform, which is
why it read as hazard-free. `yeti_map` has:

```python
SNOWBALLS = {"P7": (28, 29), "P10": (46, 53)}
```

`P10` is **floor 13 — rope-2's landing platform** — and it spans cols 0–15, i.e.
px 0–127, the platform's full width. Visually confirmed in
`experiments/003-yeti/evidence/shots/l4_f12_launch_44_70.png`.

This is the asymmetry that explains why rope 1 is easy (reach 0.82 in v4) and rope 2
is a 0/300 wall: **rope 1's landing platform (P17, floor 7) has no snowball; rope 2's
does.** So rope 2 is not just a harder rope, it is a rope whose landing must be timed
against a hazard that traverses the entire target platform. Any fix that treats it as
a pure locomotion problem is attacking the wrong constraint.

Supporting evidence from v6 (`experiments/003-yeti/evidence/l4_v6_rope2_videos/`, 6 episodes): 5/6 end in
death pose 11 (FALL); ep3 hits max_steps stuck at (44, 74) in **pose 17** on the
launch pad, never departing. Training counters at the same rung: arrives 61%,
`prog` 0.00, **5,559 precarious rejections**.

## L4 uses sprite poses the codebase does not know about

`debug/l4_rope_pose_trace.py --run <v4> --n 8`, replaying v4's policy from pools:

```
ROPE 1  f6 -> f7    policy crossed 8/8 (100%)   non-surface poses {9: 213, 14: 41}
                    moved sideways in pose      {9: 130, 14: 26}
SPRING  f8 -> f9    policy crossed 0/8          {9: 194, 16: 139, 11: 104}
ROPE 2  f12 -> f13  policy crossed 0/8          {11: 122, 10: 84, 17: 54, 9: 44, 6: 1}
   (the wall)                                   pose 17 never moved sideways
```

Poses **6, 14, 16, 17** appear nowhere in the pose table, in `SURFACE_POSES`
(`{0,1,2,3,4,5,8}`), or in `SEED_POSES` (`SURFACE_POSES | {13}`, where 13 was added
for L3's escalator ride). Consequences:

* A **rope carry is pose 14** and moves the agent sideways — a controlled
  traversal, exactly like L3's pose-13 escalator ride. It is currently classified
  airborne, so shaping freezes across every rope and **no seed can be captured
  mid-carry**. There are no intermediate seeds on any rope.
* The **spring is pose 16**; the f12→f13 crossing shows **pose 17** with no lateral
  motion plus heavy pose 11 (fall).

**CORRECTION (2026-08-24). Do not conclude from this that the rope modelling is what
blocks rope 2.** An earlier version of this file ranked "ropes modelled as plain
jump_edges + pose 14 frozen" as L4's top blocker. That is refuted by L4's own data:

* L4 has TWO ropes, per the authored route in `scripts/mo5/yeti/diag/l4_route_check.py` —
  `("rope", 92, 93)` = ROPE 1 (f6->f7), and `("rope", 54, 53)` = ROPE 2 (f12->f13,
  the wall). Plus `("spring", 74, 75)`.
* `jump_edges` contains BOTH `(6,7)` and `(12,13)`, and pose 14 is outside
  `SURFACE_POSES` for both. So ROPE 1 carries the identical model defect and the
  identical shaping freeze — and is crossed **0.90 from reset**.

A defect present at a crossing the agent makes 90% of the time cannot be the reason
another crossing is never made. The pose facts above stand as facts; their causal
weight does not.

**Poses are FACING-PAIRED, so 14 and 17 are probably the same state, two directions.**
The documented table is already paired: 0-3 walk-right / 4-5 walk-left, 9 jump-right /
10 jump-left. ROPE 1 is entered jumping RIGHT and shows 9 (213) + 14 (41), no 17.
ROPE 2 is entered jumping LEFT and shows 10 (84) + 17 (54), no 14. Read as a pair:
**14 = carried facing right, 17 = carried facing left.**

That corrects a claim made earlier in this session: the agent **does** enter a
rope-carry pose on ROPE 2 (17, fifty-four times), so "it never engages the rope" was
wrong.

The remaining asymmetry is narrower and is the thing to explain: pose 14 on ROPE 1
moved the agent sideways 26 of 41 occurrences, while pose 17 on ROPE 2 moved it
sideways **zero** times. So on ROPE 2 the agent reaches a carry pose but is not
carried. Whether that is a different mechanic, a wrong entry point, or simply the
agent grabbing and letting go, is not established — and per the section above, it is
a LATER wall and should not be worked before f11 -> f12.

Also note the scripted 0/20 does NOT license a mis-modelling claim on its own: the
`plan_rope` sweep is `i % 24`, so a rope whose period does not divide 24 is never
jumped at the right moment. It means "this script failed", not "this mechanic is
wrong".

## THE REACH TEST IS NOW ONE CODE PATH (done 2026-08-24) — and what is still broken

Two consumers decide "has the agent reached this waypoint", and they had separate
implementations:

* curriculum (`train_checkpoint_curriculum.py`) — reach EMAs, capture, seeding
* reward (`rewards.py`) — milestone marking inside the PBRS potential

They use the **same anchor** (verified: positions differ for 0 targets on L3 and L4)
and **different tolerances**: the curriculum passes 2 for ladder waypoints and 6 for
jump waypoints, the reward always passes `waypoint_reward_tol` = 2. One anchor, two
box sizes, two answers.

**The bug this produced.** L4 `Fr1` is anchored at x_ram 60 — floor 3's left
extremity. Measured from reset (10 episodes, 120 grounded steps on that floor) the
agent occupies only x_ram 64..68 there. So:

```
reward box   (tol 2)  58..62   ->  0 hits, milestone NEVER marked
curriculum   (tol 6)  54..66   ->  fires, reach reads ~0.94
```

Because it is never marked it stays in the active set and the potential sums distance
to it for the whole episode (`J2_3_b` is not in `waypoints_after_fruit`, so every step
of every episode). Measured magnitude on floor 3: the milestone term contributes
**+0.48/step** of leftward pull versus **+0.04/step** from the base route potential —
12x — and it keeps paying past the safe 3->2 takeoff at x_ram 63-64 to 62 and then 61,
where the agent dies (6/6 scripted trials, death at x_ram 61 step 4). `J6_7_b` (Rope1)
has the same defect: box 22..26, agent occupies 27 and 34.

What is NOT established: the training cost. PBRS is policy-invariant in theory, so
this distorts the learning gradient rather than the optimum.

**What was changed (pure refactor, no behaviour change).**
`retro_ai.training.targets.within_tol()` is now the single comparison, with
`Target.reached(x, y, tol_x, tol_y=None)` delegating to it, and both call sites route
through it. Tolerance stays a caller argument precisely because the two callers still
pass different values — the refactor stops the comparison drifting, it does NOT fix
the values. `Target.reached` also raises for non-positional targets (fruits are RAM
events, the princess is a flag) so a second, wrong detector cannot be invented by
accident. Equivalence to the old inline logic is pinned exhaustively in
`tests/python/test_reach_test_shared.py`, which also pins the `Fr1` defect as a
regression test. Verified: 432 tests pass; a warm-started 40k L3 run marks reaches
normally (`reset_reach` 0.86 on rungs 1-6, `route[8]: 6/19`).

**Validation, done 2026-08-24: `scripts/mo5/yeti/diag/yeti_validate_targets.py`.** Two tiers, because
they catch different failures:

* STATIC (no emulator): does the anchor's box contain any position that resolves to
  the target's own floor, via the same `agent_floor_from_pixel_xy` the reward uses?
  Done in pixels to dodge the tile-vs-agent conversion trap.
* MEASURED (`--measure`): does the box ever contain a position the agent is actually
  grounded at?

An "anchor at a platform extremity" static warning was tried and REMOVED: it fires for
all 24 L4 jump waypoints, because placing arrivals and launch pads on edges is what
`jump_waypoints` does. It cannot separate `Fr1` (never marks) from `Spring` (marks
18x) — the difference is where the agent goes, not geometry. Static alone finds nothing
on L4; the measured tier is the one that produces a verdict.

**L4 result — 6 targets never mark** (v4 snapshots 14M + 10M, 8 episodes each):

```
Lclimb3_top     reward=N curric=N    <- the STALL, not misplacement (agent sits at y82,
                                        box needs y76..80). Already documented above.
Fr1             reward=N curric=Y
Rope1           reward=N curric=Y
Spring_launch   reward=N curric=Y
Step_launch     reward=N curric=Y
Low1_launch     reward=N curric=Y
```

`Fr2`, `Fr2_launch`, `Fr1_launch`, `Spring`, `Step`, `Rope1_launch` and every ladder
anchor except `Lclimb3_top` mark fine. So five anchors are misplaced, and three of them
(`Spring_launch`, `Step_launch`, `Low1_launch`) were not on the earlier list.

**L3 is affected too — see `level3_notes.md`.** `Lesc_top` fails the STATIC check (0
standable positions: anchored at x_ram 33 on a floor whose walkable run ends at 26, so
7 units off the platform), and `A1_launch` never marks under either detector. That
changes the framing: this is not a defect that only exists on the level with nothing to
lose. L3's A2..A5 anchors are untested rather than clean — those floors were never
visited from reset, so they need seeding from their own pools.

## ANCHOR FIX APPLIED TO L4 (2026-08-24) — this is v6's one lever

Five changes, all L4-only. Two new optional `LevelMap` fields carry them, so L1/L2/L3
are untouched by construction (`jump_waypoint_pos`, `jump_waypoint_skip`).

**Two anchors moved to measured positions:**

```
Fr1     x_ram 60 -> 64   floor 3's landing in all 15 observed 2->3 crossings.
                         60 is the platform's left extremity; walking there is fatal
                         at 61 (6/6 scripted trials).
Rope1   x_ram 24 -> 27   the rope landing on floor 7, pose 1, at that floor's standing
                         y. VISUALLY CONFIRMED by the user from
                         debug/shots/f7_x27_y118_pose1.png.
```

The tool's first proposal for `Rope1` was (34, 110), which was **rejected**: that
position reads pose 8 — the agent CLIMBING the ladder off floor 7, a different event.
The proposer picks the busiest position without checking it is at the floor's standing
y, so it can propose a transient. Known limitation, not fixed.

**Three redundant launch pads dropped.** Each shared a platform with a waypoint that
already marks correctly, so it was a second, wider, misplaced box for one traversal:

```
Spring_launch  floor 8   Lclimb2_top is at (34,94) -- the very position the measured
                         proposal for Spring_launch resolved to. Same point, two names.
Step_launch    floor 9   Spring marks fine there.
Low1_launch    floor 11  Lclimb3_top is exact (dx=0 dy=0 whenever reached); the launch
                         pad's tol-6 box also reported the y82 ladder stall, 24 px
                         away, as an arrival -- the 0.36-vs-0.03 discrepancy.
```

L4 route points: 33 -> 30.

**Two bugs found while applying it, both by testing rather than reasoning:**

1. The reward read its marking position from the GRAPH NODE, not the waypoint, so
   moving an anchor did nothing to it. It now takes the position from the shared
   `Target` — a no-op wherever the two already agreed, which was everywhere before
   these overrides existed. Distances are still computed from the graph ident, so
   shaping geometry is unchanged.
2. `build_targets` linked `Fr1` to its graph name `J2_3_b` by comparing COORDINATES.
   Moving the anchor severed the link and **L4's mandatory set silently dropped from
   14 to 12**, shortening the progress ladder with no error. The alias is now derived
   from the jump-edge structure (`J{fa}_{fb}_b` is the arrival, `_a` the launch pad),
   so it survives anchor moves. Pinned by a test asserting 14 on both L3 and L4.

**Verification.** 434 tests pass. `scripts/mo5/yeti/diag/yeti_validate_targets.py --level 4 --measure`
now reports "no unmarkable targets found" where it previously found five. L3 marking
spot-checked unchanged at `Lsc4_top`, `Lsc1_top`, `A1`.

**Remaining on L3** — see `level3_notes.md`: `Lesc_top` is anchored 7 units off its
platform (fails the static check) and `A1_launch` never marks. Not fixed, because it
changes L3's reward and L3 is at 80.7%.

## !! L3 WARNING — fixing anchors or unifying tolerances MEANS RETRAINING L3 !!

L1 and L2 have no jump waypoints at all (`jump_waypoints` returns `{}` for both), so
they are unaffected by any anchor or tolerance change. **L3 has the A1..A5 ascent, so
it is affected.** Consequences, before anyone edits a tolerance:

* It changes L3's reward. Per reward-shaping pitfall #4 in `003-yeti-training.md`, the
  critic is fit to the old reward's scale, so **every L3 champion stops being a valid
  warm-start** and L3 has to be retrained from scratch (or from phase-1) to be
  comparable.
* It changes L3's route-table numbers, so historical reach figures are not comparable
  across the change.
* L3 is at 80.7% princess from reset — the best-performing level. Do not spend it to
  fix a defect measured on L4 until L4 shows the fix does something.

Also unresolved and worth checking before that decision: whether L3's A1..A5 anchors
have the same edge-placement defect. If they do, it is a candidate explanation for
L3's distributed ~1.6%-per-segment attrition, which would change the priority order.

## WAYPOINT TOLERANCE — how it should be chosen (measured 2026-08-24)

Full justification now lives on `CurriculumConfig.waypoint_tolerance` in
`python/retro_ai/training/run_config.py`. Summary and the raw data:

**The x byte moves in 4 px steps.** Holding RIGHT and logging every EMULATOR frame:
`x_ram` goes 42,43,44,45,... i.e. `px = x_ram*4 + 8` advances 4 px, changing every
~5-6 frames, with the walk pose cycling 0->1->2->3 in lockstep. So a walk step IS
4 px and there is no finer horizontal resolution.

**One `tol` is applied to both axes, in different units.** x is in 4 px RAM units, y
is in pixels, so `waypoint_tolerance: 2` = **+-8 px horizontally, +-2 px vertically**.

**Measured |delta| per GYM STEP (frame_skip 4), 2472 steps of real play:**

```
|d x_ram|   0: 58.2%   1: 39.5%   2: 2.2%   3: 0.1%
|d y_px|    0: 58.1%   2:  8.5%   4: 32.6%  6: 0.5%   8: 0.2%

by pose:  walk 0,1,4,5  dy {0,2,4}      walk 2,3  dy {0}      ladder 8  dy {0,4}
          jump 9/10, fall 11, rope 14 {0,8}, spring 16 {0,4,6}  -- ALL AIRBORNE
```

Detection is pose-gated (`pose in SEED_POSES`), so the airborne rows can never be
marked. **The largest DETECTABLE dy is 4** (the ladder climb).

**Defensible values: `tol_x = 1`, `tol_y = 2`.**
* x: walking visits every `x_ram` value, so a target cannot be skipped; tol_x 0
  suffices for a rest point and 1 (+-4 px, one step) is insurance. Current 2 is 2x
  too generous.
* y: y is always even and a climb advances 4 px per step, so the sampled lattice can
  sit 2 px off the target — the traced climb ran 94,90,86,82,78 and hit y78 exactly,
  but a phase-shifted climb samples 80 then 76 and misses. tol_y 2 catches either
  phase and covers the largest detectable dy. tol_y 1 would fail.

Splitting the field per-axis changes behaviour on every level that uses waypoints, so
it is documented, not silently applied.

**Frame skip only bites for points passed AT SPEED** — rope carry moves y 8 px per
gym step, falls and spring 6, all bigger than a +-2 px window. Those poses are
airborne so it is a non-issue today, but it returns the moment a waypoint is placed
on a carried/moving segment, or a moving pose is added to `SEED_POSES` (as L3 did
with escalator ride 13).

Visual: `experiments/003-yeti/evidence/l4_climb_visual/rest_at_66_82_zoom.png` (via `debug/l4_climb_shot.py`)
draws the current tol-2 box and the proposed tol_x1/tol_y2 box on a real frame, with
the RAM reference point crosshaired. Conventions verified in that render: `x_ram*4+8`
is the sprite CENTRE (the crosshair lands on the ladder's `centre_x`), and
`floor_top_y` is a STANDING-SPRITE-TOP level, not the platform surface — the surface
is ~18 px below it. Sprite height in the drawing is display-only; nothing in
detection or the reward uses a sprite extent, only the single `(x_ram, y)` point.

## Q3 ANSWERED (2026-08-24) — the box costs real practice, but is not the whole wall

`debug/l4_low1_pool_purity.py`, whole `Low1_launch` pool (100 states) from v4,
classified on load, then v4's 14M snapshot run per class (1 episode/seed, 40 steps):

```
composition          clean_f11   57/100  (57%)      ladder_y82   43/100  (43%)
NOOP survival        clean  median 10 gym steps     ladder  median 15
  (cap 40)           survived 40: 0/57              survived 40: 0/43
policy LANDED f12    clean  20/57  (35%)            ladder   2/43  (5%)
                     POOL TOTAL 22/100
```

Against the readings fixed in the script's docstring:

* **Contamination is real and costly.** 43% of the pool is mid-ladder, and the
  policy converts 7x worse from it (5% vs 35%). Those 43% are states where the
  scripted move is 0/7 feasible. So a large share of the 4488 practice episodes v4
  spent at this doorstep was practice on a near-impossible task.
* **The clean subset is big enough to filter to** (57/100), so the capture RULE does
  not have to be redesigned — excluding pose 8 / y < 76 from the `Low1_launch` box
  would leave a usable pool.
* **But tightening the box will not fix f11.** Clean seeds are 5/5 *scriptable*
  (frame_skip 4, n=12) and the policy still lands only 35% from them. So there is a
  genuine learning shortfall from good states, not merely bad seeds.

One property of f11 worth separating out: **nothing in this pool survives doing
nothing** — 0/100 last 40 gym steps, median 10 (clean) and 15 (ladder). Contrast
`Low2_launch`, where 20/20 survive 150 frames. f11 offers no safe dwell, so there is
no "wait for a gap" option of the kind that cracked `Step`. Whether the remaining
65% failure from clean seeds is policy skill or kangaroo phase baked into each seed
is exactly what `scripts/mo5/yeti/diag/l4_seed_determinism.py` was written to separate, and it has
not been run.

Note: `debug/l4_low1_jump_bruteforce.py --n-seeds 100` was killed by the sidecar's
hang detector at 6m (it prints nothing until the grid finishes — a false positive,
not a crash). Not re-run, because the purity probe measured composition exactly and
the n=12 grid already gave per-class feasibility.

## Q0 ANSWERED (2026-08-24) — f11 is NOT phase-determined; the pool is real practice

`scripts/mo5/yeti/diag/l4_seed_determinism.py --pools Low1_launch,Lclimb3_top --clean-only
--model 14000000 --n 25 --repeats 10` (v4 pools; two small edits made to the script:
a `--model` option, because it hardcoded the degraded `final_model.zip` and a bad
policy loses everywhere and fakes a phase-determined verdict; and `--clean-only`,
to drop the mid-ladder captures that cannot answer the question at all).

```
Low1_launch  (clean filter kept 57/100, sampled 25 x 10 repeats)
  MIXED 15   ALWAYS-win 1   ALWAYS-lose 9      mean success 0.29
  between-seed var 0.1087   within-seed var 0.0980   -> 53% seed / 47% policy

Lclimb3_top  (clean filter kept 100/100, sampled 25 x 10)
  MIXED  9   ALWAYS-win 0   ALWAYS-lose 16     mean success 0.16
  between-seed var 0.0601   within-seed var 0.0716   -> 46% seed / 54% policy
```

**Verdict: BOTH matter, roughly equally.** 15/25 clean `Low1_launch` seeds are MIXED
— the same byte-identical saved state both lands and fails purely because the policy
sampled different actions — so the agent does have agency. But 10/25 have a fixed
outcome under this policy (9 always-lose, 1 always-win) and the variance splits
53/47. That is not a bag of coin flips (a phase-determined pool would be ~100%
between-seed), and it is not "the seed is irrelevant" either.

An earlier version of this section said "NOT phase-determined; the pool is
legitimate practice". That was stated more strongly than the numbers support: half
the outcome is decided before the agent acts.

So **the capture-box fix is worth making**: the clean 57 are states the policy can
influence, and the 43 mid-ladder captures are diluting them with reps on an
unexecutable task.

Careful with the 9 ALWAYS-lose: that means "this policy never lands from here in 10
tries", NOT "unwinnable". The scripted grid solved 5/5 of the clean seeds it sampled
while the policy converts 0.29, which suggests a policy gap rather than fated
states — but those were not the SAME seeds, so it is not proven per-seed. The tight
check is to run the scripted grid on exactly the 9 ALWAYS-lose seeds; if a script
wins from them, they are unambiguously teachable and the shortfall is entirely the
policy's.

Secondary finding: **all 100 `Lclimb3_top` states pass the clean-f11 rule** (it sits
at x_ram 66 on the SAME platform as `Low1_launch`'s x_ram 60 jump-off), yet it
converts worse — 0.16 vs 0.29, 16/25 ALWAYS-lose vs 9/25.

**Do NOT explain that by starting distance — the per-seed table refutes it.** Within
the `Low1_launch` pool, outcome does not order by x at all:

```
  x_ram 62  seeds 2, 10, 21    0/10, 0/10, 0/10      lifetimes all [6, 6, 54, 54, ...]
  x_ram 65  seeds 1, 3, 11     10/10, 9/10, 9/10
  x_ram 66  seeds 22, 23       0/10, 0/10            seed 23: all ten deaths at step 8
```

x_ram 62 is CLOSER to the jump-off than 65 and loses every time. So "further right =
worse" is not the mechanism, and the `Lclimb3_top` gap needs a different explanation.

What the lifetimes suggest instead is **hazard phase**: the three x_ram 62 seeds share
one signature (die at step 6, or survive to exactly 54) and seed 23 dies at step 8 in
all ten repeats. Lifetimes across the pool cluster on a few discrete values
(6, 8, 9, 21, 23, 53-61), which is what a periodic hazard produces. That connects to
the pool phase-poverty already recorded for `Step` in the v5 config header (bonus at
capture nearly constant: `Lfruit_top` spread 2 over 100 captures, `Step` 103): if
captures happen at near-identical times they carry near-identical hazard phases, and
several "different" seeds are really the same situation.

So the next question is not about position. It is: **does the `Low1_launch` pool span
hazard phases, or is it a handful of phases repeated?** `debug/l4_step_phase_diversity.py`
does exactly this measurement for `Step` and can be pointed at this pool.

## THE DOOMED FRAME, and the anchor fix it produced (2026-08-24)

The single most useful thing learned on this level. **A state can read as grounded and
already be committed to a fall.** Walking off a platform edge produces a frame where
`y` is still the floor's standing value and the pose is still a walk pose (5, not 11),
so every "is the agent standing here" test based on `(y, pose)` says yes. One step
later it is pose 11 and falling.

Everything below follows from that.

### Sprite geometry (measured, not assumed)

Located the sprite in a frame at a known RAM position (`x_ram` 44 -> px 184, y 70):

```
width 14 px, height 18 px
x_ram*4 + 8  = sprite CENTRE   (horizontal)
y            = sprite TOP      (vertical)   -> feet at y+17
foot row spans centre-6 .. centre+2, i.e. ~9 px, narrower than the sprite
```

The convention is asymmetric — centre in x, top in y — which is easy to get wrong.

### How to measure a standable span (`scripts/mo5/yeti/diag/l4_edge_limit.py`)

Naive probes do NOT work, because of the doomed frame. The method that does:

1. walk toward the edge one gym step at a time, `save_state()` at each distinct x;
2. reload each saved state and hold NOOP for ~12 steps;
3. a position is standable only if the agent is STILL at the floor's standing y in a
   surface pose afterwards.

Results (L4, 5 seeds per direction). Note `x_max` is EXCLUSIVE
(`[col0*8, (col1+1)*8]`):

```
floor      tiles          LEFT limit        RIGHT limit      fell on NOOP
 12    [184..232)      188 = x_min+4      228 = x_max-4          232
 13    [  0..128)       (probe stalled)   124 = x_max-4          128
  3    [248..280)      256 = x_min+8      276 = x_max-4      248, 280
  9    [200..232)      208 = x_min+8      232 = x_max+0
```

Right limit is `x_max - 4` (floor 9's `+0` is unexplained; likely a small inaccuracy in
that platform's recorded extent). Left limit is `x_min + 4` or `+8` and is NOT a
formula — probe per floor.

### TWO STATIC RULES THAT DO NOT WORK — do not re-derive these

* **"the whole 16 px sprite must be inside the platform"** — flags 19 of 30 L4 anchors,
  including `Spring` and `Step`, which mark fine. No discrimination.
* **`centre <= x_max - 8`**, inferred from positions the agent had been OBSERVED
  standing at — wrong twice: the bound is 4 not 8, and "never observed at the edge"
  was mistaken for "cannot be at the edge". It declared `Low2` 8 px off the platform
  and L3's `A1..A5` broken, on no real evidence.

Geometry alone can only narrow the search: an anchor within 4 px of a tile edge is
SUSPECT and needs the NOOP probe; 8 px or more inside is safe. That is all
`debug/yeti_standable_audit.py` now claims.

### The first attempt at a fix, and why it FAILED

```
Low2_launch   x_ram 44 -> 45   px 184 -> 188   floor 12's FIRST standable centre
Low2          x_ram 30 -> 29   px 128 -> 124   floor 13's LAST standable centre
```

Correct positions, useless as a fix. Measured with a 100k probe (v7): the pool came back
with **92 of 100 seeds still at px 184**, 16 of them already falling. Barely different
from v6's 81/100.

The reason is arithmetic and should have been checked first: the anchor moved 4 px
while the capture box is +-24 px (jump tolerance 6). Moving a 48 px box by 4 px barely
changes what it admits, and the agent lingers at the edge, so the brink is still what
gets captured. **An anchor move cannot fix a box that is wider than the platform's safe
region.** See "WAYPOINT BOXES MUST FIT THE PLATFORM" below for what actually worked.

Cost of finding this out: one 3-minute probe. This is the argument for always probing at
100k before committing to a 6-hour run.

`Low2_launch` was anchored on floor 12's tile edge, where the agent cannot stand. Its
capture box therefore recorded doomed frames: **all 100 seeds fell within a step or
two, 19 of them already in pose 11 on load.** The pool meant to teach the rope-2
crossing taught falling instead, and the agent never attempted the jump — visible in
`Low2_launch_ep0`, which starts mid-fall, drops onto the trampoline below, is thrown
back up to floor-12 height, jumps left and dies. Contrast `Rope1_launch`, whose anchor
happens to sit 24 px from where its seeds land (on a ladder): 0/100 doomed.

Pool provenance was added at the same time and is REQUIRED for the anchor fix to do
anything — see below.

## WAYPOINT BOXES MUST FIT THE PLATFORM (2026-08-24) — this is what fixed it

THE INVARIANT: **a detection box must lie entirely inside the platform's standable
span.** Every waypoint defect on this game violated it, and they were all the same bug
wearing different clothes:

* `Low2_launch`'s box straddled floor 12's brink, so its pool filled with 100 states that
  read as grounded and fell a step later;
* `Low1_launch`'s +-24 px box reached a ladder 24 px away and reported a stalled climb as
  an arrival (reach 0.36 vs 0.03 for the same event);
* `Fr1` sat on an extremity the agent never occupies, so the reward could never mark it.

Measured against `[x_min+8, x_max-8]` (`yeti_map.standable_span`, conservative — it is
inside every per-floor limit measured by the NOOP probe):

```
L4   21 of 30 boxes reached outside their platform     ALL 21 were jump waypoints
L3   13 of 19                                          all 9 L4 ladder boxes passed
```

One cause, not 21 mistakes: `jump_waypoints` derives anchors from platform EDGES while
the standable run stops short of them, and the jump tolerance was a flat +-24 px needing
a 48 px span that most of these platforms do not have. The Hi-chain platforms are 16 px
wide, so their span is a SINGLE centre and no tolerance above 0 can fit.

### Tolerance-only was tried and REVERTED

Narrowing the tolerance alone is a regression, because 19 of 21 L4 jump ANCHORS were
themselves outside the span: narrowing drives them to tol 0 on a position the agent
cannot stand on, so they detect NEVER instead of occasionally. Only `Fr1` and `Rope1`
survived it, and only because those two had already been re-measured. Anchor and
tolerance have to move together.

### What was done

Both are now DERIVED from the map, per waypoint:

* anchor: placed against the edge that faces the other platform (where the agent departs
  from or arrives at), pulled inside far enough that the whole box fits;
* tolerance: `yeti_map.waypoint_tolerance` narrows the caller's ceiling until the box
  fits. The curriculum's flat jump tolerance 6 is now a ceiling, not a value.

Result: **L4 boxes outside their span went 21 -> 0**, route A and route B. Independent
cross-check: every new box contains positions the agent was measured standing at
(Rope1's landing px 116 inside 112..128; Low1's seeds at 220 inside 208..224; Fr2's at
304/308 inside 304..312).

Verified by a 100k probe (v8) — the pool purity that failed all day now passes:

```
Low2_launch on-surface seeds     positions
v6   81/100    62 at px 184 (the brink), 19 already falling
v7   84/100    76 at px 184        <- the 4 px anchor move changed nothing
v8   10/10     px 192,196,200,208  none at the brink, none falling
```

`Low1`, `Spring`, `Step`, `Rope1` all came back 100% on-surface too.

### TWO THINGS NOT TO MISREAD

* **Reach numbers are not comparable across a tolerance change.** The reach EMA uses the
  same tolerance that shrank, so the identical trajectory now registers fewer arrivals.
  v8 reads rung-10 reach 0.27 against v7's 0.50 from the same champion; that is the
  metric moving, not the policy. No v8 route number can be compared to an earlier run.
* **Fewer seeds per unit time.** `Low2_launch` captured 10 seeds in 100k where v6 got
  100, because the box went from +-24 px to +-8 px and now admits only genuine launch-pad
  stances. Correct, but thinner practice early in a run.

This is a REWARD change (marking moves with the anchors), so warm-starts are no longer
strictly clean. Direction of change is favourable — every box gains usable area rather
than losing it — but the critic was fit to the old signal.

L3 is deliberately untouched: 11 violations remain (`Lesc_top` plus the A1..A5 ascent),
pinned by a test so there is a baseline when it is picked up.

## POOL ANCHOR PROVENANCE (2026-08-24)

`checkpoints.pkl` now records `waypoint_anchors`: the anchor each pool was captured
around. On load, a pool whose anchor has since MOVED is dropped.

Without this the anchor fix is inert on a warm start. The resume path already dropped
pools for waypoints a level no longer DEFINES, but that check is on the NAME, and an
anchor move keeps the name — so the 100 doomed `Low2_launch` states would have been
re-imported and v7 would have measured nothing. (Same shape as the `build_targets`
alias bug: a name/coordinate check that an anchor move slips through.)

Files written before the field exists cannot be checked; the loader now says so instead
of trusting them silently. `debug/yeti_stamp_pool_anchors.py` backfills provenance into
an older file by re-deriving the old anchors with the override removed (not hardcoded).
Run on v6's pool file, which is why the drop fires:

```
Dropped inherited pool 'Low2_launch': anchor moved (44,70) -> (45,70), 100 states
Dropped inherited pool 'Low2':        anchor moved (30,70) -> (29,70), 7 states
pools inherited: 18 (was 20); Rope1_launch / Low1 / Lclimb3_top untouched
```

Pinned by `tests/python/test_pool_anchor_provenance.py` (8 tests).

## THE POSE CATALOGUE, AND A CHECK FOR GAPS IN IT (2026-08-24)

`yeti.POSE_NAMES` is now the single catalogue of every identified sprite pose, with
`yeti.KNOWN_POSES` and `yeti.unknown_poses()`. The trainer censuses every pose it sees
(`CheckpointManager.pose_seen`) and the status line reports any code that is not
catalogued. Silence on that line is the assertion that the catalogue is complete.

Why it matters: an uncatalogued pose is a silent behaviour change, not trivia. Every
pose-gated decision — waypoint reach detection, seed capture, reward milestone marking,
deepest-floor crediting — reads an unrecognised code as "not on a surface", so whatever
happened in that frame does not count anywhere.

**The walk cycle is FOUR poses per direction, and only the rightward one was listed.**

```
0,1,2,3   dx in {0, +4}   walk RIGHT, grounded   all in SURFACE_POSES
4,5,6,7   dx in {-4, 0}   walk LEFT,  grounded   6 and 7 ARE MISSING
```

Measured by holding a direction and logging pose against the per-step lateral delta at
the floor's standing y, across 4 L4 floors. Effect on L4 floor 12: **54% of grounded
leftward-walking frames are discarded by the surface gate, versus 0% walking right.**

L4's closing stretch is leftward — rope 2 crosses floor 12 → 13 leftward, and floor 13
to the princess ladder is leftward — so the gap lands exactly on this level's wall. It
also means the `Low2_launch` pool was being filled from only the subset of leftward
frames that happened to be pose 4 or 5, which interacts with the doomed-frame problem
above.

### FIXED (v9): poses 6 and 7 admitted to `SURFACE_POSES`

Probed at 100k. Result is mixed and the honest reading matters:

* **The gate is complete.** `grounded frames NOT counted as surface` is absent from the
  status line, against 1857 per 100k in v8. Pools stayed 100% on-surface, so admitting
  more frames did not admit bad ones.
* **Capture counts did NOT move.** Spring 142->143, Step 141->140, Lclimb3_top 134->131,
  Low1 134->133, Low2_launch 10->8. Flat, or noise.

**Why the "54% of frames" figure over-promised, and the lesson.** Capture and reach are
ONCE-PER-EPISODE events, not per-frame: a waypoint captures at most once per visit, and
the agent already had some pose-4-or-5 frame inside the box during each visit. Doubling
the eligible frames cannot produce more captures when one was always sufficient. A
per-frame statistic does not predict a per-episode outcome — check which one a mechanism
actually depends on before predicting from it.

**Where it does genuinely matter** is the shaping freeze, which IS per-frame
(`rewards.py`: `airborne = pose not in _surf`, and the potential is only sampled on
surface poses). Leftward walking earned shaping credit on roughly half its steps, so the
gradient was intermittent in one direction only. That is a real defect and this fixes it,
but it shows up as learning efficiency over millions of steps and is invisible at 100k.

So this lever is defensible on correctness and UNPROVEN on effect. It was kept because it
is correct and cheap to carry, not because the probe endorsed it.

It is still a reward change (more frames can mark a milestone), so warm-started critics
are fit to the old signal.

**Pose 15 exists and is unidentified.** The check found it within minutes of going in —
4 occurrences in a 3000-step smoke run. Not yet characterised; do not guess. Poses 6
and 7 also remain to be confirmed as a full cycle on levels other than L4.

## POSE 15 IS THE LEFTWARD ROPE CARRY — i.e. THE ROPE-2 CROSSING (2026-09-11)

The pose the trainer has reported as `UNCATALOGUED POSES 15x5901` on every status line,
for the whole history of this project, is the manoeuvre L4's wall is made of.

Identified from the first rope-2 crossing ever recorded on video: v13's champion, from
reset, episode 236 of a 1500-episode hunt (`debug/l4_rope2_video.py`, clip
`experiments/003-yeti/evidence/l4_rope2_video/crossing_1_ep236_neither.mp4` — the filename says "neither"
because the script's classifier only knew pose 14 at the time). User-confirmed on the
video.

```
t=478  [0,2,0]  px 188  y 70  pose 4   grounded   <- leaves floor 12 from px 188
t=479           px 184  y 66  pose 10  airborne
t=487           px 164  y 62  pose 15
t=488           px 160  y 62  pose 15
t=489           px 160  y 62  pose 15
t=500           px 132  y 62  pose 15
t=516  [1,2,1]  px  88  y 70  pose 5   grounded   <- floor 13, 38 steps in the air
```

Three things this settles.

**It is a rope traverse, not a jump and not the trampoline.** px advances monotonically
188 -> 88 across 38 airborne steps with no ground contact, while y oscillates
70/60/62/52/62/52/70. Repeated jumps would touch down in between. Pose 14 never appears
and neither does 16/17.

**Pose 15 is 14's LEFTWARD counterpart.** The catalogue is paired by facing direction
throughout (0-3/4-7 walk, 9/10 jump, 16/17 trampoline). Rope 1 is crossed RIGHTWARD and
shows pose 14; rope 2 is crossed LEFTWARD and shows 15, never 14. Every pose-15 frame in
the crossing sits at exactly y = 62 — the rope's height — while px keeps advancing, which
is precisely 14's documented signature, "lateral motion while held".

### ROPE 2 IS A ROPE TRAVERSE, TWO CATCHES IN 2 OF THE 3 OBSERVED CROSSINGS

(Heading corrected 2026-09-29. It read "ROPE 2 IS A TWO-CATCH TRAVERSE. THAT IS WHY IT IS
NOT ROPE 1." — which the body of this very section contradicts: crossing 2 crossed with
ONE catch. Two catches is the mode, not the rule, and the heading was being read as the
rule.)

User-confirmed on the videos for crossings 1 and 3: the agent jumps, CATCHES THE ROPE,
swings left a little, jumps left again, CATCHES A ROPE AGAIN, then jumps onto the
platform. The pose stream says the same thing — two separate runs of pose 15, with
pose-10 hops between them:

```
crossing 3  [4, 10, 15, 10, 15, 10, 4]
   catch 1   t=482-484   px 168, 172, 172   y 62
   catch 2   t=498-500   px 136, 132, 132   y 62

crossing 1  [4, 10, 15, 10, 15, 10, 5]
   catch 1   t=487-489   px 164, 160, 160   y 62
   catch 2   t=500       px 132             y 62

crossing 2  [5, 10, 15, 10, 0]            <- ONE catch, and still crossed
   catch 1   t=489-491   px 160, 156, 156   y 62      landed px 120
```

Catches cluster in two groups, px 156..172 and px 132..136, both at y = 62. The gap runs
from floor 12's edge (px 184) to floor 13's edge (px 128), so two ropes hanging inside it
fits. Whether that is two distinct ropes or one rope caught at two swing phases is NOT
settled — position alone cannot separate those, and this file already retracted one claim
for exactly that reason ("ropes move between frames, so single-frame pixel measurement
does not support it"). Crossing 2's single catch is also unexplained.

**Compare rope 1, which the agent clears at 0.68-0.83 reach:** ONE catch — pose 14 carry
for ~5 gym steps (px 60 -> 72, y 118 -> 110), then a pose 9 arc landing at px 116. So the
answer to "why isn't rope 2 as easy as rope 1" is not gap width (52 px vs 60 px, nearly
equal) and not the anchor. Rope 1 is one timed catch; rope 2 is usually two, in sequence —
but NOT always, and the difference is not the number of catches. Crossing 2 used one catch
and crossed. Measured 2026-09-29 with scripted input from px 188, the launch window for
rope 2 is 10 of every 31 steps against rope 1's 3 of 31, i.e. rope 2 is the more forgiving
of the two to TIME (see the 2026-09-29 section below). Whatever makes rope 2 hard, it is
not a narrower launch window.

**RETRACTED 2026-09-29 — a held input DOES cross rope 2, and the px-88 landings were
real crossings.** This paragraph used to claim that all ~700 scripted trials failed
because they HELD a jump input, that "a held input can never do this: it needs a release
and a re-jump timed to a moving rope, twice", and that every scripted px-88 landing was
the trampoline. All three are false. From a `Low2_launch` seed at px 188, the plan
`wait W noops, then hold JUMP_LEFT` crosses for every W in 12..21, and the per-step trace
is the rope, not the trampoline:

```
frame  16   px 188  y 70  pose  4    standing on the pad
frame  18   px 184  y 66  pose 10    launch
frame  26   px 168  y 62  pose 15    CATCH 1
frame  30   px 160  y 62  pose 15
frame  32   px 156  y 58  pose 10    hop between catches
frame  40   px 132  y 62  pose 15    CATCH 2
frame  57   px  88  y 70  pose  4    standing on floor 13
```

Pose sequence `4, 10, 15, 10, 15, 10, 4`, identical to crossings 1 and 3 above. Both
catches sit at y 62, inside the two clusters this section already recorded (px 156..172
and px 132..136). Pose 16 and 17 never appear and y never approaches the trampoline's
142, so this is not the fall-bounce loop.

WHY ONE HELD INPUT SUFFICES, which is the part nobody had worked out: both catches are
phase-locked to ABSOLUTE time, not to the release. The landing frame is 57 for every
winning W — 44 airborne steps at W=12, 36 at W=21. So the agent does not have to time two
things. It has to launch inside one window and the ropes do the rest.

What survives: the scripted falsification's conclusion that **the anchor is not the
binding constraint** still stands, on its own evidence. What does not: the reason given
for it, and any use of "no script can express this manoeuvre" as an argument.

**CONFIRMED ON 3/3 CROSSINGS (1071 episodes).** All three use pose 15, none uses pose 14
or 16/17, and every one of the 15 pose-15 frames sits at exactly y = 62:

```
crossing  ep     airborne   departs   lands   pose sequence
   1      236     38 steps   px 188   px  88  [4, 10, 15, 10, 15, 10, 5]
   2     1003     25 steps   px 184   px 120  [5, 10, 15, 10, 0]
   3     1070     45 steps   px 188   px  92  [4, 10, 15, 10, 15, 10, 4]
```

Rate: 3/1071 = 0.28%, and pooled with the n=300 champion eval, 4/1371 = 0.29%. So ~0.3%
is the right figure for this policy.

**px 184 IS NOT UNIFORMLY LETHAL — a correction.** An earlier version of this section
claimed the working route departs from px 188, therefore the +0.12 for stepping
188 -> 184 is not the route's entry and a lethal-margin filter is safe. Crossing 2
departs from **px 184** and crosses successfully, landing at px 120. The claim is
withdrawn. What the measurements actually support is narrower and conditional:

* px 184 is lethal **from rest**: 20/20 pool seeds and 12/12 walk-there-and-stop trials
  die, at a median of step 82, via the fall onto the trampoline.
* px 184 is survivable **with leftward momentum and an immediate rope grab**, 1/3 of
  observed crossings.

So a lethal-margin filter on CAPTURES is still justified — a seed is reloaded at rest,
which is the doomed case — but "px 184 is off the route" is false. Do not use this to
argue that stepping there is always a mistake.

**The landing spread straddles `Low2`'s credit window.** Landings were px 88, 92, 120
against a window of px 104..152. Only crossing 2 is credited `Low2` on landing; the other
two must walk RIGHT afterwards to earn it. That matches the 11 genuine pool captures
sitting at px 104..124 — they are captured after the walk, not at touchdown. Recorded as
data, not as a proposed anchor move: three landings is a thin census, and this file's
rule is that anchors come from a multi-policy grounded census.

Catalogued in `POSE_NAMES` as "rope carry, facing left (L4), lateral motion while held".
The name deliberately omits the word "grounded", because
`train_checkpoint_curriculum.py` derives its grounded-pose set by substring on these
names; `test_no_pose_name_falsely_claims_grounded` now guards that. Adding it is
otherwise behaviour-neutral — `KNOWN_POSES` feeds only the uncatalogued census, and 15
was already counted for reach (the `NON_TRAVERSAL_POSES` blocklist names only 11 and 12).
The census warning that has fired on every run now goes quiet, and
`unknown_poses(range(18))` is empty, so the next genuinely new code will stand out.

## POSES 16 AND 17 ARE THE TRAMPOLINE, NOT A ROPE CARRY (2026-08-24)

Corrects an earlier reading in this file. There is a trampoline at
`Platform(24, 142, 168, 200)` ("P19, spring above"), directly below the rope-2 gap.

```
14  rope carry      confirmed on rope 1: x advances while y holds, then pose 9 arcs out
16  trampoline up, facing right
17  trampoline up, facing left
```

Poses 16/17 rise 4 px per gym step at CONSTANT x while the agent is thrown back up from
the trampoline. The earlier note that "ep3 hangs at (44,74) in pose 17 on the launch
pad" was wrong: the agent had already fallen off floor 12 and was cycling
fall -> trampoline -> rise -> fall. Any analysis reading pose 17 as "stuck on the rope"
is misreading a fall-recovery loop.

Rope 1's crossing, for reference (from its pool, champion policy): walk to floor 6's
edge, `pose 14` carry for ~5 gym steps (px 60 -> 72, y 118 -> 110), then `pose 9` arc,
landing GROUNDED at px 116 — exactly the `Rope1` anchor.

## FLOOR 13'S FLAT GRADIENT IS A SYMPTOM, NOT A SECOND BUG (2026-08-24)

Worth recording because it was initially written up as an independent defect.

Standing on floor 13 with rung 11 UNMARKED, the potential is flat across px 40..104 —
64 px, half the platform. The rung-11 anchors (`Low2` px 124, `Lhi_down_bot` px 104)
are on the platform's right; the princess ladder is at px 40 on the left; and along a
1-D platform the two path-distance terms cancel exactly (sum = 88 at every x in that
span). No gradient, no signal, while snowballs cross the platform.

But a marked group LEAVES the sum (`if gi in self._reached_wp: continue`), and with
rung 11 marked the pull is cleanly left toward the ladder. So the flat zone is not a
shaping-design flaw — it is what an unmarkable milestone looks like from downstream,
the same failure mode as `Fr1`. Fix the marking and it disappears.

Snowballs on floor 13 are real (user-confirmed on video, and the hazard comment puts
them on `P10`, which IS floor 13) — they are what kills the agent while it wanders the
flat zone. They are not the root cause.

## L4 HAS TWO ROUTES TO FLOOR 13 — and we deliberately do not steer between them

```
route A   floor 11 -> floor 12 -> ROPE 2 (56 px gap, widest on the level) -> floor 13
route B   floor 11 -> ladder px 296 -> floor 14 -> five 16 px hops (f15..f19)
                   -> ladder px 104 DOWN -> floor 13
both      floor 13 -> ladder px 40 -> floor 20 -> princess
```

That is what the rung-11 OR-group `[J12_13_b, Lhi_down_bot]` encodes: one member per
route. The group is correct as designed.

Route B is unexplored — `Lhi_up_top` has ZERO captured seeds in 15M steps, though its
ladder foot is 24 px from where the agent stands 84.7% of the time. Its five jumps are
16 px each, the same size as `Fr1` (cleared 96%), versus rope 2's 56 px. The graph makes
route A look 16% shorter (152 vs 176 from rung 10), so the shaping points at the rope.

**Rejected as a lever.** Waypoints and rewards describe the level; the agent picks the
route. Biasing toward route B would be us choosing. Also note the potential is a
shortest-path sum, so it will always prefer the shorter route — route B's first three
steps are penalised (152 -> 176 -> 200 -> 216) whatever the anchors are. That is
geometry, not a defect.

## !! DO NOT WARM-START FROM A CHAMPION (measured 2026-08-31) !!

The most actionable thing learned so far, and it cost a 6.5 hour run to find.

v10 warm-started from v6's `best/best_model.zip` (mean rung 9.55 at n=60) and ended at
6.60. That looks like a catastrophic regression. It is not.

**A champion is SELECTED as the best of ~150 snapshots**, i.e. selected for being an
outlier of a very wide distribution. Continuing training from one returns to the
distribution by construction. Bisection control arm A0 — v6's own code at `bc85424`, v6's
config, v6's pools, seed 42, ONLY `resume` pointed at the champion — reproduces the
"collapse" with none of the later changes present:

```
starting policy   9.55
A0  250k          1.20
A0  500k          2.43
A0  750k          7.53
A0   1M           1.03      (v10 at 1M was 4.13, i.e. BETTER than the control)
```

So no geometry or pose change was implicated. Use a run's `final_model.zip`, as v4->v6
did, or start cold. If a champion must be used, expect to re-earn its level.

## THE POLICY OSCILLATES ACROSS ALMOST THE WHOLE DEPTH RANGE

A0's own numbers above are the clearest measurement of it: **1.03 -> 7.53 -> 1.03 within
250k steps.** v6's `Spring` reach hit exactly 0.00 at 4.5M, 6.0M and 9.5M and bounced back
to 0.75 / 0.68 / 0.77 each time, with the skill intact throughout.

Consequences, learned the hard way — four wrong causal explanations were produced in one
day by ignoring them:

* **A single snapshot eval is not a measurement of a run.** A sequence of them is not a
  "degradation curve"; it is samples of a swinging process. v10's
  `8.53, 4.13, 5.57, 6.00, 5.80, 5.60` was read as progressive collapse. It was noise.
* **Champion vs final is never a valid comparison.** Compare DISTRIBUTIONS over all
  snapshots (`keep_best_sweep`), which is what the depth scoring exists for.
* **A reach EMA hitting 0.00 does not mean the skill is lost.** Confirm with an eval before
  believing it. v6's zeros were EMA dips; v10's were real, and only an n=60 eval
  distinguished them.
* **A 1M abort gate does not work as a point reading.** The oscillation band spans the
  whole range within 250k, so any single early number is uninformative. A gate has to be a
  distribution over several early snapshots.

## RUN PROTOCOL — what to fix before launching, every time

Today's run was described as carrying "three changes" and actually carried four. The
unlisted one (warm-start source) turned out to be the only one that mattered, and a null
result would have been uninterpretable either way.

1. **Name every difference from the reference run, including the boring ones.** Warm-start
   SOURCE is a lever in its own right, not plumbing. So are seed, pools, and PPO
   hyperparameters.
2. **One lever, or a control arm.** If more than one thing moves, a control is mandatory,
   not optional. Commits make clean bisection points — `git worktree add /tmp/wt <sha>`,
   run with `PYTHONPATH=/tmp/wt/python` and the script from the worktree, cwd in the main
   repo so `output/` and `roms/` resolve.
3. **Fix the measuring stick.** Evaluate every arm with the CURRENT evaluator at one
   tolerance, never by each arm's own route table. Verify the stick reads the shared
   starting policy consistently first (the champion scored 9.38 at tol 2/6 and 9.55 at
   2/2, so either is valid; mixing them between arms is not).
4. **Judge by distribution, not by the final model or the champion.**
5. A metric change is a lever. Narrowing the waypoint tolerance changed what `reach` means,
   which made a real regression look like an artifact and an artifact look like a
   regression. Re-validate the stick whenever detection changes.

## v10 RESULT (2026-08-31) — no attribution possible

Full sweep, same settings and same stick as v6's:

```
            n     mean   median    max    >=9.5    <1.0
v6         150    6.37    7.88    10.00    8/150   10/150
v10        151    5.84    6.42     8.92    0/151    5/151
```

Modestly worse at the top, modestly better at the bottom (half as many collapses). **Not
attributable to the committed fixes**, because v10 also changed its warm-start source and
A0 showed that change alone dominates the outcome. The geometry and pose fixes therefore
remain UNTESTED for effect, though verified correct for behaviour (boxes 21 -> 0, pools
100% on-surface, gate complete).

Princess 0 across all 151 snapshots; rung 11 never reached. Nothing so far has moved the
actual wall.

Next: rerun the current code with v6's exact warm-start source (v4's `final_model.zip`,
v4's pools, seed 42) so v6's distribution becomes a legitimate control and
`mean 6.37 / median 7.88` is the number to beat.

## ANCHORS: THE GEOMETRIC DERIVATION WAS WRONG, THE CENSUS IS RIGHT (2026-09-03)

`e667206` derived all 20 L4 jump anchors from platform geometry (fit the tol box inside
`standable_span`) and narrowed tolerance to make them fit. Bisected at 1.2M each, seed 42,
v4 warm start, one lever per arm:

```
arm                          Rope1 @ healthy step
revert (neither change)          0.75 - 0.84
B1 anchors only                  0.79 - 0.85
B2 tolerance only                0.78 - 0.87
v11 both              decays to  0.01 - 0.06  and STAYS there for 11M steps
v11 repeat, seed 43   decays to  0.01         reproduced
```

### The diagnostic that actually works: divergence from the downstream neighbour

`Lclimb2_top` is reachable ONLY THROUGH Rope1's platform. So this is impossible:

```
v11        1.5M   reward 57 (healthy)   Rope1 0.01   Lclimb2_top 0.81
v11rep43  1.15M   reward 50 (healthy)   Rope1 0.01   Lclimb2_top 0.74
```

The agent crosses rope 1 and the detector is blind. **A waypoint reading far below its own
downstream neighbour at healthy reward means detection is broken, not that the agent
stopped going there.** Use that, not the absolute value. The same signature is on record
from L4 v1 (Rope1 1.8% vs Lclimb2_top 87%) where the response was to widen the box — a
workaround for the anchor, not a fix.

### Why geometry is the wrong criterion

A jump landing depends on the POLICY. Measured on floor 7, same pool, two policies:

```
v6 champion   lands px 116
v11 policy    lands px 108, then AIRBORNE 112..136, ladder at 144
```

Traced frame by frame: **grounded for TWO FRAMES at px 108, then pose 9 all the way to the
ladder.** Detection is pose-gated, so the airborne frames cannot fire. A jump platform
offers one narrow grounded window and an anchor 4 px off it detects nothing:

```
anchor 25 (px 108) box 100..116    v6 1.00   v11 0.97
anchor 27 (px 116) box 108..124    v6 1.00   v11 0.97
anchor 28 (px 120) box 112..128    v6 1.00   v11 0.00   <- sits on the jump arc
```

Anchor 28 scored 1.00 against the champion. Single-policy validation is what planted it.

### The two channels, which is why neither half of the bisect reproduced it

Curriculum tolerance is 6 (+-24 px); the REWARD tolerance is a fixed 2 (+-8 px). B1 broke
only the reward channel, B2 only the curriculum channel, v11 broke both.

### Rule adopted (debug/l4_anchor_recommend.py)

Census the positions the agent is actually GROUNDED at, over >= 2 policies from different
runs, score per EPISODE, choose by the WORST policy's score, keep tolerance flat.

### Two mandatory milestones were unmarkable in the v4/v6 baseline

The `Fr1` defect class, still live until now. A mandatory milestone whose tol-2 reward box
contains no position the agent is ever grounded at can never be marked, so its distance
term stays switched on for every episode:

```
Step    anchor 60  box 240..256   fires 0.00   (agent stands at px 256/272)  -> 66
Spring  anchor 48  box 192..208   fires 0.12   (modal grounded px 212)       -> 51
Rope1   anchor 27                 fires 1.00   -> 25, mode at box CENTRE not edge
```

`test_mandatory_reward_boxes_cover_a_measured_grounded_position` now guards this class.

### Verdict on the three censused anchors: better on every axis

`l4_anchors_v2_1200k` vs the revert probe as control, matched at HEALTHY steps:

```
step   arm         reward   Rope1   Spring   Step   Lclimb2_top
600k   control       45.9    0.75    0.74    0.70      0.74
600k   anchors_v2    53.4    0.83    0.75    0.71      0.81
700k   control       46.4    0.67    0.64    0.62      0.66
700k   anchors_v2    53.4    0.85    0.82    0.81      0.85
750k   control       46.4    0.71    0.68    0.67      0.69
750k   anchors_v2    53.4    0.87    0.84    0.81      0.87
```

## THE FROM-RESET REWARD COLLAPSES AND RECOVERS. ENDPOINT READS ARE NOT VERDICTS.

Measured directly from `episodes.csv`, mean `total_reward` of from-reset episodes per 100k:

```
revert control  ... 46 46 [2.6] 44 52 55        one bucket at 2.6, recovers
B1              ... 50 54 [21]  53 53           recovers
anchors_v2      ... 53 47 [3.5  3.0  3.5  3.0]  collapsed at 900k, run ENDED at 1.2M
v11 (15M)       collapses to ~2-20 about 20 separate times, recovers every time
```

When it collapses, **every** waypoint reads 0.00 together, including ones whose anchors
never moved (`Fr1` 0.94 -> 0.03, `Lfruit_top` 0.98 -> 0.72) while `prog` stays 0.88-0.99
and captures keep accruing. That is a global policy collapse, distinguishable from
detection blindness by exactly this: detection blindness leaves reward HEALTHY and hits ONE
waypoint; collapse takes reward to ~3 and hits ALL of them.

`l4_anchors_v2`'s 0.00 at 1.0M/1.2M was collapse, not its anchors. Read matched healthy
steps, and never treat a run's last snapshot as its result.

## A MID-JUMP STATE DOES NOT RELOAD INTO A FALL (measured, the comment was false)

Claimed three times in `train_checkpoint_curriculum.py` as the justification for gating
seed capture on grounded poses. Save a pose-9 state mid-arc, reload, feed identical inputs:

```
without reload   px 40 y 114 -> 44/110 -> 48/108 -> 56/108 -> 64/114   pose 9 throughout
after reload     px 40 y 114 -> 44/110 -> 48/108 -> 56/108 -> 64/114   IDENTICAL
```

A save-state is a full emulator snapshot; velocity and jump counters return with it.
Grounded is also INSUFFICIENT: `Low2_launch` held 100 seeds that read grounded on floor
12's brink and fell one step after load. `admit_requires_survival` is the direct test.
Grounded stays as a PREFERENCE (a mid-jump seed hands the agent a committed trajectory),
not a correctness requirement. Comments corrected.

## PROPOSED, NOT YET RUN: SPRITE OVERLAP + POSE BLOCKLIST

Today's fixes are per-anchor. This retires the class. Detection currently asks "is the
agent's POSITION inside a tolerance box". Ask instead "does the agent's SPRITE contain the
anchor POINT". Identical in x (sprite half-width IS a derived tolerance, ~7 px ~ 2 x_ram)
but very different in y: today's test is +-tol around the sprite TOP, a 4 px window, while
overlap asks whether the point lies in [y, y+17], an 18 px window. A jumping agent's y
DECREASES, so its sprite still spans the floor line for much of the arc.

Measured, fraction of episodes firing, on the good anchor and the one that broke:

```
                          v6champ a25  a28     v11 a25  a28
SPRITE allow surface           0.04  1.00         0.96  0.00   <- still anchor-sensitive
SPRITE allow +traverse         1.00  1.00         0.96  0.96
SPRITE block fall+death        1.00  1.00         0.96  0.96
SPRITE no gate                 1.00  1.00         0.96  0.96
```

Sprite overlap ALONE does not help; sprite overlap plus a non-allowlist gate makes the
anchor stop mattering. Note the top-left cell: anchor 25, the one shipped today, is 0.04
on the champion under the current allowlist test.

**Allowlist vs blocklist.** `SURFACE_POSES` is an allowlist, which FAILS CLOSED: poses 6
and 7 were missing from it for the project's entire history (~54% of grounded frames on any
leftward approach suppressed) and pose 15 is STILL uncatalogued and appears every run. A
blocklist fails open — an unknown pose counts, and only poses known invalid are named.
Excluding 11 (fall) and 12 (death) is a judgement call, not a measurement; fall frames do
not intersect these boxes either way.

Shape: detection = sprite overlaps anchor point; gate = blocklist {11, 12}; capture =
survival gate, with grounded as an eviction PREFERENCE. This changes reward AND metrics, so
it invalidates comparison to v6 and needs a fresh champion first.

## THE WALL IS Step -> Lclimb3_top, NOT RUNG 11

`prog` = P(an episode seeded here reaches any NEW route point). Both arms, healthy steps:

```
Step          reach 0.81   prog 0.02
Lclimb3_top   reach 0.02   prog  --
```

The agent arrives at Step reliably and gets nowhere from it. Effort aimed at rung 11
(`Low2`, floor 13) was two route points too far along.

### Why: the ladder-3 head is lethal on a timer

40 episodes from the `Step` pool, control's 800k policy (`debug/l4_step_handoff.py`),
measured by FLOOR OCCUPANCY so no anchor is involved:

```
floor 10  y 102  px 248..304   39/40  0.97
floor 11  y  78  px 248..320    4/40  0.10
floor 12  y  70  px 184..232    0/40  0.00
```

37/40 end ON the ladder at px 272, y 78..90. The climb itself is clean and the death flag
flips at the exact frame y reaches 78, which is floor 11's standing level:

```
px 272 y 98 pose 8 -> y 94 -> y 90 -> y 86 -> y 82 -> y 78  DEAD
```

Scripted departure sweep, `debug/l4_ladder3_timing.py` -- walk to the base, wait N frames,
hold UP, then hold one action for 20 frames:

```
after arrival   surviving waits out of 0..40
NOOP            NONE
LEFT            14..22
RIGHT           11..28
JUMP-L          11..21
```

Identical on both seeds, so the hazard cycle is deterministic. Waits 0..10 die during the
climb whatever follows. **A ~18-frame safe window exists: this is a learnable timing
skill, not a structural dead end.**

### And the reach gate makes it unlearnable

`gate_waypoints: true` with `reach_threshold: 0.15` filters a waypoint on its OWN
from-reset reach. `Lclimb3_top` is 0.02, so it is excluded as a start state. Measured over
the control's 7140 episodes (`start_key` in episodes.csv):

```
start_key        starts    reach   pool
Step                217     0.81    100
Lclimb3_top           0     0.02    100
Low1                  0     0.00    100
Low2_launch           0     0.00    100
```

Three pools of 100 seeds, sampled ZERO times in a 1.2M run. The seeds are usable --
seeded there directly with the same policy, 30 episodes each:

```
from Lclimb3_top:  floor 11 0.90   floor 12 0.17
from Low1:         floor 12 0.90   floor 13 0.00
from Low2_launch:  floor 12 0.50   floor 13 0.00
```

`Lclimb3_top` reaches floor 12 in 17% of episodes. That is a live gradient the trainer
never receives.

**The gate is self-locking at the frontier.** The frontier is by definition the point the
agent does not reach yet, so its own reach is ~0, so it is never sampled, so the skill is
never practised, so its reach stays ~0. This is what has been blocking L4 -- not anchors,
not tolerances, not warm-start choice. Fix under test as H-AR: gate on the PREDECESSOR's
reach, so `Lclimb3_top` opens on `Step`'s 0.81 while `Low1` stays shut until
`Lclimb3_top` itself clears the threshold. The frontier then advances one rung at a time,
which preserves the protection the gate was added for.

### The next wall is already visible

Floor 12 -> floor 13 is 0/60 from the `Low1` and `Low2_launch` pools. The rope-2 crossing
is untouched and sits directly behind this one.

## v12 (15M, PREDECESSOR GATE): NOT A REGRESSION. v6 SEED 42 WAS A 1-IN-4 OUTLIER.

This section previously concluded that v12 regressed v6. That was wrong, and the error is
worth keeping because it cost most of a day: a large difference between two single runs was
explained as a code regression before anyone checked whether the GOOD run was reproducible.

WHAT LOOKED LIKE A REGRESSION. v12 matched v6 on `Step` (0.7-0.87) and `Lclimb3_top`
(0.6-0.79) but `Low1` sat at 0.00 for the whole 15M where v6 held 0.4-0.76:

```
P(reach Low1 | reached Lclimb3_top), whole run
    v6  seed 42     5158/5533 = 0.93
    v12 seed 42       14/7757 = 0.00     (0.08 even in seeded episodes)
```

WHAT IT ACTUALLY WAS. v6's OWN CODE (bc85424) on three fresh seeds, everything truncated
at 2M so the comparison is apples to apples:

```
run                     L3top   Low1   P(Low1 | L3top)
v6  seed 42 (bc85424)    1825   1644       0.81
v6  seed 44 (bc85424)     165     12       0.07
v6  seed 45 (bc85424)     245      6       0.02
v6  seed 46 (bc85424)     216     16       0.07
v12 seed 42 (f069429)    1643    533       0.16
```

**v12 beats every ordinary seed of v6's own code**, and reaches `Lclimb3_top` 7-10x more
often (1643 vs 165-245) -- which is the predecessor gate working. v6 seed 42 won a coin
flip: its from-reset policy scratched past the 0.15 reach gate at ~1.1M and cascaded, and
1 seed in 4 does that.

DEAD HYPOTHESES, all of which had a mechanism and a story:
* poses 6/7 in `SURFACE_POSES`, 7b27d49's trainer changes, the censused anchors -- none is
  implicated, because v6's own code fails the same way on fresh seeds.
* `Low1_launch`'s deletion -- v6 never used it: 0 seeds, 0 reaches, empty pool.
* `Step`'s reward box as an accidental "brake" that delayed the ladder climb into the safe
  window -- seeds 44/45/46 all have `Step` unmarkable exactly like seed 42, and did not
  cascade.

WHAT IS REAL, and it is a better-posed problem: **the `Lclimb3_top -> Low1` crossing is
learned in about 1 run in 4, and nothing we control influences that.** The floor 11 -> 12
leftward jump comes right after a TIMED hazard at the ladder-3 head (departures 11-28
frames into the cycle survive, 0-10 die), the enemy IS visible in the 84x84 observation, and
the `Step` seed pool spans the cycle (77-86% of its seeds survive an immediate climb). So
it is learnable in principle and learned unreliably in practice.

## IS v6 REPRODUCIBLE? NO — and that is the answer to a week of hunting (H-AS)

Answered by 3 seeds x 2M on bc85424 rather than the 15M run that was started first. The
15M was killed at 1h10m once the point was clear; sizing a run to a 1.2M event at 15M is
recorded as method rule 8.

## BOTH v6 AND v12 WALL AT ROPE 2, AND NEITHER EVER CROSSED IT

The chain, whole run:

```
                              v6 seed42     v12
P(Lclimb3_top | Step)            0.79       0.74
P(Low1 | Lclimb3_top)            0.93       0.06    <- v12 loses mass here
P(Low2_launch | Low1)            0.96       0.88
P(Low2 | Low2_launch)          0.0004     0.0021    <- BOTH die here
```

v6 stood on the rope-2 launch pad **25,520 times and crossed 10**. So the difference
between the runs is upstream THROUGHPUT to the launch pad, not progress on rope 2. Fixing
the floor 11 -> 12 crossing buys more attempts at rope 2, and v6 already showed what 25,520
attempts buys: nothing.


## FLOOR 13 HAS BEEN REACHED 24 TIMES EVER, AND NEVER SEEDED FROM

Across every L4 run on disk (`reached_points` is `;`-separated -- splitting on `|` gives
a false zero):

```
                                    episodes reached    ever seeded from
Low2_launch   rope-2 launch, f12          27,944              2,811
Low2          rope-2 LANDING, f13             24                  0
Lhi_down_bot  the other way onto f13          23                  0
Lhi_up_top    Hi-chain entry                 103                  0
```

So the launch pad is thoroughly explored (v6 alone: 25,520) and the landing has been
touched 24 times in the project's history -- v6 x10, v10 x6, v4 x3, v11 x2, singles in
v8/v9/v11rep43. Never once used as a start. Floor 13 is not unreachable; it is reached
too rarely for a pool to form, and under the own-reach gate `Low2` was never eligible to
train from, so the 7 seeds v6 banked were dead weight.

## ROPE 2 (2026-09-09): THE DIAGNOSIS WAS ALREADY WRITTEN DOWN, AND THE FIX WAS REVERTED

Read `yeti_map.py`'s comment above `jump_waypoint_pos` FIRST. It already says, in full:

> `Low2_launch edge 44 -> 45. px 184 -> 188. Floor 12's tile edge is 184 but the agent
> CANNOT stand there... Measured left limit is 188 = x_min + 4 (scripts/mo5/yeti/diag/l4_edge_limit.py,
> which confirms a stance by reloading it and holding NOOP). This is why all 100
> Low2_launch seeds were doomed -- the capture box was centred one step past the edge, so
> the pool taught falling instead of the rope-2 crossing, and the agent never attempted
> the jump.`

That is the whole diagnosis, including the tool. A full day was spent re-deriving it from
video and traces. **Read the existing comments before investigating.**

### THE REAL DEFECT: the comment claims a fix the code does not contain

The comment is written in the PAST TENSE, as though `edge 44 -> 45` had been applied. It
has not: `jump_waypoint_pos` has no `Low2_launch` entry, so the anchor is the derived
default, px 184. `e667206` applied it; reverting that commit wholesale over the `Rope1`
regression removed the code and LEFT THE COMMENT ASSERTING IT. Anyone reading the file
concludes this is fixed. See method rule 10 (revert surgically, never wholesale).

### What 2026-09-09 actually added

1. **The survival gate cannot clean this pool, and that was not previously recorded.**
   `admit_requires_survival` keeps a capture if the agent survives >= `min_survival_steps`
   (30). Holding NOOP from each of the 100 `Low2_launch` seeds:
   ```
   start px:                    184: 81   188: 8   192: 10   196: 1
   fell off immediately:        81/100
   steps to death:              median 83   min 82
   passes the >=30-step gate:   100/100
   ```
   The trampoline keeps a doomed state alive ~82 steps, so every doomed seed is ADMITTED.
   Raising the threshold is not a fix -- it would only have to beat one bounce cycle. The
   criterion must become "GROUNDED ON A PLATFORM when the window ends".

2. **CORRECTION to the recorded span.** `standable_span`'s docstring records floor 12 as
   `188 .. 228 = x_min+4, x_max-4`. The right end is wrong: px 228 falls 0/8, px 224 stands
   8/8. Usable span is **188..224**. (The left end, 188, is confirmed -- two independent
   measurements now agree, which is the main reason to trust it.)
   ```
   FLOOR 12, tile extent 184..232     walking LEFT   walking RIGHT
     184                                  0/8            0/8      falls
     188 .. 224                           8/8            8/8      OK
     228, 232                             0/8            0/8      falls
   ```

3. **Approach direction does NOT affect standability.** Tested because the foot row is
   asymmetric (centre-6..centre+2), so a mirrored sprite could plausibly change it. It does
   not: 8/8 both ways at every usable px, 0/8 both ways at every bad one. Hypothesis
   rejected. (No RAM-write exists on the interface, so direction-INDEPENDENT standability
   is not measurable; "can the agent BE here and survive" is the criterion that matters for
   anchor placement anyway.)

4. `Low1`'s anchor (px 232) is ALSO outside the span, yet its pool is HEALTHY (96 seeds at
   px 220, 4 at 224) because the agent falls at 228 before it can reach 232. **A lethal
   anchor can be harmless**, so "this anchor is wrong" does not imply "this is the blocker".

### DECIDED, not yet applied

* Re-land `Low2_launch` px 184 -> 188, and make the comment's tense match the code.
* `Low1` px 232 -> 224. Harmless today, but the potential aims at an unreachable pixel.
* Correct `standable_span`'s docstring: floor 12 right limit 228 -> 224.
* Survival gate: require grounded-on-platform at window end. Widest blast radius (every
  pool, every level), so check how many L1/L2/L3 seeds it would newly reject.

### NOT VERIFIED

* That any of this improves anything. `Low1` shows a lethal anchor can be inert.
* That the crossing is even POSSIBLE from px 188. An earlier claim here comparing rope-1's
  grab distance to rope-2's was RETRACTED: ropes move between frames, so single-frame pixel
  measurement does not support it.
* **Cheapest falsification, before any GPU:** from px 188, sweep scripted jump timings and
  ask whether ANY sequence crosses. If none does, the anchor is not the binding constraint.

Tools: `debug/l4_crossing_trace.py` (per-step reward trace + video),
`debug/l4_low2launch_fix_visual.py`, `debug/l4_pull_direction.py`.
Figure `experiments/003-yeti/evidence/l4_rope2_geom/low2launch_fix_v2.png`;
clip `experiments/003-yeti/evidence/l4_rope2_fromreset/rope2_failed_ep0.mp4`.


## v13 RESULT (2026-09-11) — PASS as "not worse", and there is now a champion for current code

v13's purpose was an ARTIFACT, not a hypothesis test: current code had no champion
because v6's was built at `bc85424`, before the censused anchors (`dc8e5a0`), the
re-landed `Low2_launch` px 184 -> 188 fix (`9802fbe`) and the predecessor gate
(`f069429`). It ran 15M on v4's warm start with seed 42 — v6's exact warm-start source,
which is what the v10 RESULT section asked for so that v6 becomes a legitimate control.

**The stick changed, so v6 was re-swept on current code.** `dc8e5a0` moved the
`Rope1`/`Spring`/`Step` anchors, and route depth is scored through those anchors, so
v6's published row is not apples-to-apples with v13. Both runs have exactly 150
snapshots; both were swept at 12 episodes, 1 fruit, `--level 4`, the L4 start state,
stall 40, max-steps 1500. The control went to `best_restick/` so v6's historic
`sweep_state.json` is preserved.

```
                              n    mean   median    max    >=9.5    <1.0
v6  (published, OLD stick)   150   6.37    7.88    10.00   8/150   10/150
v6  (re-swept, CURRENT)      150   6.48    7.88    10.00  11/150    9/150
v13 (CURRENT)                150   7.03    7.88    10.00  12/150    9/150
```

The stick change alone is worth +0.11 mean; v13 is +0.55 over the same-stick control,
with identical median and max, one more snapshot >= 9.5 and one fewer collapse. By this
file's own rule — at-or-near v6 reads as "not worse", only a repeated multi-seed win
reads as "better" — **v13 is a PASS for "current code is not worse"**. It is NOT
evidence of "better": one seed, and v6 seed 42 is a known 1-in-4 outlier.

### The champions, re-eval'd at n=300 — and the n=12 pick does NOT hold up

Both champions read rung 10.00 at n=12, which cannot separate them, so both were
re-eval'd at n=300 on the same stick per the trigger-vs-measurement rule above:

```
                          n     mean_rung   princess    >=rung 10      >=rung 11
v13         1.3M         300      8.49       0/300     190 (63.3%)    1/300
v6-restick  2.1M         300      8.52       0/300     226 (75.3%)    0/300
```

**v13's champion is not better than v6's.** Mean is a tie (8.49 vs 8.52) and v6's holds
rung 10 far more reliably (75.3% vs 63.3%). This is the trigger-vs-measurement warning
playing out in full: a 12-episode selection picked a v13 snapshot that does not hold its
rank at n=300. So the DISTRIBUTION result (v13 7.03 vs v6 6.48 over 150 snapshots) and
the CHAMPION result point different ways, and only the distribution supports "current
code is not worse". Do not quote v13's champion as an improvement.

**RUNG 11 IS PASSED IN A FROM-RESET CHAMPION EVAL, once.** Rung 12 had been touched
during v6's TRAINING before (see the v6 table row), but every champion eval on record
read rung 11 at 0/300, and this file concluded from that it was "a hard barrier, not a
low-probability crossing". That conclusion is now qualified — see the superseding note in
the Status section. v13's champion, episode 68 of 300 from reset,
reached **rung 12** — credited BOTH `Low2` and `Lhi_down_bot`, i.e. it crossed rope 2 and
arrived on floor 13 — then died after 533 steps at (x_ram 35, y 94). Full credit list:
`F1, Fr1, Fr1_launch, Fr2, Fr2_launch, Lascent_top, Lclimb1_top, Lclimb2_top,
Lclimb3_top, Lfruit_bot, Lfruit_top, Lhi_down_bot, Low1, Low2, Low2_launch, Rope1,
Rope1_launch, Spring, Step`. One episode in 600 across both champions, so ~0.3% from
reset — a real crossing, not a ceiling. Note the n=12 sweep could never have shown this:
a mean over 12 episodes cannot surface a 1-in-300 event, which is why the earlier
reading of "no snapshot exceeds rung 10.0" said nothing about whether the wall is
passable.

Princess remains 0 everywhere: 0/300 for both champions and 0 across all 300 snapshot
evals. v13's good snapshots are scattered (1.3M, 5.6M, 8.1M, 8.9M), the same oscillation
v6 shows, so the champion is again an early transient and NOT a policy the run converged
to. Remember "DO NOT WARM-START FROM A CHAMPION": v13's champion must not seed the next
run.

## ROPE 2: THE SCRIPTED FALSIFICATION, RUN (2026-09-11)

This answers the "NOT VERIFIED / cheapest falsification" item above — from px 188,
sweep scripted timings and ask whether any sequence crosses — with the stated
conclusion: **the anchor is not the binding constraint.** Scope limit up front: three
plan families, ~700 trials, on v13's pools. Not exhaustive over all input sequences.

Plan families tried, all from grounded floor-12 pool seeds (`Low2_launch` px >= 188 and
`Low1`), classified against `Low2`'s real credit window px 104..152 (anchor px 128,
`jump_waypoint_tolerance` 6 RAM = +-24 px — NOT the ladder tolerance of 2):

```
A  wait W (0..30) on the pad, then hold jump-left   reaches floor 13, lands px 88 (80/89)
                                                    or px 92 (9/89) -> 0/89 in the window
B  hold jump-left K steps, then neutral             nothing reaches floor 13 below K=26
C  run-up: N (0..14) left steps from Low1, then     0 crossings. jump-left and
   hold jump-left / jump-upleft / jump-up           jump-upleft died 6/6 at EVERY N,
                                                    including N=0 from px 224
                                                    <-- RETRACTED, see below
```

`pose 14` (the rope carry) occurred **0 times in ~700 trials**.

**RETRACTED 2026-09-29 — that was the wrong pose to look for, and both conclusions drawn
from it are false.** Pose 14 is the RIGHTWARD carry, which is rope 1. Rope 2's carry is
pose **15**, and on 2026-09-11 pose 15 was still uncatalogued — the trainer was reporting
it as `UNCATALOGUED POSES 15x5901` on every status line. So "0 occurrences of pose 14" was
never evidence about rope 2 at all, and it was used to conclude two things that are both
wrong:

* *"Every floor-13 arrival was the trampoline loop, not the rope."* No. Re-measured from a
  px-188 seed: the arrival shows pose 15 twice at y 62 with px running 188 -> 88, and
  poses 16/17 never appear. It is the rope. Family A's 80-of-89 px-88 landings were real
  crossings, discarded only because px 88 falls outside `Low2`'s credit window px 104..152.
* *"C's result is the giveaway — holding a jump input just marches the agent off floor 12's
  left edge."* No. Family C's grid swept run-up length N but had **no wait term**, so it
  never swept the rope's phase, and N=0..14 could not reach the launch pixel anyway: the
  leftward walk cycle stalls (dx in {-4, 0}), so from `Low1`'s px 220 it takes **12** steps
  to reach px 188, and 14 steps walks off the edge. Re-measured with a phase sweep, the
  run-up crosses: **26 of 136 plans**, run-up 11..13 then wait 16..24 then hold jump-left,
  from a `Low1` seed at px 220.

HOW THE ZERO SURVIVED SCRUTINY, worth not repeating: an approach that overshoots the edge
does not fail loudly. It falls onto the spring, which delays death to a median of step 82,
so the attempt spends most of its budget bouncing and then reports a death like any
mistimed jump. Nothing in `jump_bruteforce`'s output distinguished "the jump missed" from
"the agent was never on the pad". It now prints the launch pixel of every plan, and
re-running the misleading grid says `launch pixels this grid actually used: {196: 34,
200: 68}` — visibly never 188. A zero from this tool is only meaningful once you have
checked WHERE the plan actually jumped from.

The 11 genuine `Low2` captures sit at px 104..124, y 66..70 — inside the credit window,
but they are captured AFTER the agent walks right from its landing, not at touchdown (the
three observed crossings touched down at px 88, 92 and 120, so only one was credited on
landing). An earlier version of this sentence read "the real crossing DOES land in the
window", which those three landings do not support; corrected 2026-09-29.

It demonstrably happens in training — 21 episodes that started UPSTREAM of rope 2 were
credited `Low2` (9 of them from reset), and one reached `Lprincess_top` from reset at
step 1,607,256 (633 steps, reward 63.59). This sentence used to end "just not by any
script tried here"; scripts DO reproduce it, see the retraction above.
Earlier wording in this session's analysis claimed the crossing "never happened"; that
was wrong and is retracted.

**New measurement worth acting on: the shaping's optimum on floor 12 IS the lethal
pixel.** Path distance to `J12_13_b` from floor 12 falls monotonically leftward and
bottoms out at px 184 (56, against 60 at px 188). Stepping 188 -> 184 is the ONLY
positive shaped reward anywhere on that pad (+0.12 measured with the trainer's own
reward); holding position pays 0.000 and the whole 85-step fall-bounce pays 0.000.
Lethality, measured on a hazard-free floor (no snowballs, no kangaroos on floor 12, so
an idle death there can only be a fall):

```
px 184   12/12 die (walk there and stop)   20/20 die (pool seeds)   median death step 82
px 188    7/20 die
px 192    0/11
px 196     0/1
```

Death at step 82 against `min_survival_steps` 30 is why these captures are admitted, and
61 of 100 v13 `Low2_launch` seeds sit on px 184. This is direct support for the
lethal-margin admission filter already proposed above, and it supplies the threshold.
Note the filter must be admission-side: the comment in
`train_checkpoint_curriculum.py` records that narrowing DETECTION took `Low1` and
`Low2_launch` from 0.61 to 0.00.

Method traps hit while measuring this, worth not repeating:

* `build_training_env(...).gym.step` returns reward 0 for every action. The trainer does
  NOT use the gym's reward; it builds a `RewardContext` per step and calls
  `reward_fn(ctx)`. Any reward probe must reproduce that, INCLUDING
  `restore_reached_waypoints` and the curriculum's `fruit_presence_addrs` (defaulting to
  the level-1 dict makes the reward chase a stale fruit and raise `KeyError 'F3'`).
* A persistence check for "can the agent stand here" needs a window LONGER THAN ~82
  STEPS. A 10-step idle hold reports px 184 as standable, the same blind spot that lets
  doomed px-184 captures through admission.
* `Platform.x_min`/`x_max` are a LOGICAL walkable line that may span jumpable gaps by
  design. Diffing them against measured standable spans does not reveal bugs; an
  apparent 13-floor mismatch built this way was meaningless and is withdrawn.

Tools added: `scripts/mo5/yeti/diag/l4_pad_reward.py` (trainer-identical reward on a pad),
`debug/l4_pad_wait_sweep.py`, `debug/l4_pad_hold_length.py`, `debug/l4_pad_phase_seed.py`,
`debug/l4_low2_landing.py`, `debug/l4_rope_compare.py`, `debug/l4_rope2_runup.py`,
`scripts/mo5/yeti/diag/l4_platform_audit.py` (note its hold is too short, see above).

## PLAN OF RECORD (2026-09-14): THE REWARD CANNOT MARK A JUMP LANDING

### What led here

`de21939` made sprite overlap the default reach test for BOTH detection and capture. The
tolerance box had no correct value: 12 pairs of L4 capture boxes overlapped where the agent
can stand (up to 33 px -- the whole Hi chain, plus `Lhi_down_bot`+`Low2` and
`Lclimb1_top`+`Rope1_launch`), so one grounded frame credited two waypoints; and a box
narrow enough not to overlap missed real arrivals. Sprite overlap gives ZERO standable
overlaps because the region is the sprite's own 14x18 instead of 49x13.

**v14 (cold, 2M, sprite, seed 42, 57m34s) confirms sprite capture works.** It is the first
L4 run with no inherited weights and no inherited pools, so it is the first observation of
where sprite capture actually puts seeds. Every pool lands on its anchor:

```
                    v13 (box capture)              v14 (sprite capture)
Fr1_launch      px 208, y 162  (24 px off, ladder)   px 228, y 158  (4 px)
Rope1_launch    px  32, y 122  (24 px off, ladder)   px  52, y 118  (4 px)
Rope1           px 124         (16 px off)           px 112         (4 px)
Step            px 256         (16 px off)           px 268         (4 px)
Fr2             px 308         (12 px off)           px 300         (4 px)
```

Max |dx| is 4 px for every one of the 14 pools it filled. Reach held to `Spring` 0.55, then
`Step` 0.01 and `Lclimb3_top` 0.00; the deep route stayed empty, as predicted for a cold
2M run. Figure: `experiments/003-yeti/evidence/l4_v14_seeds/v14_anchors_and_seed_heads.png`.

### The defect

Only MANDATORY targets are summed in the reward, as eleven OR-groups. All five mandatory
JUMP LANDINGS -- `Fr1`, `Rope1`, `Spring`, `Step`, `Low2` -- are points the agent flies
through. Measured over 120 from-reset episodes with v14's policy, comparing what the sprite
test fires on against what the reward actually marks:

```
target   px,y       sprite fires   reward marks   poses at which it fires
Rope1   108,118      107/120          2/120       9(jump-right)x126, grounded x5
Spring  212, 94       95/120         14/120       9(jump-right)x321, 10(jump-left)x53
Step    272,102        3/120          1/120       9(jump-right)x20
```

`Rope1` is inside the agent's body in 107 of 120 episodes and marked in 2. The cause is
that the marking loop sits AFTER the airborne early return, so it only ever sees frames in
the reward's own `SURFACE_POSES`; 576/576 observed marks were on a grounded pose. A landing
is precisely when the agent is airborne. The comment claiming marking is "UNGATED on
purpose" is true of the loop and false of the code path.

**Consequence, computed on floor 12 (the rope-2 launch platform):**

```
                       px 184    px 232     the shaping pulls
groups 0-7 marked         320       368     LEFT, toward rope 2 and the princess
4,6,7 unmarked (real)     984       888     RIGHT, back the way it came
```

Four backward terms (`Rope1` 320, `Spring` 200, `Step` 144, `Lclimb3_top` 96 at px 184)
outweigh the two forward ones. So an agent on the rope-2 pad is pulled AWAY from the gap by
milestones it physically passed but never got credited for. That is a better explanation of
the floor-12 wall than anything else measured, and it fits the otherwise-odd finding that
the only positive shaped reward on that pad was +0.12 for a single leftward step.

Note this is the same failure already on record for `Fr1` and `Step` -- "a MANDATORY
milestone that had never been markable since v4, so its distance term never switched off"
(`dc8e5a0`) -- reappearing for a different reason.

### The three steps, in order

1. **Move marking above the airborne early return.** Keep the `{11, 12}` blocklist so a
   FALL past a waypoint does not mark it (measured: `BLOCKLIST` and `ANY` are identical for
   every mandatory target, so the blocklist costs nothing). Leave the shaping freeze
   completely untouched -- returning `Phi=None` while airborne deleted the return-leg debt
   and PPO farmed it for 15M steps (H-AH). Only marking moves.
2. **Verify WITHOUT training.** Re-run `debug/l4_reward_marks.py` against v14's own policy.
   The marking rates are near-deterministic, so this settles whether the mechanism works,
   for ~5 minutes instead of ~55. Expect `Rope1` 2 -> ~107/120 and `Spring` 14 -> ~95/120.
   If it does not move, the fix is wrong and no run is justified.
3. **Then one training run**: cold, 2M, seed 42, identical to v14 except the fix, so v14 is
   a genuine control -- same cold start, same sprite mode, same seed, ONE lever. It will
   show whether the agent now gets past `Spring`/`Step`. It CANNOT establish "better": one
   cold seed, and v6 seed 42 is a known 1-in-4 outlier.

### Deliberately NOT in this change

A new OR-group with one member per route -- `Low2_launch` (px 188, f12) and `Hi2` (px 224,
f16) -- to give the low route a mid-course milestone. Measured: it doubles floor 12's
leftward gradient (48 -> 96 across the platform), and it does NOT force a route, because
the two members swap which is nearer at f11 px 296, exactly where route B's `Lhi_up` ladder
is. It is a REWARD change with no measured failure behind it, so it is a separate lever
after the marking fix is judged.

Also settled while looking: **floor 12 has no mandatory target by necessity, not oversight.**
Route B (f11 -> f14 -> Hi chain -> f13) never touches f12, so mandating it would force
route A. That is what group 9's OR `[J12_13_b, Lhi_down_bot]` exists to avoid. An earlier
reading of this as an omission is withdrawn.

### v15 RESULT (2026-09-14) — THE FIX WORKS AND THE RUN IS WORSE

Step 3 ran: cold, 2M, seed 42, byte-identical to v14 except marking moved above the
airborne return. 58m34s, exit 0. v14 is a true one-lever control.

```
matched at 2M          v14 reach   v15 reach   delta    v14 pool  v15 pool
Lfruit_top                  0.95        0.91   -0.04        100       100
Fr1                         0.92        0.79   -0.13        100       100
Lascent_top                 0.72        0.55   -0.17        100       100
Lclimb1_top                 0.71        0.53   -0.18        100       100
Rope1                       0.65        0.25   -0.40         88       100
Lclimb2_top                 0.61        0.16   -0.45        100       100
Spring                      0.55        0.08   -0.47        100        50
Step                        0.01        0.00   -0.01         54         3
Lclimb3_top                 0.00        0.00    0.00          9         —
```

Not noise on one waypoint: a MONOTONE decline that widens with depth, -0.04 at the top to
-0.47 at `Spring`. `Step`'s pool fell 54 -> 3 and `Lclimb3_top`'s 9 -> empty.

So the mechanism did exactly what it was built to do and the outcome went the other way.
`Rope1` marking went 2/120 -> 107/120 (verified on v14's own policy before the run) and
`Rope1` reach fell 0.65 -> 0.25.

**Both of these are true at once, and neither cancels the other:**

* The defect is real. An unmarked group stays in the sum and reverses floor 12's gradient
  (984 at px 184 vs 888 at px 232, i.e. away from the gap).
* Removing the defect made this run worse at every depth.

**NOT explained.** A candidate, recorded as a hypothesis and NOT as a finding: marking now
happens mid-jump, so the `active_wp` change forces the rebaseline onto the LANDING frame,
discarding that landing's shaping delta. Before, mark and landing coincided on one grounded
frame. On the shared trunk the five ladder groups and `Fr1` were ALREADY marking fine at
111-117/120, so the fix bought them nothing while possibly moving their rebaselines — and
the trunk is exactly where the decline starts. Untested.

**What this run does NOT test.** Neither v14 nor v15 reached `Spring` in a state where the
fix could pay off, and neither reached floor 12 at all. So it measures only the collateral
cost on the trunk, never the benefit it was built for. A cold 2M run cannot reach the wall
the fix targets; testing that needs a warm start from a policy that already holds
`Low1`/`Low2_launch`, which is what v13's lineage had and this cold pair does not.

Also one cold seed, and seed 42 is a known 1-in-4 outlier, so "worse" here is one sample.

## v16 (2026-09-15/16): THE v15 HYPOTHESIS WAS RIGHT, AND FIXING IT BREAKS THE WALL

The v15 section above closes with an untested candidate: *marking now happens mid-jump, so
the `active_wp` change forces the rebaseline onto the LANDING frame, discarding that
landing's shaping delta.* That is now measured, and it was the whole story.

### v16a vs v16b — the A/B at 6M, one lever (`mark_airborne`)

Cold, 6M, seed 42, sprite reach, configs identical but for that flag. Final `reset_reach`,
and the mean/max over the 4M–6M window (401 progress lines per arm):

```
                        idx7        idx8 Spring   idx9 Step    idx10 Lclimb3_top
v16a mark_airborne 0   0.64/0.81   0.62/0.81     0.10/0.25    0.00/0.02
v16b mark_airborne 1   0.57/0.84   0.03/0.10     0.01/0.05    0.00/0.00
```

Trunk matched within ~0.05; `Spring` collapsed 0.62 → 0.03. So marking a jump landing is
harmful ALONE, which is what v15 saw at 2M, replicated at 6M with room to peak.

### The mechanism, measured rather than argued

Method: record ONE trajectory from v16a's policy, then replay the identical action
sequence through two reward instances differing only in `mark_airborne`
(`scripts/mo5/yeti/diag/l4_spring_trace.py`). Same states, same order, so any divergence
is the lever. 40 episodes, 29 of which cross climb2 → Spring:

```
floor-9 arrival (climb2 -> Spring)   +3.200 -> 0.000    29/29, zero variance
floor-7 arrival (Rope1)              -5.120
floor-10 arrival (Spring -> Step)    unchanged
```

Three rules collide, each sane alone:

1. Reward pays the CHANGE in summed distance to targets still on the list.
2. The D2 freeze pays nothing airborne and HOLDS `prev_phi`, so the landing frame pays
   everything covered since take-off.
3. A target-set change skips that frame's payment (it cannot tell earned progress from an
   entry leaving the sum) and only rebaselines.

Marking mid-air hits `if airborne: return reward` BEFORE `_prev_active_wp` is updated, so
the change is first noticed on the next grounded frame — the LANDING. Rule 3 therefore
eats the single most valuable frame of the episode. Under the old placement the mark lands
on an ordinary walking frame worth ~0.2 (measured: step 335 of the traced episode paid
+0.000 against −0.200 unskipped, i.e. the skip was worth ±0.2, not 3.2).

**climb2 → Spring is the worst possible frame to lose.** It is not a jump: floor 8 (y 94,
px 120–168) and floor 9 (y 94, px 200–232) are separated by a gap at px 168–200 with the
trampoline (floor 24, y 142) directly beneath, and the route goes OVER it. Measured: 29/29
crossings contain a trampoline pose (16/17), and the arc is 61 steps of frozen shaping
banked into one frame.

### The fix: `pay_on_target_change` (default False)

Price a frame whose waypoint list changed against the list it STARTED with
(`_prev_active_wp`, measured at the CURRENT position, so only movement is charged), then
rebaseline to the new list. Nothing is paid for a deletion, nothing earned is lost, and
marking POSITION stops mattering. Scope: the waypoint arm of the D3 `if` only — death,
fruit pickup, deferred-fruit credit and princess still skip, because pricing the old
fruit set means reconstructing which fruits were uncollected.

Same harness, both arms marking airborne so the new flag is the only difference:
floor-9 arrival **0.000 → +3.920, 29/29**, and only **3 frames of 359** differ from v16b.
Pinned by `tests/python/test_reward_airborne_marking.py`: the landing pays exactly what a
run that never marks the group pays (+0.3200 in the synthetic case), which is the real
property — paid for movement, never for the deletion.

### v16c (6M) — the wall at `Step` → `Lclimb3_top` breaks

`mark_airborne: true` + `pay_on_target_change: true`, otherwise identical. Final route
table, reach from reset:

```
route point      v16a   v16b   v16c
Rope1            0.69   0.61   0.69
Lclimb2_top      0.64   0.59   0.55
Spring           0.62   0.04   0.52
Step             0.15   0.01   0.48
Lclimb3_top      0.00   0.00   0.39     <- never reached by ANY cold run before
Low1             0.00   0.00   0.00
route depth        13     13     17
```

4M–6M means: `Spring` 0.55, `Step` 0.40, `Lclimb3_top` 0.19 (peak 0.51). `Lclimb3_top`
first exceeded 0.05 at step **4,845,000** and the sweep's champion is
`model_5000000_steps.zip` — the run was still climbing when the budget ended and the best
snapshot IS the breakthrough, not a plateau.

### But it is NOT yet as good as v6/v13

Champion re-eval, n=300 stochastic from reset, same start state, same script as the
v13/v6 pair above:

```
champion              mean rung   >= rung 10
v6-restick 15M warm    8.52/13    226/300  75.3%
v13        15M warm    8.49/13    190/300  63.3%
v16c        6M cold    7.91/13    135/300  45.0%
```

Not like-for-like, all three ways against v16c: 6M vs 15M, cold vs warm, and best-of-60
snapshots vs best-of-150. **v13's 15M also sits on top of v4's 15M of weights**, so
matching v13's total training is nearer 30M than 15M.

### Three differences from v6/v13 that ride under every v16 number

* **warm vs cold** — v6/v13 both `resume:`; the v16 series is cold (deliberate: a warm
  policy was trained under the old reward and would conflate the comparison).
* **box vs sprite** — v6/v13 ran the box reach test.
* **`target_kl: 0.07`** — set in v6 AND v13, **absent from all three v16 configs**. Not
  dropped on purpose by any decision recorded here; it was already missing from v16a/v16b.
  UNEXPLAINED, and its own lever.

### In flight

`yeti_curriculum_l4_v16c_payonchange_cold_15m` — v16c's config with ONE line changed
(`timesteps`), which also yields 150 snapshots and so removes the selection handicap. At
40% it is already past where the 6M run finished: `Lclimb3_top` 0.47 vs 0.39, `Spring`
0.77, `Step` 0.68. `Low1` still 0.00.

## Open questions, in priority order

The single wall is now **rung 10 → 11 = reach floor 13**, measured at 0/300 for v6's
champion but **1/300 for v13's** (2026-09-11) — so it is a crossing at ~0.3%, not zero.
Confirmed independently at 3/1071 in a from-reset hunt; pooled 4/1371 = 0.29%.
Everything before it is at 84.7%.

1. **Pick v9's lever.** Waypoint geometry is now sound (boxes 21 -> 0, pools verified
   clean at 100k), so the next run is the first one in a while whose seeding is not
   defective. Two honest candidates:
   * poses 6/7 into `SURFACE_POSES` — 1857 discarded grounded left-walk frames per 100k,
     affects every level, and L4's closing stretch is leftward. Biggest expected effect,
     but a reward change on top of one we just made.
   * simply run the corrected geometry at 15M and get a clean baseline to compare
     against. Slower to inform, but it is the run that tells us what the fixes bought.
2. Where does the rope actually DEPOSIT the agent on floor 13? Every landing on record
   is inside an existing detection box, so the distribution is self-selecting and
   cannot be read off the pools. Roll out from floor-12 seeds and log x whenever the
   agent reaches floor 13's standing y in a surface pose, ignoring all boxes. Decides
   whether `Low2` at px 112 sits where arrivals actually happen.
3. Does the snowball on floor 13 gate the landing, once the agent gets there often
   enough to measure? Confirmed present and lethal, but with rung 11 at 0/300 there is
   no arrival sample to measure the hazard phase against. Revisit after v7.
4. **Admit poses 6 and 7 to `SURFACE_POSES`** — its own lever, and probably the biggest
   single one available, because it affects every level and every leftward approach (54%
   of grounded left-walk frames currently discarded). It is a reward change, so it
   invalidates champions as warm-starts; see the pose-catalogue section. Sequencing
   question: it interacts with the `Low2_launch` fix, since that pool was being filled
   from only the pose-4/5 subset of leftward frames.
5. Identify pose 15 (found by the new uncatalogued-pose check, 4 occurrences in a
   3000-step run) and confirm the 4/5/6/7 left cycle on L1–L3 as well as L4.
4. Does gate hysteresis unlock the two full-but-unused pools? One lever, and it
   serves L3 as well. v6 spent roughly a third of the run in collapse (9 episodes of
   300k–900k steps; 10/150 snapshots below rung 1), which is what the `reach_threshold`
   0.15 hard cutoff produces. L3 v16 prescribed this and it has still never been run.
5. Why did SPRING replay cross 0/8 here when training reach is 0.89? Probe used
   v4's `final_model.zip`, which is degraded — re-run from a mid-run snapshot
   before reading anything into it.
6. Run the scripted jump grid on exactly the 9 ALWAYS-lose clean seeds (see Q0). If
   a script wins from them, the f11 shortfall is entirely the policy's and nothing
   about those states needs changing.
7. Fix the depth proposer in `scripts/mo5/yeti/diag/yeti_validate_targets.py`: it proposes anchors at
   transient positions (it suggested `Rope1`→(34,110), mid-rope-carry, and
   `Low1_launch`→(66,86), mid-climb). Constrain suggestions to the floor's standing y.

## v16c AT 15M: pay_on_target_change VALIDATED, and the wall is ARRIVAL TIMING (2026-09-18)

### The champion result — a cold run matches the warm incumbents

`keep_best_sweep` over 150 snapshots then `eval_from_reset` at 300 stochastic episodes,
the same stick as the v6/v13 pair above:

```
champion                 mean rung   >= rung 10    training
v6-restick  15M warm      8.52/13    226/300 75.3%  15M on top of earlier weights
v13         15M warm      8.49/13    190/300 63.3%  15M on top of v4's 15M
v16c        15M COLD      8.49/13    219/300 73.0%  15M from scratch
v16c         6M cold      7.91/13    135/300 45.0%
```

So the 0.6-rung gap reported at 6M was STEP COUNT, not the reward change. v16c cold now
equals v13 on mean rung, beats it on the rung-10 rate, and does it with roughly half the
cumulative training. Champion is `model_9300000_steps.zip`; the run peaked over 9M-12M
(`Lclimb3_top` 0.49 mean / 0.77 max) and DECAYED to 0.39 by 15M, so the final route table
understates it badly — read the window, not the endpoint.

### THE WALL IS NOT THE JUMP. It is when the agent arrives on floor 11.

Measured, in this order, each step killing the previous explanation:

**The jump is easy.** `diag/jump_bruteforce.py` sweeps scripted plans (approach A steps,
wait W, hold jump H) from `Lclimb3_top` seeds. Best plan lands floor 12 in **12/20 = 60%**:
walk left 8 steps, then jump left immediately. `hold` is irrelevant (2/4/6/8/10 all give
12/20) so the jump commits at takeoff; `wait=2` collapses it to 3/20. Takeoff is px 252,
inside floor 11's declared edge, landing at px 224 — exactly `Low1`'s anchor.

**No policy comes close.** Three champions on the IDENTICAL 20 seeds, 8 repeats:
v6 0.14, v13 0.07, v16c 0.03, and **not one always-win seed in 60 pairs**. Variance is
mostly WITHIN-seed, so the policy has agency; the skill is learnable and unlearned.

**Floor 11 is lethal ground and nothing else nearby is.** NOOP survival (hold NOOP, count
frames to death) with the cap raised past the bonus timer:

```
                        exposure                  NOOP frames
Step         floor 10   not on the patrol route   ~1200 (the timer)
Lclimb3_top  floor 11   exposed, no ceiling       3-13   ALWAYS dies
Low1         floor 12   landing zone, sheltered   ~1170 (the timer)
Low1 at px 228          right edge, NOT sheltered 11     kangaroo reaches it
```

**And here is the mechanism.** Arrival survivability on floor 11 from reset, per champion:

```
          NOOP frames of life at the moment of arrival
v6        min  9   median 12   max 15
v13       min  0   median 12   max 14
v16c      min  0   median  3   max  7
```

The approach needs 8 steps. v6/v13 arrive with 12 frames and can execute it; v16c arrives
with 3 and is DOOMED ON LANDING whatever it does. That is the entire difference between
walling at rope 2 and walling at floor 11. **The skill is timing the ladder climb**, and
arrival survivability is the quantity to track — with a hard threshold at 8.

Confirmed on video: from reset the champion tops the ladder at step 368 and is crushed at
step 372, at px 264, grounded, mid-walk, 12 px short of takeoff. 31 of 40 episodes reach
floor 11 and then die. Clips + filmstrips via `diag/record.py --want "reach=X+died"`.

### The seed pool at Lclimb3_top is 2.7x easier than reality

Same position, pool seeds vs the policy's own reset arrivals:

```
POOL    n=40   median 8 NOOP frames
RESET   n=27   median 3 NOOP frames
```

`admit_requires_survival` only admits states the agent SURVIVED, so it keeps the benign
hazard phases and discards exactly the hard ones. Practising there under-trains the
situation the policy meets, and lets it skip the decision that matters. This is the one
like-for-like comparison in this section (same position, two phase distributions); do NOT
compare NOOP numbers ACROSS positions, see the next subsection.

### THREE DIFFERENT THINGS, and I conflated them for an afternoon

* **exposure / safety** — structural: does geometry shelter this spot, is it on a hazard's
  path? A property of the map, timeless.
* **survivability** — a property of a STATE: position PLUS where the hazard is right now.
  NOOP survival measures only this. An exposed spot scores the cap whenever the hazard
  happens to be far away, so a HIGH number proves nothing about safety; only a LOW number
  is informative ("this state was doomed"). And any cap shorter than the hazard cycle
  truncates: a bimodal result (a few early deaths, the rest at exactly the cap) means the
  cap is too small. The ~1150-1210 cluster is the BONUS TIMER, not a hazard.
* **admission ratio** (`cap/rej` in the route table) — did the agent survive 30 steps
  ONWARD, which includes danger at LATER waypoints. `Step` scores 10.4% while being
  completely safe, because the agent climbs the ladder and dies on floor 11 inside the
  window. That is correct behaviour for seed admission and useless as a safety index.

### Levers refuted by measurement, not opinion

* **A reward milestone on the floor-12 landing** (the deferred `[Low2_launch, Hi2]`
  OR-group). The median reset arrival cannot survive long enough to depart for floor 12,
  so this pays more for something unreachable. Note the shaping ALREADY pays the jump:
  +0.48 walking `Lclimb3_top` -> the edge, +0.64 for the landing, and g9's nearer member
  is the low route at every position on floor 11.
* **`credit_requires_survival`**. Under `gamma: 1.0` shaping telescopes to
  Phi(end) - Phi(start), so refunding it on death zeroes shaping for every episode that
  dies. L3 v14 did this: 17,024 of 17,033 episodes, chain collapsed, princess 89 -> 9.
* **A flat death penalty.** Measured: the death frame pays exactly 0.000 and a fatal
  episode still banks +66.300, so dying IS free. But 99.5% of episodes end in death
  (74,612 of 75,015), so a constant terminal penalty is a per-episode constant and PPO
  normalises it away. The only residual is PPO's 0.99 discount making LATE deaths cheaper,
  which rewards dawdling on safe ground — a worse failure than the one it fixes.

### What survives

Stop seeding `Lclimb3_top`; let `Step` carry that segment. `Step` is the same px (272) one
floor down, measured safe, and its pool and reset arrivals are identical — so the timed
climb happens INSIDE the episode instead of being skipped. One curriculum line, no reward
risk. It makes practice representative; it adds no signal that arrival timing matters, so
whether it is sufficient is unknown.

Minor, same class: `Low1`'s sprite window spans 14 px (~px 217-230). The right end is
outside the shelter, so a seed captured at px 228 is dead on load whatever it does (1 of 8
in this pool). `admit_requires_survival` missed it because the agent walked left off 228
inside the 30 steps.

### STILL OPEN: what gave v6/v13 the arrival timing?

They arrive with 12 frames, v16c with 3, and the map and reward are IDENTICAL —
`Low1_launch` was skipped before all three runs (`bc85424`, 2026-08-27, predating v6 on
08-28 and v13 on 09-10), so no launch pad, no floor-12 milestone and free death in every
case. Three candidates remain, none isolated: **warm vs cold** (v6/v13 both `resume:`, so
they inherited weights that had already practised this), **box vs sprite** reach geometry,
and **`target_kl: 0.07`** which v6/v13 set and all three v16 configs omit.

## v17 / v18 (2026-09-19): v18 IS CHAMPION, AND EVERY COLD RUN WAS MIS-SPECIFIED

Two 15M runs, each ONE lever from v16c's 15M, launched together because they answered
DIFFERENT questions -- v17 was the candidate fix, v18 the diagnostic. They are not two
arms of one comparison and must not be compared to each other.

```
v17 = v16c + curriculum.seed_waypoint_skip: [Lclimb3_top]        (cold, as v16c)
v18 = v16c + resume: <v13>/final_model.zip, warmstart_weights_only  (warm)
```

### v18 IS THE CHAMPION, on the shared stick

`run_champion_eval.sh` (sweep 150 snapshots at n=12, then eval_from_reset n=300
stochastic from the L4 start state -- do not edit those flags, they are what make these
numbers comparable):

```
champion                 mean rung   >= rung 10    princess
v6-restick  15M warm      8.52/13    226/300 75.3%   0/300
v13         15M warm      8.49/13    190/300 63.3%   0/300
v16c        15M cold      8.49/13    219/300 73.0%   0/300
v18         15M warm      8.96/13    243/300 81.0%   0/300   <-- best on both
```

Best mean rung and best rung-10 rate on this level, +0.44 rung over the previous best,
and **the only policy that reaches floor 12 from reset at all**: `Low1` reach 0.42 and
`Low2_launch` 0.40 against 0.00 for every other run. `Lclimb3_top`'s `prog` -- the
hand-off that was 0.00 in v16c -- is 0.95. Pools: `Low1` 100, `Low2` 11,
`Lhi_down_bot` 100, and `Lprincess_top` 13, i.e. seeds at the princess ladder for the
first time. Princess itself still 0/300.

### v18 did NOT learn the arrival timing. It arrived holding it.

`diag/l4_seed_determinism.py --over-snapshots`, arrival survivability at `Lclimb3_top`
(NOOP frames of life on arrival; the floor-12 approach needs 8):

```
v18 (warm)    100k: 10   1.9M: 11   3.7M: 11   7.3M: 11   10.9M: 10   15M: 9
v16c (cold)   3.7M:  0   5.5M:  5   7.3M:  3    9.1M:  3   12.7M:  3   15M: 3
```

v18 is at 10 frames at its FIRST snapshot, before it has trained, and its own 15M adds
nothing (10 -> 9). So the skill came entirely from v13's weights. **That kills the
"warm just means more training" reading** -- which mattered, because the chain is
v18 <- v13 <- v4 <- v3 at 15M each, i.e. ~60M cumulative, and volume was the obvious
alternative explanation.

v16c reaches its ceiling of 3-5 by 5.5M and sits there for the remaining TEN MILLION
steps. Two thirds of the run spent flat, well short of the threshold.

### v17: NO VERDICT. Its apparent negative is inside the noise.

v17 looked worse than v16c (`Low1` pool 8 -> 2, `Lclimb3_top` 9M-12M window mean
0.49 -> 0.28, arrival survivability 5 -> 0) and I reported that as a clean negative.
It is not supported. Measured run-to-run variance at FIXED seed 42, two runs identical
but for `timesteps` (v16c's 6M and 15M, over their shared first 6M):

```
                        v16c-6M   v16c-15M
Lclimb3_top  4M-6M         0.19       0.35     <- pure nondeterminism
Lclimb3_top  0-6M          0.06       0.16
```

The noise band is the size of the effect. And level3_notes.md already says so: *"a
single 600k control on L3 proves nothing. 1 in 3 seeds collapses"* -- three seeds, no
change at all, princess 70.3% / 0.0% / 69.6% -- ending *"this invalidates every n=1
verdict taken this session"*. I had read that section the same afternoon.

**So `seed_waypoint_skip`'s direction is unknown, not refuted.** Treat any single-run
reach difference below ~0.16 as no signal.

### THE ROOT CAUSE OF THE COLD PLATEAU: every cold run used a recipe known to fail

```
config                      n_steps  target_kl  start
v14/v15/v16a/v16b/v16c/v17    512     ABSENT    COLD
v13, v6                       512     0.07      warm
```

experiments/003-yeti-training.md already documents this exact shape as unable to learn
from scratch, and names the run that proved it
(`yeti_curriculum_v16_coldsteady.yaml`, n_steps 512 + target_kl 0.05, cold, 20M):
*"every snapshot scored princess 0.0 AND reach4 0.0 ... the steady recipe cannot learn
from scratch; the big-step phase-1 was doing essential exploration"*, with the rule
*"never start cold-steady"*.

Its interpretation is annealing on the policy:

* **phase 1** `n_steps: 16`, no `target_kl` = HIGH temperature. Small rollouts give
  large, noisy, sometimes destructive updates (measured KL up to 68) that can stumble
  across a behavioural plateau. Crossing one needs a whole new skill chunk, so small
  greedy steps cannot -- nothing nearby improves the return.
* **phase 2** `n_steps: 512`, `target_kl: 0.05` = LOW temperature, settles into the
  basin without destroying it.

L4's floor-11 wall is exactly a behavioural plateau, and the flat 3-5 arrival
survivability across ten million steps is that rule being demonstrated. **Roughly 50h
of cold compute (v14-v17) went into a configuration already recorded as unable to learn
cold.** The L2/L3 phase-1 configs (`l2_v1/v2/v3`, `l3_v1/v2`) all use n_steps 16;
phase 2 (`v15_phase2`, `l2_v10`) uses n_steps 512 + target_kl 0.05 warm.

Also checked and NOT the explanation: `target_kl` does not damp the oscillation. From
tb, `reach/from_0/ge_10` mean |step-to-step| is 0.007 (v6), 0.006 (v13), 0.005 (v16c),
0.006 (v18) -- the two with `target_kl: 0.07` are if anything slightly noisier. The
notes say the instability and the oscillation are the SAME mechanism, so equal jitter
refutes nothing; it is consistent with v6/v13 having had their exploration phase
earlier in the chain.

### v19: phase 1 for L4 (running)

One lever from v16c, `n_steps: 512 -> 16`, cold, 15M. Throughput measured rather than
assumed: 2335 emu_fps against v16c's 2271, so ~32x more optimiser updates cost nothing
and 15M is still ~7h.

**Read it on arrival survivability, not reach.** The question is only whether
high-temperature exploration finds the floor-11 timing from scratch. Rising toward 8
means phase 2 (`n_steps: 512`, `target_kl: 0.05`, warm from v19) should anneal it,
which would give a CONFIG-ONLY path to a champion instead of an unreproducible
four-run chain. Flat at 3-5 means `n_steps` is not the missing piece either.

### THE STANDING PROBLEM

v18 is the best policy and we cannot reproduce it. Its skill was inherited from a chain
of four runs under four different code states, and it was already complete at v18's
first snapshot. Until a cold recipe acquires arrival timing, every improvement built on
v18 inherits that irreproducibility.

## ROPE 2 IS AN EXPECTED-VALUE PROBLEM, AND THAT IS WHERE +1.0 COMES FROM (2026-09-22)

> **SUPERSEDED 2026-10-07 — read "v30" at the end of this file first.** The diagnosis
> below is that rope 2 is a reward problem. It was not the binding constraint: the agent
> could not SEE the rope (a nearest-neighbour resize drops it), and changing only the
> resize took a cold run from 0 princess touches to 0.69 of from-reset episodes, with
> this section's +1.0 carry bonus unchanged. The pad-reward measurements here are still
> correct; the conclusion drawn from them is not.

DECIDED: pay **+1.0, once per episode, for the rope carry (pose 15)**. NOT YET
IMPLEMENTED. This section is the derivation, so the number can be argued with instead of
re-guessed.

### What each option on the pad actually banks

Measured with `scripts/mo5/yeti/diag/l4_pad_reward.py`, which rebuilds the trainer's own
`RewardContext` per step and calls `reward_fn(ctx)` — the gym's reward is NOT what the
trainer uses, and an earlier attempt through `gym.step` returned 0.00 for everything.
Seeds are the 65 of 100 `Low2_launch` states on safe ground (px >= 188), 90 steps each,
under v22's reward params (`mark_airborne: true`, `pay_on_target_change: true`, gamma 1):

```
  behaviour                                        banked
  hold still on the pad                             0.000
  walk LEFT off the edge                           +0.075
  jump and miss the rope                            0.000
  COMPLETED CROSSING  (NOOP:20,JUMP_LEFT:70)       +2.940   median 2.920, n=6
```

**The ordering was already correct** — a crossing pays 39x the cliff-walk. The failure is
the expected value:

```
  walk off the edge      +0.075  CERTAIN
  attempt the rope       ~1% x 2.940 = +0.029 expected, and 0.000 on every miss
```

0.075 > 0.029, so stepping off the cliff is the better bet and the agent takes it. This is
NOT an exploration failure and not a bad policy; the arithmetic favours the fall. Nothing
in the reward pays anything for a failed attempt, and the whole +2.94 arrives on landing
(the first 6 steps pay 0.000).

Holding still costs nothing, so there is no gradient pushing the agent to leave early.
There is only a gradient pulling it left if it walks.

### Scale, so the magnitude is not picked out of the air

Per-episode `total_reward` from `episodes.csv`:

```
  run          n        min     median    p75      max
  v22 (1M)     3628    -5.96     6.80    57.16    70.39
  v21 (6M)    24416    -7.20    13.72    57.16    70.51
```

### Why NOT +10 (the first proposal)

+10 for the grab is **3.4x the entire value of the crossing** and 1.5x a median episode.
The ordering would survive (grab-and-fail +10, complete +12.94) so it would change the
choice, but it says touching the rope matters more than what touching it achieves, which
is the shape of reward that gets exploited. It also MUST be once-per-episode either way:
the recorded crossings show pose 15 in two bursts (crossing 1: `[4,10,15,10,15,10,5]`), so
a per-frame +10 banks +20..50 for one manoeuvre, up to 70% of a maximum episode.

### Why +1.0

```
  walk off the edge      +0.075
  grab and fail          +1.000    13x the cliff-walk -> the choice flips
  complete the crossing  +3.940    4x a grab          -> finishing still dominates
```

Both margins are wide and both come from the measurements above. The risk that remains,
and it is real: a signal on an event that currently fires ~1% of the time may do nothing
measurable in 6M steps. If +1.0 produces no change, the next step is the magnitude, not a
different mechanism.

### The crossing plan this was priced with

`NOOP:18..22` then `JUMP_LEFT` held, from a px-188 seed. 6/6 recorded clips crossed the
gap, USER-CONFIRMED ON VIDEO. The mp4s were deleted as too heavy to commit; the PNG
filmstrips are kept at `experiments/003-yeti/evidence/l4_rope2_carry/wait{18,20,22}/` and
the pad seeds either side of the lethal pixel at
`experiments/003-yeti/evidence/l4_rope2_seed_compare/px{184,188}/`. Regenerate any clip
with `record.py --from pool:Low2_launch --seed-index 0 --plan "NOOP:20,JUMP_LEFT:60"`.
This matches the scripted sweep already recorded in `l4_pad_reward.py`'s header (40
crossings / 3312 trials, working departure window wait 17..23) and the behavioural half
(v13 pressed LEFT+FIRE in 34/34 episodes but departed at wait 0-3 in 33/34). The input is
right; the timing is wrong; the reward pays for the wrong timing.

### Corrections this session, all of them mine

* **`jump_bruteforce.py --from-pool Low2_launch --land-on Low2` returned 0/20 over 75
  plans and I read it as "the crossing is impossible".** Wrong twice: the tool holds a
  jump input and a rope carry needs a release and a re-jump timed to a moving rope, and
  its top plans used wait 2 and 5, nowhere near the 17..23 window that works. That 0/20
  reproduced the ~700-trial negative already documented above; it is not a feasibility
  result. I also proposed deleting jump edge (12,13) on the strength of it, which would
  have removed a real route.
* **"87% of the `Low2_launch` pool is poisoned" was a correlate, not a cause.** Under the
  policy, 90 steps from every seed: px 184 dies 80% (n=35), px 188 dies 95% (n=44), px 192
  dies 81% (n=21), with identical longest-airborne runs and identical leftmost reach. The
  px-188 seeds are H-AU's 8/8-survivable pixel and they die MORE. Pool composition is not
  the lever, so `admit_requires_grounded` and offline pool cleaning were both aimed at the
  wrong thing.
* **Airborne duration cannot separate the two failures.** Fall-to-spring and
  jump-and-miss look the same in the air, confirmed on video. Longest consecutive airborne
  run does separate an ordinary jump (Spring, median 15 frames) and the legitimate
  trampoline route to Spring (`Lclimb2_top`, 59, 0 deaths) from the rope-2 fall (82-83) —
  but it is useless for ADMISSION because ~90% of `Low2_launch` captures are followed by
  death regardless, so any survival-based gate empties the pool.
* **The policy's pose-15 rate is lower than I claimed.** I measured 1/100 in one pass;
  a 400-episode hunt (`record.py --want pose=15`) found 0/400. Treat 1/100 as a single
  lucky event. This is why the crossing was priced with the scripted plan, not the policy.

### Control arm when this is run

v21 (`ent_coef: 0.01`, warm from v13's `final_model`), NOT a fresh cold run. The +2.94
arrives entirely on landing, so a grab reward changes WHEN credit arrives as well as how
much — and on this level that class of change has a history (`mark_airborne`,
`pay_on_target_change`). One lever, one control.

## Where the diagnostics went (2026-09-16)

`debug/` was gitignored with zero tracked files, so every diagnostic in this project was
one `git clean` from gone along with 836M of stale run data. It is now scratch only —
safe to wipe at any time — and anything that backs a claim lives somewhere durable:

| what | where now |
|---|---|
| diagnostic tools (19) | `scripts/mo5/yeti/diag/` (tracked) |
| figures/videos a note cites (24 files) | `experiments/003-yeti/evidence/` (tracked) |
| run scripts for the A/Bs (5) | `experiments/003-yeti/runs/` (tracked) |
| monitor logs for cited runs (13) | `output/monitor/<job-id>/` |
| everything else | deleted |

Paths in these notes were rewritten to match. **83 scripts were retired**, and because
`debug/` was never in git they are NOT recoverable — sections above still name about 25 of
them (`l4_reward_marks.py`, `l4_pad_wait_sweep.py`, `l4_gate_admitted_proof.py`,
`l4_crossing_trace.py`, and so on). Those names are kept as provenance for how a number
was obtained; the number itself is in these notes, which is the record. Two tools replace
most of them: `diag/record.py` for video and `diag/figure.py` for annotated frames, both
parameterised over source, actor and overlay.

Retired for a stated reason, not by age: superseded by a decision already taken (sprite
reach is the default, so all the box/tolerance tooling is moot), superseded by those two
tools, or a one-shot investigation whose answer is written above.

## Method notes

* `scripts/mo5/yeti/diag/l4_low_route_probe.py` scripts through a raw `BaseEnv` (`go_explore.make_env`),
  i.e. **1 emulator frame per step**. `l4_low1_jump_bruteforce.py` uses the gym
  stack at frame_skip 4, matching training. A move scriptable only at 1-frame
  granularity may still be out of reach for a policy acting every 4 frames; prefer
  the frame_skip-4 number when judging learnability.
* `climb_snapshot_sweep.py` and `climb_reach_probe.py` still hardcode a 5-step
  settle after `load_state`, which commits `d8fe780` / `bcbcc1b` fixed elsewhere
  (measured 4/40 vs 32/40 on L4). Both are L3-only, so no L4 number is affected,
  but their absolute rates are lower bounds. Same for `capture_level3_start.py`
  (`--settle` default 5), `dump_probe_frames.py`, `dump_probe_videos.py`,
  `eval_chained_policies.py`.
* **`eval_from_reset.py` used to infer `level = 2 if start_state else 1`.** L3 and L4
  both boot from a save-state, so **every L3 and L4 eval ever run through this script
  silently used level-2 geometry** — level-2 waypoints, level-2 floors, level-2
  princess. Fruit and princess counts came from RAM so they were unaffected, but
  anything geometric (and all route-depth scoring) was meaningless. There is now an
  explicit `--level`; it defaults to the old inference so existing invocations are
  unchanged, but **pass it always**. `keep_best_sweep.py` gained `--level` and
  forwards it. Any pre-2026-08-24 L3/L4 eval output that mentions waypoints or floors
  should be re-run, not trusted.

## BOTH ROPES RE-MEASURED (2026-09-29), AND WHY THE EARLIER NUMBERS WERE WRONG

Every rope window number in this file before today came out of
`scripts/mo5/yeti/diag/jump_bruteforce.py`, which had two defects that between them
produced a false "the two ropes have different periods" and a false "a long sweep and a
short sweep disagree, so the tool is non-reproducible". Both are fixed. The defects first,
because the numbers are only worth reading once you know they are gone:

1. **The report printed `results[:10]`.** A 131-plan wait sweep with 14 winners had its
   four LOWEST winning waits cut off, because the list is sorted descending. That is the
   whole of the "short sweep says waits 1,2,3,32,33,34 and long sweep says 34,64,65,66,..."
   mystery — the long sweep's answer was the top ten of a set that also contained the low
   windows. It now prints every plan that landed.
2. **`EnvConfig.max_steps` bounded the whole sweep, not the attempt.** `attempt` reloads a
   save-state per plan but never resets the gym, so the env's own step counter accumulated
   across all plans; truncation fired partway through and every later plan ran in a
   truncated env. Now `max_steps=10**6` with the per-attempt bound in the loop.

An order-dependence test after the fixes: wait=1 run three times inside one sweep,
including once after an intervening 190-step plan, gives 1/1 every time. There is no state
leak, which is what the two defects had been masquerading as.

### The two ropes have the SAME period. Rope 2's launch window is WIDER.

Scripted, one seed each, `--approaches 0 --holds 60 --max-steps 400`, waits swept 0..130:

```
             winning waits                                  period   width
rope 1   0-1, 30-32, 62-64, 93-95, 125-127                   31.25     3
   (px 52 seed, Rope1_launch -> Rope1)
rope 2   12-21, 44-52, 75-84, 106-115                        31.3     9-10
   (px 188 seed, Low2_launch -> Low2)
```

The user said from the start that the ropes oscillate at the same rate and that a claim of
"rope 2's period is 62" made no sense. Correct: 62 was the `[:10]` truncation deleting the
middle window. Both ropes run at ~31.3 agent steps.

Rope 2 is the more forgiving of the two to TIME — 10 launch phases in 31 against rope 1's
3. Any argument of the form "rope 2 is harder because the timing is tighter" is dead. The
mechanism, from per-step traces (`.kiro/tmp/rope_trace.py`):

* **rope 1 is a graze.** Airborne 18-19 steps, pose 14 present for only 1-2 frames,
  landing frame = launch + 18, landing px 112 or 116. A launch one step early lands one
  step early somewhere else: the phase error stays in the arc, so the window is ~3.
* **rope 2 is a carry.** Pose 15 for 4-7 frames at y 62 while px runs 188 -> 88. The
  landing frame is **57 for every winning wait 12..21** — 44 airborne steps at wait 12,
  36 at wait 21. The rope absorbs the launch error and deposits the agent at the same
  place at the same absolute frame, so the window is as wide as the catchable interval.

### What actually blocks rope 2 is the DEPARTURE PIXEL, not the phase

> **SUPERSEDED 2026-10-07 — see "v30" at the end of this file.** The px-184 measurements
> below stand (it is lethal from rest and crosses 0 of 34 waits), but it was not what
> blocked the crossing. Once the agent could see the rope, ~95% of episodes that reached
> the pad crossed, with the departure geometry and the pool logic unchanged.

Same sweep, per departure px, waits 0..33 (a full period), one seed each:

```
px 184   0 of 34 plans cross      60 of v23's 100 Low2_launch seeds sit here
px 188   10 of 34 cross (12..21)  18 seeds
px 192    8 of 34 cross (14..21)  22 seeds
```

And px 184 is not merely a bad launch pixel, it is a dead state when loaded at rest.
120 NOOP steps from v23's own pool seeds: px 184 dies at step 82 (pose 11, falling, never
gets past px 176 while floor 13's edge is px 128); px 188 and px 192 survive the full 120.
`min_survival_steps` is 30, so the current gate admits all of them.

Measured with `scripts/mo5/yeti/diag/pool_revalidate.py --only Low2_launch --window 30` on
v23's pool, which is what `admit_requires_grounded` would reject going forward:

```
Low2_launch   100 -> 39   dropped=61   died_in_window=0
   dropped by px: {184: 60, 188: 1}   end_pose: {17: 61}
```

`died_in_window=0` is the point: every one of the 61 is ALIVE at step 30 because the spring
catches it, so the survival-only gate cannot see them. They are caught by the end-pose test
instead — all 61 end in pose 17, the trampoline rise.

This supersedes the run_config comment's "L4 rejects 23 of 415 seeds (6%)", which cited
`debug/l4_survival_gate_blast.py`. That file does not exist in the tree or anywhere in git
history, so its number cannot be reproduced and should not be used.

### Starting from `Low1` instead of the pad is executable

Tested because of the proposal to drop `Low2_launch` and let the agent train from `Low1`
(mechanism: add it to `jump_waypoint_skip`, which L4 already uses for `Spring_launch`,
`Step_launch` and `Low1_launch`). `Low1`'s pool is healthy — 95 of 100 seeds at px 220, on
safe ground — and `Low2_launch` is not in `reward_waypoints`, so removing it does not touch
the reward's mandatory milestones.

From a px-220 `Low1` seed: **26 of 136 plans cross**, run-up 11..13 left steps then wait
16..24 then hold jump-left. The run-up length matters because the leftward walk cycle
stalls: 8 steps reach px 200, 9 reach 196, **12 reach 188**, 13 reaches 184, 14 is already
falling.

NOT yet checked, and it should be before any run: `Low2_launch` is in `route_order`, and
`_wp_eligible` refuses any pool absent from that list, so `route_order` carries
start-eligibility semantics despite its "DISPLAY-ONLY" comment. Removing the waypoint
without removing its `route_order` entry is untested.

### The crossing can succeed and not be credited

Landings from the three policy crossings were px 88, 92 and 120; every scripted crossing
from px 188 lands at px 88. `Low2`'s credit window is px 104..152. So the most reproducible
version of this manoeuvre — the one a script finds and the one the policy performed twice
of three times — deposits the agent 16 px short of being credited for it, and only earns
`Low2` after walking right. That is a reward-side problem, independent of pools and
independent of the launch pixel, and it has not been acted on.

## v25's "REGRESSION": RE-DERIVED 2026-09-29 — CLOSED 2026-10-01, IT WAS NOISE

(Heading amended. It read "AND STILL OPEN" for one day. The three replicates in the
next section settle it: the gap is below what the design could resolve. Read both.)

Earlier in this session the v24-vs-v25 gap was declared to be inside the noise. **That
verdict is retracted.** It compared a difference of RUN MEANS against the within-run
wander of 1.2M-step block means (sd 0.106), which is the wrong reference distribution. The
right one is the spread of run means at FIXED config, and that has never been measured on
L4.

What the runs actually are. v24, v25, v26 and v27 all resume from the same parent
(`yeti_curriculum_l4_v23_ropebonus_warm_v13_6m/final_model.zip`), all with
`resume_pools: output/mo5/yeti/pools/empty.pkl`, all `seed: 42`. Identical starting
conditions:

```
run   mean Low2_launch rate   carries rewards.SURFACE_POSES = {0..8}?
v24          0.349             no   (control)
v25          0.195             yes  + LEVEL4.edge_inset=4 (since reverted)
v26          0.209             yes  alone
v27          0.223             yes  + the reach-universe fix
```

So this was never a v25 regression. It is v24 against three runs that agree with each other
to a spread of 0.014 against a gap of 0.129, and the one change present in all three and
absent from v24 is `rewards.SURFACE_POSES` gaining the leftward-walk poses 6 and 7.

**The "very nearly inert" defence is measured on the wrong quantity.** `850f7ae`'s message,
and v25/v26/v27's config headers, all argue the change cannot matter because 12 replayed
episodes give identical totals and "at gamma 1.0 the PBRS sum telescopes". That gamma 1.0
is the REWARD's, under `reward.params`. PPO trains with `gamma: 0.99, gae_lambda: 0.95`,
and telescoping does not preserve a discounted return. Re-measured on the same 12 episodes
(`.kiro/tmp/price_surface_poses.py`, extended to report per-step and discounted figures):

```
mean UNDISCOUNTED   old 60.717   new 60.717   delta -0.000
mean DISCOUNTED     old 11.142   new 11.159   delta +0.017   (+0.2%)
per-step rewards differ on 59 of 6475 steps, max per-step |delta| 0.400
```

So the change is NOT inert: per-step rewards differ, and the objective PPO optimises moves.
But +0.2% is small and it moves in the POSITIVE direction, so this does not explain a 38%
relative drop either. The suspect is un-ruled-out, not convicted.

**Why it cannot be settled with what we have.** v23 warm-started from v13, not from itself,
so v24 is the only run with this parent that lacks the change — group "without" is n=1.
"v24 was the lucky draw" fits the data exactly as well as "SURFACE_POSES costs 0.13". The
only thing that separates them is replicates at fixed config: the same yaml at two or three
different seeds, both with and without the change. That has not been run, and until it is,
no attribution of the 0.129 to any lever is supportable.

## THE RUN-TO-RUN SPREAD, MEASURED AT LAST (2026-10-01) — AND v25 CLOSED

Three replicates of ONE config: v24's yaml with only `training.output` changed, same
parent, same `empty.pkl` pools, same `seed: 42`, same commit 723993e, run sequentially
on one machine. Verified identical by parsing all four configs and comparing the
dataclasses. ~2.83h each, 60 snapshots each, referee at n=30, all six phases exit 0.

```
run    config                mean_rung   0-3M   3-6M   headline   princess
v28a   REPLICATE                 3.998   4.79   3.21      0.149      0.000
v28b   REPLICATE                 5.009   4.20   5.81      0.219      0.000
v28c   REPLICATE                 4.782   3.98   5.59      0.239      0.000
                        sd 0.530, range 1.011        sd 0.047, range 0.090

v24    control, old gate         5.651   5.75   5.54      0.349      0.000
v25    SP + edge_inset=4         4.543   3.78   5.31      0.195      0.000
v26    SP alone                  4.378   5.44   3.32      0.209      0.000
v27    SP + reach-universe       4.322   4.15   4.49      0.223      0.000
```

**Three identical runs span 1.011 rungs.** That is the number this project never had and
every comparison in it depended on. Consequences, all arithmetic from the sd above:

```
                                        mean_rung    headline
two SINGLE runs can resolve nothing below   1.470       0.131
3 runs/arm detects (80% power)              1.213       0.108
```

The gap that five 6M runs were spent on is **1.24 rungs / 0.129 headline**. Both sit at
or below the resolution of the design that produced them. v25, v26 and v27 land within
0.52 sd of the replicate mean -- indistinguishable from draws of the same config. So
there was never a v25 regression to find, and `rewards.SURFACE_POSES` is not implicated.

**The half-run split was also noise.** v26's -2.12 and v25's +1.53 were read in this
file as evidence that no common cause could explain all three arms. The replicates swing
-1.58, +1.61, +1.61 on the same config. A run's own halves carry no signal; do not
compare them.

**What is NOT settled.** v24 is the highest of all seven, +1.99 sd on `mean_rung` and
+3.09 sd on headline. With n=3 the prediction interval for one new run is too wide to
exclude it, so "v24 was a high draw" and "v24's gate is genuinely better" both survive.
Deciding it needs 3 runs with `reward.params.surface_pose_ids: [0,1,2,3,4,5,8]`, ~9h,
and that is not worth it ahead of rope 2: princess is 0.000 in all seven runs.

**A retraction I got wrong in both directions.** The steering rule "with one run per side
ignore any difference under ~0.13" was derived from within-run block wander, which is the
wrong reference distribution. I retracted the rule on that basis. The measured resolution
for two single runs is **0.131**. The number was right and the reasoning was not; the
correct response to an unsupported derivation is to go and measure it, not to discard the
conclusion. It now rests on these three runs.

### Operating rule for every future L4 comparison

Minimum 3 runs per arm. A single run per arm answers nothing about any lever whose effect
is under ~1.2 rungs. Where an arm's readout is a quantity pinned at exactly 0 -- `Low2`
reach, princess -- a single run IS informative, because the null is "never happened in
420 evals across seven runs"; use that asymmetry rather than paying for replicates to
detect a mean shift nobody needs.

## v30 (2026-10-07): ROPE 2 BROKEN — THE AGENT COULD NOT SEE THE ROPE

**Result.** One lever, `env.resize_mode: max`, cold, 15M, seed 42, commit `f7d2d43`, clean
tree. The champion (13.25M) reaches the princess in **208 of 300** stochastic from-reset
episodes, **0.69** (95% Wilson interval 0.64–0.74), measured with
`eval_from_reset.py --episodes 300 --stochastic --resize-mode max`. Before this, princess
was 0.000 in every eval of every L4 run, v23 to v29.

```
from-reset reach, v30 champion, n=300
Lfruit_top 0.99   Rope1 0.83   Lclimb3_top 0.74   Low1 0.73   Lprincess_top 0.69
princess 0.69     (all 208 via rope 2; 0 touched Lhi_up_top or any Hi point)
```

About 95% of episodes that reach the rope-2 pad go on to touch the princess (0.69 of
0.73). Rope 2 is no longer the wall; the remaining losses are upstream, mostly rope 1
and the climb after it.

The referee's own number for this champion was 0.80 on 30 episodes. The 300-episode
figure is ~0.1 lower, the same selection inflation measured on v23 (0.967 -> 0.85).

### How the run went

Referee, 60 snapshots, every one scored with `--resize-mode max`:

```
frontier counts      Lprincess_top 37, Step 8, Lfruit_bot 4, Lhi_down_bot 4, Low1 3, other 4
princess, mean over all 60 evals (unselected)   0.22    max 0.80
frontier at Low1 or deeper                      0/4 evals per 1M block until 3M,
                                                3-4 of 4 in every block after
pools at the end                                Low1, Low2, Lhi_down_bot, Lprincess_top
                                                all 100 seeds
```

Training episodes: 20,075 from reset, of which 4,333 reached `Low2` (21.6%) and 3,320
touched the princess, the first at step 4.35M. Every one of the 3,320 passed through
`Low2`. For comparison v29, warm on the old resize, reached `Low2` in 1 of 7,442
from-reset episodes. Genuine `Low2` crossings (episodes not seeded at `Low2`): 14,166,
against 3–7 per 6M run before.

### Why: the observation dropped the rope

Rope 2 is a one-pixel-wide pure-red line (RGB 255,0,0, measured in the raw frame).
`preprocessing.py` does two things to it:

* grayscale by luminance, `0.299R + 0.587G + 0.114B`, turns pure red into **76**, the
  darkest colour on L4's screen (green platforms 149, yellow ladders 225, white 255;
  those exact values are what the policy's input contains in the gap);
* the 320x200 -> 84x84 resize was nearest-neighbour, keeping one source pixel per output
  pixel (about 1 column in 4, 1 row in 2.4). A one-pixel line mostly falls between the
  kept pixels. Measured over 20 frames across the route: the rope keeps **0–4** pixels,
  and in some frames none at all.

And position alone does not say "jump now": wait 0 misses and wait 16 crosses, with a
near-vertical rope in both. The cue is the direction of swing, which the policy could only
read from how a few dim, intermittent dots moved across its 4 stacked frames. Block-max
keeps the brightest pixel of each block instead, and keeps **12–15** rope pixels in every
frame. The rope stays dim (76); only the resize changed.

Figures, in `experiments/003-yeti/evidence/l4_v30_rope_visibility/`:
`rope2_agent_view.png` (raw frame vs the policy's input across one swing),
`rope2_resize_closeup.png` (raw / nearest / block-max, same crop),
`rope2_resize_fullscreen.png` (whole screen, and the 14% of input pixels that change).
The whole-screen figure shows rope 1 is invisible in the old view too, consistent with
rope 1 being crossed because the ladder delivers the agent in phase, not by sight.

Side finding from the frames: the rope-2 gap holds ONE rope. The two catch positions
(px 168 and px 132, both y 62) are that rope at two points of its swing, which settles the
"two ropes or one rope at two phases" question left open above.

### Why it had to be a cold run

Block-max changes ~14% of every input pixel, not just the rope: every bright edge grows
by up to a pixel, ladders lose their rungs, the HUD text merges. Measured, the v29
champion from reset, 60 episodes each way: mean rung **8.17** under nearest, **0.02**
under block-max (`Lfruit_top` 1.00 -> 0.00). No existing L4 policy survives the change.
A policy must be trained AND evaluated under the same `resize_mode`; nothing errors on a
mismatch, it just reads as a total collapse.

### Attribution, and the one gap in it

Three earlier cold 15M L4 runs, all under the nearest resize, never touched the princess:

```
                    resize    episodes reaching Low2   princess touched
v16c cold 15M       nearest           39                      0
v17  cold 15M       nearest           29                      0
v19  cold 15M       nearest            0                      0
v30  cold 15M       max           14,166 (genuine)        11,783
```

So a cold start alone did not do it. Those three ran on older commits, though, so the
one combination not yet run is cold + nearest on today's code. That control (~9h) is the
only thing between "the resize did it" and certainty.

### What this overturns, and what it doesn't

Overturned: rope 2 as a reward problem (the 2026-09-22 expected-value section) and as a
departure-pixel problem (2026-09-29). Both measured real things — the pad's shaping does
pay +0.12 per step toward the lethal px 184, made of three unreached targets each closing
4 px times `scale` 0.01 — but neither was what stopped the crossing. v30 crossed with the
reward, the pad geometry and the pool logic all unchanged.

Measured on the way there, 2026-10-01 to 10-06, with the old view, and still true of it:
77% of `Low1`-start episodes ended in the spring pit, in episodes a median 97 steps long
whose fatal step is around step 13; the v29 champion's argmax policy touched the rope in
0 of 105 phase-shifted attempts, its stochastic policy in 3 of 105 (4 of 88 in a second
sample), and none of those grabs converted; the crossing needs jump-left
held for at least 26 steps; and both catch positions are genuine decision points (from the
second catch, 2 held steps land). All of that describes an agent acting on a rope it
could not see.

### Evaluator defect this exposed — FIXED 2026-10-07

`eval_from_reset` read `Low2` at **0.01** for this champion while training saw it at
21.6%. The rollout (`yeti_rollout.py`) claimed to mirror the trainer's detection but used
the box test with a grounded-only pose gate, while the trainer has used sprite overlap
with a fail-open pose gate since `de21939`. The crossing FLIES over `Low2`'s anchor
(px 128), lands at px 88 and walks left, so it was never grounded inside the box. The
princess count is a RAM flag and was never affected.

Fixed: the rollout now takes `reach_mode` (default `"sprite"`, the trainer's default),
`eval_from_reset` and `keep_best_sweep` take `--reach-mode`, `run_train_and_score.sh`
reads it from `curriculum.waypoint_reach_mode`, and the eval JSON records both
`reach_mode` and `resize_mode`. `--reach-mode box` reproduces every eval made before this
date exactly (`test_box_mode_is_identical_to_the_old_rollout_rule`), and
`test_eval_default_matches_the_trainer_default` fails if the two defaults ever drift again.

Re-measured, v30 champion, 30 episodes, sprite: princess 24/30, `Low2` **0.80** (was
0.01), and `Low2` recorded in 24 of the 24 princess episodes.

**What the fix exposes, not fixed here.** Those 24 princess episodes now read `max_rung`
**13/13**, not 12. Sprite overlap counts the flight over BOTH members of the OR-group
`[J12_13_b, Lhi_down_bot]` (anchors px 128 and px 104), and the rollout's `max_rung`
counts mandatory IDS, not satisfied groups — the wart already recorded beside
`reward_waypoints` in `yeti_map.py`. The trainer counts groups since `ded0032`; the
rollout does not. So eval mean rung now reads one rung HIGH on a crossing episode,
relative to group counting. Mean rung from before and after this fix is not comparable
on any run that crosses rope 2.
