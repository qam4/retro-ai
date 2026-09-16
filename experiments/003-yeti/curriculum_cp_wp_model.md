# Curriculum model: Checkpoints (CP) and Waypoints (WP) — canonical reference

Purpose: we have re-derived this same model at least three times across
sessions and keep stumbling on the same confusion. This is the single source of
truth. If something here is wrong or changes, edit THIS file — don't re-argue it
in chat.

## TL;DR

There is no single "milestone" type, and we deliberately did NOT merge CP and WP
into one class (that refactor is churn-for-churn on solved L1/L2 code — see
"Decisions"). Instead: **CP and WP are two kinds of route point that share some
roles and differ on others.** The confusion comes from five *independent* roles
a route point can play; CP and WP each carry a different subset.

## The five independent roles

A "point on the route" can, independently, be any combination of:

1. **Graph node** — a vertex in the nav-graph used for path-distance shaping
   (`yeti_map` nodes + edges; distance = walk `|dx|` + ladder climb `|dy|`).
2. **Reward target** — the PBRS potential sums path-distance *to* it. Can be
   mandatory or optional; grouped into OR-groups (reach any member).
3. **Seed source** — we capture save-states there (survival-gated) and start
   episodes from them (`StartPool`).
4. **Progress-tracked** — there is a reach/success EMA telling us how often the
   agent gets there (`reset_reach_ema`, `seg_success`, the `success=` line).
5. **Trigger** — how "reached" is detected: a **game event** (fruit collected,
   princess flag) vs a **position** (grounded within a tol-box of x,y) vs a
   **pose** (escalator ride = pose 13).

Two properties are orthogonal to the roles and describe the point itself:
- **Mandatory vs optional** — must the route pass through it?
- **Ordered vs unordered** — does sequence matter? (We do NOT impose ordering;
  the reward sums over all not-yet-reached targets, like fruits always have.)

## What CP and WP are today (as implemented)

| Role / property        | Checkpoint (CP)                          | Waypoint (WP)                                   |
|------------------------|------------------------------------------|-------------------------------------------------|
| Trigger (role 5)       | fruit collected (event); princess = terminal | position within tol-box (now incl. pose 13 = escalator ride) |
| Graph node (role 1)    | yes (fruit / princess node)              | yes (ladder top+bot; both ends always exist)    |
| Reward target (role 2) | yes — the potential sums dist to remaining fruits | ONLY if listed in `reward_waypoints` (mandatory); OR-groups; unreachable dropped |
| Seed source (role 3)   | yes — `checkpoints[n]` StartPool          | yes if listed in `waypoint_ends` — `waypoints[id]` StartPool |
| Progress-tracked (role 4) | YES — `reset_reach_ema`, `seg_success`, frontier gating | **NO** — only pool size + capture/reject counts; `goal_score` is keyed to the ULTIMATE fruit goal so it reads 0.00 and says nothing about intermediate progress |
| Mandatory / ordered    | mandatory, unordered (collect all fruits, any order) | per-WP: some optional, some mandatory; unordered |
| Indexing               | integer `n` = fruits collected (0..fruits_total, +1 princess) | string id (e.g. `Lesc_top`, `Ldown_bot`)        |

## What is UNIFIED vs SEPARATE

- **Unified (shared code):**
  - Seeding: both use `StartPool` (same reset-origin retention + goal-score
    weighting; WP group is count-invariant in `pick_start`).
  - Survival gate: both admit seeds via the SAME `_admit_by_play` (reached-next
    OR survived >= `min_survival_steps`), scored at episode end. (Waypoint
    capture was made survival-gated to match CP — earlier it seeded doomed
    dying-fall states, e.g. the `Lesc_bot` clips.)
  - Reward: mandatory WPs are reward targets via `reward_waypoints`, summed in
    the SAME potential as remaining fruits (unordered; unreachable dropped).
- **Separate (deliberately):**
  - CP carries fruit-ordered curriculum gating (frontier, reach-gate) that is
    genuinely fruit-specific. We did not generalize it to arbitrary points.
  - Role 4 (progress tracking) exists ONLY for CP. **This is the one real
    asymmetry left** (see Open items).

## Settled design decisions (do not relitigate)

1. **No `milestone` supertype / no CP↔WP merge.** StartPool already removed the
   seeding duplication; a full merge would generalize CP's gating machinery onto
   solved L1/L2 code for zero functional gain and real regression risk. If, after
   several more levels, the CP/WP split becomes a maintenance drag, THAT is the
   trigger to revisit — test-first.
2. **Mandatory WPs are reward targets** (`reward_waypoints`), summed like fruits,
   unordered, OR-groups supported, unreachable (>= 1e8 sentinel) dropped. DONE.
3. **Reward distance is graph path-distance, not euclidean** — euclidean would
   reward jumping into gaps toward a target (kills L1/L2). Where the graph has no
   path (true jump gaps: escalator pre-`Lesc`, the A1-A5 ascent), shaping is
   **0 across the gap** (no euclidean fallback) — the gap is learned by
   exploration + reach reward, exactly like L2's timing gaps. (`Lesc` was later
   added as a ladder edge so the escalator IS graph-connected; segment shaping
   then credits the descent.)
4. **Graph needs BOTH ladder endpoints as nodes** for connectivity, even when
   only one end is used as a waypoint (`waypoint_ends` picks the seed/target end;
   the nodes exist regardless).
5. **Ordering is not imposed.** Sum over all not-yet-reached targets (fruits +
   mandatory WPs). L1/L2 are byte-identical because they have no
   `reward_waypoints`.

## Detection rules (how "reached" fires)

- CP: `curr_fruits < prev_fruits` (fruit collected); princess flag rising edge.
- WP: agent grounded (`pose in SEED_POSES` = surface poses {0-5,8} ∪ {13} the
  escalator-ride pose) AND within `tol` of the WP's (x_ram, y). Same tol-box used
  for capture and for marking a `reward_waypoints` group done.

## Two progress signals: pool size vs reach-EMA (they measure different things)

We actually have TWO complementary "how far did we get" signals, and conflating
them caused this session's confusion:

- **Pool size per WP (ALREADY EXISTS — `wp[..]: id=N`).** Measures the
  exploration FRONTIER / *local* reachability: each WP pool is filled by the
  reverse curriculum from the seeds just BELOW it (an episode seeded at `Ldown`
  that reaches `Lsc1` fills `Lsc1`'s pool). An unsaturated/empty pool = the agent
  can't reach that WP even from just below = a genuinely hard segment. v5 example:
  `Lsc4_top=7` (vs 100 for everything below) correctly flags SN3 as the frontier.
  NOTE: capture COUNT (`wp_near ...xN`) is the uncapped lifetime magnitude; pool
  size caps at `max_states_per_checkpoint` (100), which is enough for block
  detection ("did it ever fill").
- **Reach-EMA from a FIXED base origin (role 4 — MISSING).** Measures end-to-end
  CHAINING: can the policy do it start-to-finish. This is what catches the case
  where a pool is FULL but the chained run fails — v5: `Lsc1_top` pool = 100 yet
  from-goat reach = 1% (the pool was filled by reverse-curriculum seeds just
  below BR, not by chained goat runs). Pool size HIDES chaining failures; the
  reach-EMA exposes them.

Because pool size already covers the local/frontier signal, the reach-EMA must
measure the COMPLEMENTARY thing: reach from a fixed base origin (NOT "at/below
the WP", which would just duplicate pool size).

- Fix (additive, no refactor): add `wp_reach_ema[id]`, updated per episode from
  the set of WPs actually reached, over episodes started from a FIXED base
  origin, printed in the summary, e.g.
  `wp_reach: Lesc_top=0.81 Ldown_bot=0.78 Lsc1_top=0.01 ...`.
- Origin decision: **reset-origin** (exact parity with CP `reset_reach_ema`,
  general across levels) — caveat: `reset_fraction=0` makes the sample thin, so
  either guarantee a small reset fraction for measurement, or use goat-origin for
  L3 specifically. Do NOT condition on "at/below the WP" (that duplicates pool
  size and re-introduces the self-seed inflation).

## The model, restated: TARGETS, and two ways to index seeds

This supersedes the CP-vs-WP framing above for everything except detection.
Decision #1 ("no merge") was overturned by evidence: the same behaviour missing
from one type twice produced real bugs (the retreat pull, and paying for a fatal
arrival). See `training/targets.py`.

**A TARGET is anything on the route that can be reached and pays credit once** —
a fruit, a waypoint milestone, the princess. The five roles are data on it; the
only type-specific part is the TRIGGER, and the trigger is what decides whether
"reached" survives a save-state:

* ``event`` (fruit) / ``flag`` (princess) — state lives in emulator RAM, so it
  survives a save-state for free.
* ``position`` (waypoint) — computed from x/y, so it does NOT survive; it must be
  captured with the seed and restored on load.

**Seeds are stored once and indexed two ways**, because two different questions
get asked of them:

* ``at[<target>]`` — "start me AT this spot." Keyed by target identity.
* ``done[N]`` — "start me where N mandatory targets are done." Keyed by COUNT.

### The progress ladder (``done[N]``)
A rung is "one more mandatory target done". This replaces "one rung per fruit",
which starved levels whose fruits are few and deep:

* L1: 4 rungs, L2: 2 — UNCHANGED, because their mandatory targets are exactly
  the fruits. Their champions are therefore untouched by construction, which is
  what makes this reframing safe (asserted in `test_progress_rungs.py`).
* L3: **1 rung -> 13**. With a single fruit at the summit the old ladder had one
  step, so `cp=[0, 100]` / `success=[0->1]` carried no information through v5-v13
  and "reached the next checkpoint" (used by seed admission) could never fire.

Consequences that fall out rather than needing separate patches: seed admission
credits progress again on L3; `reset_reach` / `seg_success` gain resolution; and
pools stay fat (keyed by count, not identity, so no fragmentation).

Working hypothesis for WHY this matters, from the L2 history: waypoints did not
solve L2 by escaping the reach gate (waypoint pools are filled by
capture-on-reach, so they only ever hold states the agent HAS touched — they were
never unreachable, just rarely reached). They solved it by giving the curriculum
many intermediate segments where it previously had two very deep ones. L3 had the
same poverty; the ladder is how it gets the same granularity.

### Known asymmetry, deliberately left alone
``at[...]`` pools ignore the reach gate; ``done[N]`` pools honour it. The gate
("don't start here until the agent reaches it from reset") is L1-era and fits a
monotone fruit count; waypoints arrived later with a deliberately different rule.
The unexamined middle option is a LOCAL gate: "is this reachable from the pool
below it?" Not changed here — it is a behavioural experiment, not a refactor, and
the route table now provides the measurement (`prog` says whether a pool leads
anywhere; `reach` says whether it is reachable from reset).

## Route view: what goes in the LOG vs a FILE vs a TOOL

Adopted after the log grew three parallel walls (`wp[..]`, `wp_reach:`,
`wp_near:`) that each rendered a DIFFERENT role of the SAME points in a
DIFFERENT sort order, and after we kept answering pairwise questions with
throwaway emulator probes. The rule:

**The start x reached data is a MATRIX (~N^2). It never goes in a log line.**
For L3, N ~ 20 route points = ~400 cells. Every attempt to squeeze a slice of it
into stdout produced another wall. So split by where data lives:

1. **FILE — the matrix (raw, complete).** `episodes.csv` carries two columns:
   - `start_key` — the TRUE start ("0" = real game reset, else a CP level or a
     waypoint id). `start_level` CANNOT serve this: it is derived from
     fruits-remaining, so a WP-seeded episode reports 0, identical to a reset.
     (That bug silently corrupted from-reset analysis, and the TB `reach/from_0`
     tags with it.)
   - `reached_points` — ';'-joined route points reached that episode.
   Together these make ANY pair/window derivable offline, from the real training
   distribution — no new log field, no emulator probe.
2. **LOG — two fixed-size 1-D projections** (they do not grow as points are
   added), rendered as a route-ordered table (`CheckpointManager.route_table`)
   at a LOW cadence, plus a compact per-interval line:
   - `reach` — reset-origin reach EMA -> does the chain COMPOSE end-to-end?
   - `prog` — ORDER-FREE progress EMA -> is this hand-off HEALTHY?
   ...alongside `pool` (frontier), `near` (got close but never landed) and
   `cap/rej` (survival-gate filtering). The per-interval line carries only the
   scalar `route[N]: k/N reached>=0.5 from reset`.
3. **TOOL — `scripts/mo5/yeti/route_report.py`.** Renders the full matrix or any
   slice from `episodes.csv` (no emulator, no run). This replaces the ad-hoc
   link probes.

### Why `prog` is order-free (and there is no "next point")
An earlier draft used `link = P(reach the NEXT route point)`. That is
ill-defined: waypoints are UNORDERED (decision #5), levels BRANCH (L2 has two
ladders per floor), and L2's route goes DOWN to the fruits before going UP, so
"next" cannot even be derived from distance-to-goal. Instead:

> **prog = P(an episode started here reaches at least one NEW route point it did
> not start from or inherit from its seed)**

This is `seg_success` generalised to every route point: branch-safe, needs no
ordering, and still flags every real pathology (SN3 seeded -> reaches nothing
new -> prog ~0). Pairwise numbers stay a DIAGNOSTIC (the tool), not a metric.

`LevelMap.route_order` exists ONLY to sort the table rows top-to-bottom in
travel order. It carries no semantics, nothing gates on it, and levels may omit
it (rendering falls back to a stable order so no point is hidden).

## READING THE LOG: what each number is, and what it is NOT

Written down because we have re-derived this after every run and drawn a wrong
conclusion each time. **None of the training-log numbers is a capability
measure.** Quote them only for what the third column says.

| field | exact formula | means | does NOT mean |
|---|---|---|---|
| `reach` | `wp_reach_ema[w] = .98*prev + .02*[w in reached_wps]`, updated ONLY when `start_level == 0` (real reset) | P(a from-reset episode stood, GROUNDED, inside `w`'s tolerance box) | that any objective was achieved; the box is a POSITION |
| `prog` | `progress_ema[start] = .98*prev + .02*[progressed]`, `progressed = (reached_wps - inherited - {start_wp}) != {}` or `reached_level > start_level` | P(an episode started HERE touched at least one NEW route point) | that it got closer to the goal, or reached the NEXT point |
| `pool` | `len(waypoints[w])` | retained seed states | reachability from reset (the reverse curriculum fills pools from other seeds) |
| `cap/rej` | lifetime capture / survival-gate reject counts | how hard the admission gate is working here | anything about the policy's from-reset skill |
| `near` | min grounded Chebyshev distance ever seen | got close but never inside the box | a near-miss rate |
| `reset_reach[n]` | P(from-reset episode reached RUNG n) | rung = COUNT of mandatory targets banked | fruits collected — the fruit is ONE of the mandatory targets (12 on L4) |
| `gscore[n]` | EMA of `_episode_score` | sampling weight input (`1 - gscore`) | success rate. Absolute-depth by default, so deep seeds score high for free |

### Two pool-indexing schemes (they coexist; do not conflate)

| pool | key | filled when |
|---|---|---|
| `checkpoints[n]` (rungs) | `n = rung_of(reached) = len(reached & mandatory_ids)` — COUNT of mandatory targets banked, NOT fruits | **only on a FRUIT PICKUP** (`if fruits < prev_fruits`) |
| `waypoints[wp_id]` | the waypoint NAME | any grounded frame inside the waypoint's tolerance box |

The rung INDEX was generalised to mandatory targets; the rung FILL TRIGGER was
not. Scope of that mismatch, precisely:

* **General** (all mandatory targets, fruits *and* waypoint milestones):
  `rung_of`, `_current_rung`, `reached_level`, `_episode_score`, `goal_score` /
  `gscore`, and the reach EMAs. `_reached_targets` unions inherited waypoints,
  waypoints reached this episode, and collected fruits, then `rung_of` filters
  by `mandatory_ids` (from `build_targets`, princess excluded).
* **Fruit-specific**: ONLY the rung-pool fill trigger,
  `if fruits < self._prev_fruits`.

So scoring is fine; only STORAGE is affected. `gscore` moves whenever a
milestone is banked, but the deep rung pools it nominally indexes can never
receive a state on a one-fruit level because nothing triggers a save there. (The
principled fix — fire a rung save whenever `_current_rung()` increases — changes
what fills `checkpoints[]` on every level, so it needs the usual sweep.)

L4: taking the fruit banks `Lfruit_top`, `J2_3_b` and `F1` = 3 mandatory
targets, so every rung save lands at index 3:

```
cp=[0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # v4, 15M
```

One usable pool, twelve permanently empty — and `reset_reach` still prints an
entry per rung, most of which describe rungs no state can occupy.

### Rungs count an UNORDERED SET, so they credit unfinishable routes

`rung_of = len(reached & mandatory_ids)`. The fruit is ONE element, weighted the
same as a ladder top, and curriculum reach-marking is ungated (only the REWARD
sum honours `waypoints_after_fruit`). So an agent that skips the fruit and climbs
the ascent banks milestone after milestone and reads as most-of-the-way-done
while being incapable of completing the level.

Measured, v4 final `reset_reach` against the 300-episode eval:

```
rung:     0     1     2     3     4     5     6     7     8     9    10   11-13
reach: 1.00  0.96  0.93  0.93  0.91  0.90  0.89  0.88  0.87  0.83  0.06  0.00

fruit collected from reset: 12.3%
```

83% bank 9 of 13 mandatory targets; 12.3% take the fruit. So ~71% of reset
episodes reach rung 9 WITHOUT the fruit. L4 v1 is the extreme case on record:
`Lascent_top` 99%, `Lfruit_top` 0.1%, **0 fruits in 3000 reset-origin episodes**
— an agent that would score `gscore` ~0.85 while being unable to finish.

This is not only a reading hazard: allocation weights starts by
`1 - goal_score`, so a start that banks milestones on an unfinishable route looks
nearly solved and receives the SMALLEST sampling weight.

Fix direction (same notion as the reward's phase gate): give targets
PREREQUISITES and count a rung only when its prerequisites are satisfied, so
ascent milestones do not count until the fruit is banked. That replaces the
global `fruits_left` boolean and generalises to multi-fruit levels.

### The two traps we keep falling into

**1. `prog` saturates.** `progressed` is "touched anything new", which is free
from any start that is not the very last one. v4's final table, whole early
route:

```
Lfruit_top 1.00  Fr1 1.00  Fr2 1.00  Lfruit_bot 0.99  Lascent_top 0.98
Lclimb1_top 0.98  Rope1 0.98  Lclimb2_top 0.97  Spring 1.00  Step_launch 0.93
```

Pinned at 1.00. It is informative ONLY at the frontier (v4: `Step` 0.58,
`Lclimb3_top` 0.27). This is the same defect already noted for
`seg_success_ema` — see "Why `prog` is order-free": order-freeness fixed
branch-safety, not saturation.

**2. `reach` is a position, not an objective.** A tolerance box near a target is
not the target — `Lfruit_top` is a ladder top on the WAY to the fruit, so reading
it as "collected the fruit" is wrong (that misreading produced a claimed
"fruit from reset 0.61 -> 0.98" for v4).

But do NOT over-correct: the reach EMAs are otherwise accurate. When the route
table and an eval disagree, suspect the EVAL first. v4:

```
route table   Fr2 reach 0.94
eval_from_reset (settle=5)   fruit 12.3%      <- WRONG, harness bug
eval_from_reset (settle=1)   fruit 75-100%    <- agrees with the table
```

`rollout_episode` defaulted to FIVE settle NOOPs after `load_state` while the
training env takes ONE (it was fixed there; the fix never propagated). Five
burns ~20 emulator frames while the level runs on, which on L4's timed opening
costs almost everything: 4/40 fruit at settle=5 vs 32/40 at settle=1, same
weights. Every from-reset number ever produced through that harness — L1 and L3
included — is therefore a LOWER BOUND and needs re-measuring before it is
compared against anything.

### The only authoritative numbers

From-reset capability comes from ONE place:

```
1. train
2. scripts/mo5/yeti/keep_best_sweep.py   --snapshots-dir <run>/snapshots
3. scripts/mo5/yeti/eval_from_reset.py   --model <run>/best/best_model.zip \
       --episodes 300 --stochastic
```

`final_model.zip` is NOT the run: it is usually in a dip. v4's final model got
0 fruits in 10 from-reset episodes while its best snapshot got 12.3% — and every
policy probe run against a final model is therefore unattributable. A single
snapshot is n=1; the sweep evaluates all of them, so use its spread.

## DETECTION vs frame_skip: why WP placement must be "where the agent RESTS"

Measured on L3 (`scripts/mo5/yeti/diag/l3_lesc_boarding.py`), and the reason `Lesc_top` reads
reach 0.4% while the point just PAST it (`Ldown_bot`) reads 78% from reset — the
agent obviously crosses it, we just don't see it.

**The mechanics.** Detection is `abs(x - wx) <= tol and abs(y - wy) <= tol` on
grounded/ride poses, evaluated ONCE PER GYM STEP — and a step advances
`frame_skip = 4` emulator frames. So the tol-box is sampled in 4-frame jumps:

- **Unit asymmetry (easy to miss):** waypoints are `(x_ram, y_px)` and the SAME
  `tol` is applied to both, so with `tol=2` the window is **±8 px horizontal but
  only ±2 px vertical** (x is in 4px RAM units).
- Horizontal walking covers ~4 px/step -> comfortably inside the ±8 px window.
- Vertical motion is ~4 px/step riding (measured: y deltas of exactly 4) and
  faster falling -> LARGER than the ±2 px window, so a vertical pass can step
  clean over the box and never register.

**Why the system is nonetheless sound:** every WP is placed where the agent
COMES TO REST (the arrival end of a ladder, a jump landing). Standing still, y is
constant for many steps, so the box is always sampled no matter the frame skip.
The capture counts show it: platform/ladder points fire thousands of times
(`Lsc1_top` 1600+, `A1` 300+); only `Lesc_top` starves (3 captures).

**The exposure to remember:** a waypoint the agent passes VERTICALLY WITHOUT
STOPPING will silently under-detect — under-reporting `reach` AND under-capturing
seeds (so it can never be a usable seed source). `Lesc_top` is the current
example, and it has a second, independent problem: its position is not on the
path every time — measured BOARDING y was 94 (6x), 98 (9x), 102 (2x), 106 (1x),
so only 6/18 crossings even pass within ±2 px of its y=94.

**Rules that follow (check these when adding waypoints to a new level):**
1. Place WPs where the agent RESTS, not mid-trajectory. This is already the
   `waypoint_ends` "arrival end" rule — this is WHY.
2. If a point must be on a moving trajectory, don't fix it by widening `tol`
   (a larger y-window risks FALSE positives, e.g. crediting a milestone while
   falling past it). Detect the SEGMENT instead — "on the Lesc edge" via
   `agent_ladder_from_pixel_xy` / pose 13 is dwell-independent and true for the
   whole ride (role 5 = pose trigger, machinery we already have).
3. Remember the unit asymmetry above before reasoning about any tolerance.

Not fixed today on purpose: the escalator is solved (`prog` from `Lesc_top`
0.83, and `Lesc_top -> Ldown_bot` = 82.7%) and nothing needs escalator seeds.

## INVARIANT: route progress must survive a seed load

Learned from the L3 "paid to retreat" bug (see level3_notes.md):

> Any reward state representing ROUTE PROGRESS must either be derivable from the
> EMULATOR state, or be captured with the seed and restored on load.

- **CPs satisfy this for free**: fruit presence / the princess flag are RAM, so a
  collected fruit IS gone in the save-state.
- **Milestones did NOT**: "reached" is POSITIONAL, held in the reward's
  `_reached_wp`, and wiped by the per-episode reset. Seeded episodes therefore
  re-targeted milestones BEHIND them, and the potential paid them to RETREAT
  (the milestone-sum was globally minimised at SN3: seeds above gained ~+3.7 by
  descending; a seed AT SN3 lost reward for leaving -> SN3->A1 measured 0%).
- **Fix**: seeds carry the reached set (accumulated TRANSITIVELY along the
  reverse-curriculum chain), restored via `rewards.restore_reached_waypoints()`,
  which accepts either naming scheme for the same point (the curriculum's `A1`
  or the graph's `J10_11_b`). Pre-fix pool files are migrated by
  `scripts/mo5/yeti/backfill_seed_milestones.py`.
- Guarded by tests in `test_rewards.py` (see the "Seeded-start milestone
  restore" block). L1/L2 have no `reward_waypoints`, which is why this bug was
  L3-only and went unnoticed for so long — a new level adding milestones MUST
  re-check this invariant.
