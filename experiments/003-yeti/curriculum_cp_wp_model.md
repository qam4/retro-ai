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
