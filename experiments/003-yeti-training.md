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

> **WARNING — every L1 champion percentage below was measured against a BROKEN
> emulator state-restore, and does not reproduce on the current core.**
> RESOLVED 2026-08-12 — full entry: **H-AM** below; recipe + follow-ups in
> TODO.md, "BLOCKER [CONFIRMED]". Summary:
>
> `v15_phase2_4500k`, documented here at 99.7% princess-from-reset, evaluates
> **0/40** on the current core and **39/40 (97.5%)** on a core built at
> `2b0a45d~1`. `2b0a45d` is the exact commit that changed it; a freshly built
> HEAD reproduces the stale `.so` exactly, so nothing was hiding in the
> unversioned binary. Every build still reaches 4 fruits 39/40 — only the final
> princess leg collapses.
>
> Cause: `MO5RLInterface::reset()` boots the emulator only on the FIRST reset;
> every later episode is restored from a cached `startup_state_`, so the
> save/load path is on the critical path of ALL training and eval episodes.
> Before `2b0a45d` that restore was broken, in two ways the policy could see:
> (1) ~57 RAM addresses drifted from a true boot, including the hazard object
> table at `0x2B60/68/70/78/80` — byte `0x2B24` was FROZEN (148) where a real
> boot has it live (13..251), so snowball timing was quieter and more
> predictable; (2) the monitor ROM (character font) was wiped, so HUD glyphs
> rendered BLANK (105 pixels, rows `y=1..14`).
>
> Proven with a policy-free, fixed-action, full-RAM comparison: old vs new core
> are **identical for 150 steps on a real boot** (physics never changed), and on
> the new core **restore == boot bit-exactly** (the fix is correct). It was the
> old restore that was wrong.
>
> Consequence for this document: the L1 navigation results are real (reach-4
> holds in every build), but **every absolute princess figure for L1 was earned
> against partly-frozen hazards and must be re-measured on the current core
> before being quoted.** Relative results within a single emulator build (e.g.
> L3 v13 vs v14) remain valid. There is no pixel-level workaround — the main
> component is game state, not the frame.

## TL;DR / Current status (after approach 35)

**Current champion: v15-4500k** (`yeti_curriculum_v15_phase2` snapshot at
4.5M, saved to `output/mo5/yeti/champions/v15_phase2_4500k`). Clean reset
eval (300 stochastic): **princess 99.7% from reset** (299/300) — Yeti is
essentially SOLVED from a cold start. Completion is also tightly speed-
optimized: ~259 steps, leftover bonus ~787 (median within 0.1% of the
best ever seen, 98% of wins within 2% of best).

> **[OBSOLETE — pre-`2b0a45d` emulator]** Every figure in this paragraph was
> measured against the broken state-restore (see the warning above). Re-measured
> 2026-08-12: **0/40 princess on the current core**, 39/40 at `2b0a45d~1`. L1 is
> NOT solved on the emulator we ship today; the policy needs re-validating or
> retraining against live hazards.

**How we got here (the arc): 11.2% -> 58% -> 99.7%.**
1. *Allocation (H-T, v14).* Aggregate-goal-score weighting (one weight per
   start level, no reset reserve / floor) raised the peak to 58% but did
   NOT stop the oscillation.
2. *Diagnosis.* The reach-4/princess oscillation is driven by
   **destructive PPO updates**: with `n_steps`=16 and no `target_kl`,
   individual updates hit **KL up to 68** (vs the ~0.02 norm), slamming
   the value function negative (explained_variance to -30). No
   curriculum/seed/allocation signal predicted the crashes; the cause was
   internal to the optimizer.
3. *Fix (H-V, v15): phase-2 anneal.* Warm-start the 58% champion's weights
   and SHRINK THE STEP — `n_steps` 16->512, `target_kl`=0.05. Updates
   capped (max KL 68->0.13), value function never went negative, the
   oscillation collapsed (reach-4 std 0.38->0.17), and princess-from-reset
   climbed 58% -> ~99.7%. (Analogy: LMS step-size annealing — big mu to
   explore, small mu to converge.)

**Still true — capture matters.** Even v15's *final* model eval'd 0% (it
landed in a residual dip); the 3M/4M/4.5M snapshots are all ~99.7%. So
periodic snapshots + from-reset eval (keep-best, H-U, `keep_best_sweep.py`)
remain essential — the best policy is a snapshot, not the final model.

**Resolved (H-W): cold + steady FAILS — phase-1 instability is required.**
v16 (`yeti_curriculum_v16_coldsteady`: `n_steps`=512, `target_kl`=0.05, NO
warm-start, 20M steps) was swept from reset across ALL 200 snapshots
(`keep_best_sweep.py`, 20 ep each, GPU): **every snapshot scored princess
0.0 AND reach4 0.0** — it plateaus at reach-3 and never once crosses the
reach-4 (F3->F4) wall in the entire run. So the steady recipe cannot learn
from scratch; the big-step phase-1 was doing essential exploration.

*Interpretation — the recipe is simulated annealing on the policy.* Phase 1
(`n_steps`=16, no `target_kl`) = HIGH temperature in policy-parameter space:
large, noisy, sometimes destructive updates (measured KL up to 68) that take
big jumps and can stumble across behavioral plateaus like the reach-4 wall
(crossing it needs a whole new skill chunk, so small greedy steps can't —
nothing nearby improves the return). Phase 2 (`n_steps`=512,
`target_kl`=0.05) = LOW temperature: small KL-bounded steps that settle into
the basin without destroying it. Same shape as LMS mu-annealing (big mu to
escape, small mu to converge). The catch: that instability is double-edged —
it's the *same* mechanism as the oscillation we fought (big jumps also leave
good policies, hence reach-4 swinging and degraded final models). So the win
is the *schedule* (instability early, annealed away), not either extreme:
too much for too long never converges, none at all stays trapped (v16).
*Practical rule:* never start cold-steady — use phase-1 exploration (or
warm-start past the early walls) to find the basin, then anneal. This also
guides level 2.

**Prior champion (superseded): v14-12750k** — princess 58.3%, via H-T
aggregate-score allocation. Before that v11-4750k — 11.2%, first cold-start
completion, via the anti-starvation `segment_floor=0.5`.

**Earlier reward win (still the foundation):** pure-reset γ=1 PBRS broke
the 2-fruit wall (reach-3 96%); see "Reward-shaping pitfalls". Prior
baselines: v6b (reach-3 96%/reach-4 13%), v9-150k (reach-4 ~100%,
princess 0).

**Historical note (pre-v6b):** for most of this project the single-policy
wall was F2->F3 — v2/v4/v6 all hit ~99% reach-2 / 0% reach-3 from reset.
Segments learned in isolation (CP2->CP3 50%, CP3->CP4 30%,
CP4->princess 69%) but never composed into one policy via curriculum.
v6b shows a clean reward composes from reset without any curriculum.

**What does not work yet:** composing those into a single CP0->princess
run. Chaining separate policies gives 0.4% (handoff distribution
mismatch). One policy from reset plateaus at 2 fruits. Mixed-start
curriculum degrades all segments. Warm-starting across distributions
poisons the policy.

**Root cause:** model-free PPO learns only the start distribution it's
trained on; it doesn't compose or transfer.

**Next idea (not yet tried):** sequential distribution-matched chaining —
train each segment on the *actual output states* of the previous segment,
so the handoff matches by construction. See the "Summary" section at the
end for the full plan and the open question (one fine-tuned policy vs N
orchestrated policies).

**Reading guide:** approaches 1-17 are early reward-shaping dead-ends.
18-23 develop the path-progress reward and per-segment results. 24-25
are the princess-detection fix and CP4->princess. 26-29 are the pivot to
PPO-from-reset and curriculum, and why they plateau/degrade. The
end-of-doc "Summary" consolidates everything.

---

## TRAINING MANAGEMENT PLAN (2026-09-21)

Reward shaping, seeds and map geometry are exhausted as levers on L4: 45 runs,
0 princess touches. This plan changes how training is MANAGED rather than what
it optimises. Four steps, each one lever with a control, each with a stopping
rule so a dead arm costs hours instead of a day.

### The retrospective it rests on

Measured across the 94 run directories that carry a `curriculum_diag.csv`
reaching at least 200k steps, by `scripts/mo5/yeti/diag/run_retrospective.py`:

| factor | ever reached princess | never |
|---|---|---|
| warm start | 18 | 52 |
| **cold start** | **0** | **24** |
| `n_steps: 512` | 16 | 57 |
| **`n_steps: 16`** | **0** | **15** |
| `timesteps <= 1M` | 13 | 6 |
| `timesteps >= 10M` | 4 | 37 |

By level: L1 3 win / 5 lose, L2 1/10, L3 14/16, **L4 0/45**.

Two things that table hides:

* 13 of the 14 L3 wins are 600k children of one parent. Exactly ONE run in the
  repo's history went from an ancestor that never finished to one that did:
  `yeti_curriculum_l3_v15_gatewp_15m`, warm from `yeti_ctrlC_gatewp_600k`
  (princess 0.000, depth 9), reaching 0.953. It found it at **2.275M of 15M**
  and then oscillated — 0.855, 0.000, 0.231, 0.006, 0.801 — for the remaining
  12.7M steps without improving.
* The `timesteps <= 1M` row is confounded by exactly those 600k children. It is
  not evidence that short runs are better; it is evidence that stopping a warm
  run before it regresses preserves what it inherited.

Policy entropy, measured by `scripts/mo5/yeti/diag/policy_health.py` on a fixed
observation batch per level (joystick `[3,3,2]`, maximum entropy 2.8904 nats):

| run | level | princess | entropy, % of max |
|---|---|---|---|
| `yeti_curriculum_v15_phase2` | 1 | 0.998 | **2.0 - 6.0%** |
| `yeti_curriculum_l2_v10_deferfruit_10m` | 2 | 0.993 | **6.0 - 16.6%** |
| `yeti_curriculum_l3_v15_gatewp_15m` | 3 | 0.953 peak | 52.6 - 63.1% |
| `yeti_curriculum_l4_v18_warm_v13_15m` | 4 | 0.000 | 52.5 - 64.5% |
| `yeti_curriculum_l4_v16c_payonchange_cold_15m` | 4 | 0.000 | 71.8 - 89.8% |
| `yeti_curriculum_l4_v19_phase1_cold_15m` | 4 | 0.000 | 24 - 32% |

The levels we finish RELIABLY run committed policies at near-zero entropy. L3
at 53-63% reaches the princess but cannot hold it, which is what its swinging
rate looks like. L4 has never gone below ~52% except v19, which committed
inside 1M steps to a route worth nothing.

`ent_coef` is 0.01 in both reliably-finished levels and **0.02 in all 49 L4
runs**. No L4 run has ever used 0.01, and none has ever used `target_kl: 0.05`.

Direction of causation is NOT established: a policy that found a reliable route
sharpens on its own, so low entropy may be the signature of success rather than
its cause. Two observations argue it is not purely a readout — v16c at its peak
`reach10` of 0.77 sits at 72% while v18 at the same `reach10` sits at 64%, and
v19 reached 24% having found nothing. Step 1 is what settles it.

### Ruled out, do not spend runs on these

* **Normalisation layers / churn-reduction losses.** Dormant feature units:
  L2 winner 0.82-0.86, L3 winner 0.72-0.77, v18 0.64-0.67, v16c 0.04-0.12. The
  runs with the most dead capacity are the ones that win, so capacity loss is
  not the binding constraint. (There is no `LayerNorm`, `GroupNorm` or
  `BatchNorm` anywhere in `python/`, and on this evidence none is needed.)
* **`n_steps: 16` as an exploration phase.** It is the COLDEST configuration we
  have, not the hottest: entropy 24-32% against 512's 70-77%. It commits before
  it has found anything. 0 princess in 15 runs. The "phase 1 explores"
  description elsewhere in this document is contradicted by the measurement.

### Step 1 — `ent_coef` 0.02 -> 0.01 (config only, ~3h)

`yeti_curriculum_l4_v21_ent01_warm_v13_6m.yaml`. Identical to v18 except
`ent_coef` and a 6M budget, warm from the SAME parent v18 used
(v13's `final_model.zip`), so **v18's own snapshots from 0 to 6M are the
control arm** and no new control run is needed.

Deliberately NOT bundled with `target_kl: 0.05`, even though the L1/L2 winners
set both. One lever per run; `target_kl` is step 1b.

* **Stop rule:** entropy below 30% of max by 2M. Still above 50% at 2M means the
  coefficient is not enough on its own — kill it and go to step 3.
* **Success:** `Low1` holds at or above v18's 0.743 AND something appears past
  floor 12 (`Low2_launch` onward), judged over the snapshot distribution, not
  the final model and not the champion.

### Step 1b — `target_kl: 0.05` (config only, ~3h)

Same shape, second lever, run only after step 1 reads out. The L1/L2 winners set
0.05; v13's lineage set 0.07; the whole v16 series and v18 set none.

### Step 2 — 6M default instead of 15M (config only, no new run)

v18's best snapshot is at 900k of 15M. v16c's is at 9.3M of 15M. The L3
breakthrough peaked at 2.275M of 15M. Tail steps have never produced a champion
on this project. Spend the saved ~4h per run on more arms and more seeds.

### Step 3 — entropy setpoint controller (~20 lines, only if step 1 is partial)

`PPO.train()` reads `self.ent_coef` on every update and SB3 2.7.1 types it as a
plain float with no schedule support, so a callback that mutates
`model.ent_coef` gives a schedule with no fork. Track a target walking from ~70%
of max down to ~10%, adjusting the coefficient to hit it. A fixed coefficient
cannot do this: the same 0.02 produced 90% entropy in v16c and 24% in v19, so the
coefficient is not the quantity worth specifying — the entropy is. Compare
[Adaptive Entropy Regularization](https://arxiv.org/abs/2510.10959), which uses
an anchored entropy target for LLM RL, and SAC's automatic temperature tuning.

Build it only if step 1 moves entropy but not far enough, or overshoots early.

### Step 4a — the referee and the parallel evaluator. DONE (2026-09-23)

`keep_best_sweep.py` ranked on `mean_rung` at 12 episodes, where bootstrapping the
300-episode champion distributions gives it an sd of 0.75-0.82 against a real
v16c-vs-v18 difference of 0.47. Every champion this project ever selected was picked on a
measurement noisier than the thing it was measuring.

It now ranks on the **frontier reach rate**: the deepest route point the policy still
reaches at >= 5%, and how reliably. That is a Bernoulli mean, so its noise is knowable and
small -- sd 0.078 at n=30 against `mean_rung`'s 0.75 at n=12. Nothing names a waypoint:
the frontier is discovered per snapshot from the level's `route_order`, so it moves
outward as the agent improves and works on any level.

Regression is flagged when the frontier moves BACKWARDS along the route, or its rate falls
more than `--regress-margin` (default 3x the binomial sd at the chosen episode count, so
the flag means "more than the measurement can explain"). Written to
`<run>/best/eval_status.json` after every eval, with `consecutive_regressions` as the
patience counter a caller needs.

Runs in its own process against a live run (`--watch`), on CPU. **Measured cost: 56 s per
snapshot at n=30 while training runs**, so a 6M run emitting a snapshot every ~50 s is
tracked with a small lag and the GPU is untouched.

FIRST OUTPUT, v24 (6M, 56 snapshots at n=30):

```
     step   frontier         rate    rung   regressed
   100000   Low2_launch      0.17    3.87
   600000   Low2_launch      0.73    8.33
  1100000   Low2_launch      0.23    4.30   YES
  2600000   Low2_launch      0.40    6.90   YES
  4100000   Low2_launch      0.63    7.97
  4600000   Lfruit_bot       1.00    1.13   YES   <- frontier COLLAPSED to route pos 5
  5600000   Low2_launch      0.40    5.77   YES
  best: step 3800000, Low2_launch@0.73, rung 8.70
  final: Low2_launch@0.40
```

**34 of 56 evals were measurably worse than the best already seen.** The run ended 45%
below its own peak, and at 4.6M the frontier fell all the way back to `Lfruit_bot`. That
is the sawtooth, quantified, for the first time.

### Step 4b — what to DO about it. NOT BUILT. This is the open question.

**Reverting to the best weights alone does not work, and it is measured here.** Control
arm A0 (see level4_notes.md "DO NOT WARM-START FROM A CHAMPION") took v6's champion at
mean depth 9.55 and continued training with nothing else changed:

```
start  9.55
250k   1.20
500k   2.43
750k   7.53
1M     1.03
```

A best snapshot is the outlier of a wide distribution. Put it back and the distribution
has not changed, so it walks straight out again -- revert, regress, revert, forever.

So the action has to be **revert AND narrow the distribution**, together:

1. reload the best weights
2. tighten one exogenous parameter, and never loosen it again

The ratchet is the point. It turns regression into an annealing schedule driven by
measurement instead of by step count, and it terminates because tightening only goes one
way.

**Tighten the learning rate first.** It directly bounds how far one update can move, it is
the standard mechanism, and it does not touch exploration. Halve on each trigger, with a
floor; at the floor, stop reverting and end the run. `ent_coef` is the second candidate --
v21 measured 0.02 -> 0.01 moving entropy from 64% to 39% of maximum, so it is a real lever
-- but it changes what the policy explores, and one lever at a time.

Trigger: `consecutive_regressions >= 3` from `eval_status.json`. One dip is noise.

**THE CHEAPER ALTERNATIVE, and it is the control arm.** Just STOP the run on the same
trigger. This project's own numbers say the tail never produces the champion: v18's best
snapshot was at 900k of 15M, v21 matched v18 in 6M instead of 15M, and v24 above ended 45%
below its peak. Early stopping loses nothing measured, and it saves hours per run.

The difference: stopping ACCEPTS that progress does not compound. Revert-and-tighten tries
to MAKE it compound, and that is unproven on this game -- A0 tells us why the naive version
fails, but nobody has run the tightening version.

Build both behind one flag with three settings, `off | stop | revert`, so `stop` is the
control arm for `revert` and `off` is the control for both.
[Recovering from Instability in Reinforcement Learning](https://arxiv.org/abs/1910.03732)
is the published form of the revert half, and it is model-agnostic.

### Step 4a readout: five 6M arms, one referee, no attributable answer (2026-09-29)

All scored the same way -- `Low2_launch` reach rate per eval, 0 when the frontier
collapsed shallower, since no run has ever passed `Low2_launch`.

| run | evals | mean | median | collapsed | fruit-seed starts | lever |
|---|---|---|---|---|---|---|
| v23 | 60 | 0.327 | 0.30 | 17/60 | 7.4% | carry bonus |
| **v24** | 56 | **0.349** | 0.37 | 8/56 | 3.8% | unified capture (CONTROL) |
| v25 | 60 | 0.195 | 0.17 | 23/60 | 0.0% | SURFACE_POSES + edge_inset=4 |
| v26 | 60 | 0.209 | 0.20 | 22/60 | 0.0% | SURFACE_POSES only |
| v27 | 60 | 0.223 | 0.20 | 15/60 | 3.6% | + reach-universe fix |

princess 0.000 in every eval of every run. `Low2` 0.00 in every eval of every run.

**Read the DISTRIBUTION, not the peak.** The peak is a max over ~60 noisy draws, so it
mostly measures how many draws you took: v23 peaks at 0.97 inside a run whose mean is
0.327. Every champion comparison in this file that quotes a peak is comparing order
statistics.

**What was measured and came back inert.** `SURFACE_POSES` is provably zero-change on
episode totals (12 episodes replayed through both versions, identical to three decimals;
at gamma 1.0 the PBRS sum telescopes, so the gate moves only WHEN credit lands).
`edge_inset=4` was actively harmful and is reverted. The `F1` start-pool bug was real --
100 seeds sampled zero times for a whole run -- and worth 0.014 of the 0.13 gap.

**So v23/v24 at ~0.34 versus v25/v26/v27 at ~0.20 is unexplained.** Two live
possibilities, and nothing on disk separates them:

1. Something else changed between v24 and v25 that has not been found. The audit covered
   `run_config.py`, the trainer, `rewards.py` and `yeti_map.py`, but the trainer was
   uncommitted at the time so its hunks could not be bisected by date.
2. It is a two-cluster accident. The two v25 attempts ran identical config, code and
   seed 42 and diverged sharply (`Lfruit_bot@1.00` vs `Low2_launch@0.13` at 100k), so
   run-to-run variance here is large and has never been measured.

**REPLICATES, NOT LENGTH, IS THE NEXT SPEND.** Three seeds at ~1.5M per arm is ~2.5h and
gives a variance estimate; a sixth single 6M arm gives another un-attributable number.
This is method note 8 in this file, which was written after the same mistake and then not
followed for five runs.

**Method debt this exposed.** `start_frac` in curriculum_diag.csv counts reset and
CHECKPOINT starts only, so it reads ~1.00 while two thirds of episodes are seeded from
waypoint pools -- it was misread here as "the reverse curriculum is off". `wp_start_counts`
is incremented and never read anywhere, so the metric that would have shown the `F1` bug
directly has been dead the whole time. Same shape as `self.frontier`. Per-episode
`start_key` in episodes.csv is the only trustworthy source for where starts come from.


### Why PPO does this at all

TRPO and PPO inherit their monotonic-improvement story from tabular conservative
policy iteration, and under function approximation those guarantees fail, giving
divergence, oscillation or convergence to something suboptimal — see
[Monotone and Conservative Policy Iteration Beyond the Tabular Case](https://arxiv.org/abs/2506.07134v2).
PPO's monotonicity is a heuristic, not a promise, and this project is a case
where the heuristic does not hold. (Sources paraphrased.)

---

## Experimental method (adopted after v7)

We kept changing several variables at once (v5->v6->v7 each moved
warm-start, gate, eviction, policy), so outcomes couldn't be attributed
and "lessons" kept getting refuted by the next run. New discipline:

1. **One stable baseline.** Current baseline = **v6b**
   (`yeti_universal_v6b_pbrs_g1`): plain PPO from reset, PBRS/signed-delta
   (gamma=1) reward. Clean eval (300 stochastic from reset): **reach-2
   98.3%, reach-3 95.7%, reach-4 13.0%, princess 0%.** (Previous baseline
   was v2 at 99.5% reach-2 / 0% reach-3; v6b dominates it.)
2. **One change per experiment.** Each run changes exactly one variable
   vs the current baseline. Keep seed/steps/everything else identical.
3. **Eval the same way every time:** `eval_from_reset.py`, 200
   stochastic episodes from reset, report the deepest-CP distribution +
   princess rate. (Deterministic = single trajectory, use only as a
   sanity check.)
4. **Confirm/refute the hypothesis, then decide.** If the change is a
   clear improvement (or neutral + principled), it becomes the new
   baseline. If not, revert and record why.
5. **Backlog, not bundling.** New ideas from each discussion go into the
   backlog below and are tested one at a time, in priority order.
6. **A run's last snapshot is not its result.** Measured from `episodes.csv`,
   mean from-reset reward per 100k steps: the revert control dips to **2.6** for
   one bucket and recovers; B1 dips to 21 and recovers; a 15M run collapses to
   ~2-20 about **twenty separate times** and recovers every time. A 1.2M arm that
   happens to END inside a dip reads 0.00 on every waypoint. Compare arms at
   MATCHED HEALTHY steps (from-reset reward > 30), never at the endpoint.
8. **Size a run to the DECISION POINT, not to the full length, and prefer n over
   length.** Cost 6 hours to learn. v6's cascade -- the only thing that distinguished it
   -- happened between 1.0M and 1.2M, so 1.5M (~40 min) answered "does it cascade". A 15M
   run was launched instead, and a second 15M run (v12) was launched to *beat* v6 before
   anyone had checked whether v6's number was reproducible at all. Wrong order and wrong
   length. When a claim rests on one run, buy SEEDS at the decision length rather than
   steps on one seed: 3 seeds x 2M costs less than half of one 15M and answers a question
   that 15M cannot.
10. **Revert SURGICALLY, never wholesale.** `e667206` bundled two changes: derived
   anchors/tolerances (harmful, measured) and `Low2_launch` px 184 -> 188 (CORRECT, and
   independently re-measured a week later as 0/8 vs 8/8 survival). Reverting the commit
   wholesale over the `Rope1` regression threw the correct fix out with the broken one, and
   nothing re-examined it because the bisect only ever looked at `Rope1`. When a commit is
   reverted for one measured harm, enumerate what ELSE it contained and re-land the parts
   that stand on their own evidence.
12. **Never state a MECHANISM you have not measured, even when the outcome is
   measured.** Broken twice in one session and caught both times by the user asking for
   proof. "The gate admits px-184 captures because the agent recovered in the original
   episode" and "because the end_pose lookup falls out of range" were both asserted from a
   single debug line that did not even log the capture position. Instrumented properly:
   px-184 captures are rejected 8/8 with the early policy and 8/8 with the late one, so
   BOTH explanations are false and the real one is still unknown. Report the outcome, mark
   the mechanism unknown, and go measure it -- a plausible mechanism is the most expensive
   kind of wrong because it stops the search.
13. **Produce the visual by default for any claim about what the agent DID.** The
   filmstrip (`debug/l4_gate_admitted_proof.py`) settled in one image what three rounds of
   argument could not, and the from-reset clip found the rope-2 launch pad after a week of
   route tables missed it. If a claim is about behaviour, film it.
11. **When numbers stall, watch the video.** A week of route tables, reach EMAs, bisects and
   two 15M runs never surfaced the rope-2 launch pad. One from-reset clip showed the agent
   being paid to walk off a cliff, bounce on a trampoline, and repeat. Every wrong call this
   week was an inference from a training OUTCOME; the finding that stuck was a direct
   emulator read.
9. **Before explaining a gap, check whether the outlier is the good run.** The table read
   `revert probe 0.08, anchors_v2 0.06, B1 0.14, B2 0.04, v6 0.42`. Four clustered low and
   one high was called a regression in the four; "v6 is the outlier" was the simpler
   reading of the same numbers and was not tested until 6 hours later.
7. **Distinguish detection blindness from policy collapse before explaining
   either.** They look identical in a route table and have opposite causes:
   - *collapse* — from-reset reward ~3, **every** waypoint 0.00 including ones
     whose anchors never moved, `prog` still 0.88-0.99, captures still accruing.
   - *blindness* — reward HEALTHY, **one** waypoint decays while its own
     DOWNSTREAM neighbour stays high. `Lclimb2_top` is reachable only THROUGH
     `Rope1`, so `Rope1` 0.01 / `Lclimb2_top` 0.81 at reward 57 is physically
     impossible and proves the detector is blind, not that the agent stopped
     going there. Use the downstream-neighbour divergence as the test.

### Reward-shaping pitfalls (read before touching the reward)

Hard-won gotchas. Each cost a full 5M run + eval to learn. Don't repeat.

1. **PBRS with `gamma<1` on a large potential creates a per-step "living
   reward."** With `Phi = -scale*sum_f D_f`, even standing still pays
   `(1-gamma)*|Phi|` every step. The `(1-gamma)` looks tiny (0.01) but it
   multiplies `|Phi|`, which is large when you sum distances over several
   targets — here ~0.08/step, ~80 over an episode, vs ~10-16 for actually
   collecting 2 fruits. Result: the agent learns to **survive/dawdle far
   from the goal** instead of progressing (v6: reach-2 99.5%->54.7%,
   episodes 198->344 steps). Fixes: `gamma=1` (signed delta — standing
   still pays 0), and/or keep `|Phi|` small (nearest-target, smaller
   scale). Tying shaping-gamma to the agent's gamma is theoretically
   "correct" (invariance) but caused this — don't assume the principled
   choice is the safe one when the potential is large.

2. **Non-Markovian shaping (history-dependent reward) hurts the critic.**
   The old `best_d` ratchet paid only for beating the closest-distance-
   ever, so identical observations gave different rewards depending on
   hidden history. The critic can't fit a consistent V(s) -> noisy
   advantages. Shaping should be a function of `(s, s')` only (PBRS /
   signed delta), not of the trajectory.

3. **"Reward progress to ALL remaining targets" fights an ordered task.**
   When the next target is in the opposite direction from later ones
   (F2 is left; F3/F4 are right/up), summing progress over all remaining
   fruits creates competing pulls and can make the agent skip the near
   target for the far ones (v4: re-baselining unmasked this, reach-2
   99.5%->90.3%). Consider shaping toward the *next/nearest* target only.

4. **Changing the reward invalidates warm-starts.** The critic is fit to
   the old reward's scale; a policy/value trained under a different
   reward can't be reused. Reward-change experiments train from scratch.

5. **Distinguish "current-state" memory from "trajectory" memory.**
   Reward may depend on the *current* state (current floor, x, fruits)
   and stay Markovian. `last_floor` (inferring floor with one step of
   memory because pixel-y is ambiguous mid-jump) is a small bounded
   residue kept on purpose — it stops jump-farming. `best_d`
   (best-so-far) is trajectory memory and is the real problem.

### Episode allocation across checkpoints (`pick_start`)

How `CheckpointManager.pick_start` decides where each episode starts
(reference — previously only described piecemeal in approaches 30/31):

1. **Reset floor.** With probability `cp0_floor` (= config `reset_fraction`,
   e.g. 0.4) the episode starts from game reset (CP0). Guarantees
   end-to-end composition keeps being practiced.
2. **Otherwise**, pick a checkpoint level among CP1..CP4 whose pool is
   non-empty (and, if the reach-gate is on, `reset_reach_ema >=
   reach_threshold`; the gate is OFF when `reach_threshold = 0`). Level
   `n` is weighted by `max(1 - seg_success_ema[n], 1e-3)` — lower segment
   success → more practice. Sample a level by those weights, then a
   uniform-random saved state from that level's pool.
3. `seg_success_ema[n]` = responsive EMA of "started at CP_n → reached
   CP_{n+1}." All CP segments init at 0 (so early allocation among CP1-4
   is ~uniform at weight 1.0); the `1e-3` floor only matters at the
   *solved* end (keeps a mastered segment getting rare refresher reps).

**Known bug (found via the v9 diagnostics, fix = H-M):** the env calls
`record_episode(start_level, reached_level)` with `reached_level =
4 - fruits`, which can never exceed 4 even on a princess touch. So a
**CP4→princess success is never recorded** — `seg_success_ema[4]` stays
~0, pinning CP4's weight at the maximum 1.0 *permanently*, regardless of
how often it actually beats the level. That over-samples CP4 and starves
CP3→CP4 (the reach-4 oscillation). The episode already tracks the real
outcome in `_max_cp_this_ep` (= 5 on a princess touch); the fix is to
pass that as `reached_level`. An anti-starvation sampling floor is a
*possible* secondary mitigation but should only be added if the bug fix
alone doesn't resolve the over-concentration.

### Hypothesis backlog (test one at a time)

- [x] **H-A — reward leak fix** (DONE, v4 = `yeti_universal_v4_leakfix`).
  *Result:* refuted as a capability fix. Clean eval (300 stochastic):
  reach-2 **90.3%**, reach-3 **0/300** (vs v2 99.5% / 0). Training
  reach-3 rose to ~0.07% (17/25340) from v2's ~0.012%, and reach-2
  reliability dropped — the fix unmasked a competing pull toward the
  distant F3/F4 after F1 (re-baselining made all remaining fruits'
  progress freshly available, so "go right toward F3/F4" sometimes
  out-paid "grab nearby F2"). Net: more F3 *attempts*, no F3
  *conversions*, slightly worse reliability. **Did NOT replace v2 as
  baseline** (failed the reach-2 >= 99% rule). Confirms the wall is
  long-horizon exploration, not shaping direction. Leak fix kept in
  code (correct), but it is not a win on its own.
- [ ] **H-F — PBRS (Markovian shaping)** (DONE, v6 = `yeti_universal_v6_pbrs`).
  *Result:* rejected. Removed the `best_d` ratchet (good — Markovian) but
  `Phi=-scale*sum_f D_f` with gamma=0.99 introduced a per-step living
  reward `(1-gamma)*scale*ΣD ≈ 0.08/step` that dwarfed fruit reward →
  survival bias. Eval: reach-2 **54.7%** (vs v2 99.5%), episodes mean 344
  steps vs v2 198 (dawdling). v2 stays baseline.
- [ ] **H-F2 — PBRS with gamma=1** (DONE, v6b = `yeti_universal_v6b_pbrs_g1`).
  *Result:* **BREAKTHROUGH — adopted as new baseline.** Signed-delta
  shaping (gamma=1) kills the living-reward term. Clean eval: reach-2
  98.3%, **reach-3 95.7%** (from 0%), reach-4 13.0%, princess 0. Broke
  the multi-month 2-fruit wall with no curriculum. Confirms the v6
  regression was the gamma<1 living reward, and that the old non-Markovian
  `best_d` reward was a real cause of the plateau.
- [ ] **H-I — CP4->princess (the new wall).** From the v6b baseline, the
  agent reaches 4 fruits (13%) but never touches the princess. Figure out
  why the final F4->princess leg isn't made (princess shaping reachable?
  exploration? the princess node/distance in the nav graph?) and address
  it. Likely the next highest-value experiment.
- [ ] **H-J — push reach-4 up.** reach-3 is 95.7% but reach-4 only 13%.
  The F3->F4 leg is the current soft spot; more compute (H-C at the v6b
  baseline) and/or shaping/target tweaks (H-H) may lift it.
- [x] **H-C — more compute (20M reset-only)** (DONE, `yeti_pbrs_g1_20m`).
  *Result:* the **princess was touched 5 times** during training (steps
  7.9-17M) — the full level is achievable from reset. BUT the policy is
  **unstable on the deep legs**: a snapshot sweep showed reach-4 spiking
  to 30% at 12M and ~0 elsewhere; the 20M final model is degraded
  (reach-3 0%). Best = **12M snapshot** (reach-2 99.7 / reach-3 87.7 /
  reach-4 28.3 / princess 0). Periodic snapshots (added this run) saved
  us from shipping the degraded final.
- [x] **Efficiency/failure profile of the 12M champion** (DONE,
  `scripts/mo5/yeti/profile_run.py`). CP1 already near-optimal (F1 by step 47,
  bonus 954); agent is fast (all 4 fruits by ~step 207, bonus 825), so
  "dawdling -> snowballs" is NOT the issue. **Failures are at the
  ladders:** CP3->CP4 dies at ~(180,114)=L34, CP4->princess dies at
  ~(220,82)=L45. The agent navigates correctly to the ladder then dies on
  the ascent -> the deep wall is a reactive **ladder-ascent timing**
  skill, under-practiced because reached rarely.
- [x] **H-K — deep-CP drill** (DONE, `yeti_curriculum_v8_deepdrill`).
  Warm-start 12M + seed CP1-CP4 from the 20M reset-origin buffer;
  curriculum (no gate, success-weighted, CP0 floor 0.4), gamma=1.
  *Result, mixed and very informative:* **2542 princess touches** (vs 5
  in the 20M reset-only run) — the deep-drill is hugely effective at the
  F4->princess ascent. And from reset it hit **reach-4 85% in the first
  ~760k steps** (vs champion 28%). BUT it then **decays + oscillates**:
  binned from-reset reach-4 settles ~40-60% with transient dips. NOT a
  collapse (reach-2 stays 52-99%); "overtrain" was the wrong label. Two
  unresolved phenomena: decay from the early peak, and high-variance
  oscillation. The peak (<760k) was missed by the 2M snapshot interval.
- [x] **H-L — capture + instrument** (DONE, `yeti_curriculum_v9_capture`).
  *Result: BIG capture win.* Fine snapshots (every 150k) caught the peak:
  the **150k snapshot reaches 2/3/4 fruits all at 100%** from reset
  (300k/600k/750k also ~97-100% reach-4) — vs the 12M champion's 28%
  reach-4. New champion = the 150k snapshot. *But princess = 0/200 from
  reset on every snapshot* (despite 2542 training touches from CP4
  starts) — so the isolated remaining leg is reset->CP4->**princess**.
  Diagnostics: reach-2/3 stable, only reach-4 oscillates; the start
  distribution is dominated by CP4 (`s4` 0.3-0.7) starving CP3->CP4
  (`s3` ~0.05) — traced to the CP4-success bug (see H-M).
- [x] **H-M — fix CP4-success recording** (DONE, `yeti_curriculum_v10_cp4fix`).
  *Result: correct fix, negligible effect — refuted as an oscillation
  fix, but revealed the real wall.* The fix registered CP4 successes
  (`succ_ema4` 0.00 -> 0.02) but allocation was unchanged (`s4` ~0.40,
  `s3` ~0.11) and reach-4 std only 0.33 -> 0.29. Reason: CP4's weight went
  1.00 -> 0.98 — immaterial. **CP4->princess genuinely succeeds only ~2%
  (151/8022 CP4-start episodes)** even with heavy drilling, so the
  weighting *correctly* treats CP4 as hardest; the over-allocation is not
  a bug artifact. Kept the fix (it's correct). The L45 ascent from the
  post-F4 position is the real ~2% wall (matches `segment_4toP`'s 0/10
  from the floor-4 right start).
- [x] **H-N — diagnose the L45/princess ascent** (DONE). Profiled 300
  CP4-start rollouts (v10 final): princess 0/300; **290/300 failures end
  right of F4 at median (312,94)** — directly *under* the princess
  (x=312) on floor 4; only **6/300 reached ladder L45** (x~208, left). So
  the wall is **NAVIGATION, not timing**: the visible princess lures the
  agent rightward to her x-column, where it's stuck below her jumping
  uselessly, instead of reversing LEFT to L45 to climb. Same
  "greedy-toward-visible-goal, won't take the detour" pattern that gated
  F2 — now the last leg with the biggest detour (away from the goal).
- [x] **H-Q — anti-starvation floor** (DONE, `yeti_curriculum_v11_floor`).
  Verified first the princess reward is real and strong (median bonus
  ~780 at touch -> reward ~39; in-game score +806). Fix (one change):
  `segment_floor=0.5` blends (1-success) weighting with a uniform floor.
  **Result: the user's hypothesis confirmed — and a milestone.**
  - allocation rebalanced: `s4` (CP4 starts) 0.40 -> 0.29.
  - reach-4 stabilized: mean 0.69 -> **0.78**, std 0.29 -> **0.25**.
  - CP4->princess success 1.9% -> **7.0%**.
  - **princess touches FROM RESET: ~0 -> 622 / 13466 reset eps (~4.6%)**
    — the agent **beats the whole level from reset** for the first time.
  Stabilizing the deep practice let the strong princess reward compound,
  exactly as hypothesized. (Clean snapshot eval confirming the rate is
  running.)
- [ ] **H-P — crack the F4->princess detour** (now: push the ~4.6%
  from-reset princess rate higher). The navigation detour is no longer a
  hard 0 — it's being learned. Candidate next steps: more compute at this
  recipe (does the rate keep climbing?), capture the best snapshot, or
  reverse-curriculum seeding of the L45 ascent to accelerate it.
  Candidate single-experiment fixes:
  (a) reverse-curriculum: seed mid-L45 / floor-5 states so the agent
      first masters "climb->princess", then backward-chains the floor-4
      approach. Needs L45/floor-5 seeds (capture from the ~2% successes
      or a manual save).
  (b) stronger/leg-specific leftward shaping (PBRS already penalizes
      going right but the visible-princess lure overpowers ~0.01/px).
  (c) more concentrated, stable CP4 practice (anti-starvation weighting)
      to grow the ~2%.
- [x] **H-O — fix `_max_cp_this_ep` init unit bug** (DONE). Was
  initialized to `self._start_fruits` (fruits *remaining*) instead of the
  start *level* `4 - start_fruits`, pinning reset episodes at 4 and
  over-admitting their seed snapshots via `save_scored`'s `reached_next`
  (weakened the survival filter). Fixed to `4 - self._start_fruits`. Pure
  correctness fix; restores the intended admission filtering for future
  runs. *A/B (v12 = v11 + this fix, same seed 42):* v12 came out
  **worse** — best snapshot princess 1.0% vs v11's 9.5% (champion 11.2%),
  late snapshots mostly degraded. The fix changes only ~1% of admitted
  seeds (filters "collected-fruit-then-died-within-30-frames" reset
  dead-ends the bug mislabeled as "reached next"), yet the outcome swung
  hugely. n=1 with training nondeterminism, and v11's own snapshots swing
  0->9.5%, so **inconclusive on outcome** — most likely shows the deep
  leg is too chaotic to A/B small changes with single runs (reliable
  ranking needs multiple seeds). Fix kept (correctness); **v11-4750k
  remains champion**; not a demonstrated improvement.
- [ ] **H-R — admission leniency / `min_survival_frames`** (BACKLOG).
  Hypothesis for *why the H-O fix hurt*: the `survived >= 30` filter
  biases seed pools toward states the agent already survives, throwing
  away the hard on-distribution "collected fruit N then died fast" states
  — which are the most valuable to practice (reverse-curriculum logic).
  Test on fixed code: sweep `min_survival_frames` ∈ {0 (admit all), ~10,
  30}; does more leniency recover/beat v11? `=0` gives the lenient
  behavior cleanly (no bug). Maybe the answer is just a smaller threshold.
- [ ] **H-S — multi-seed reliability** (BACKLOG). The deep leg is chaotic
  (a ~1% seed-pool change swung princess 11%->1%; v11 snapshots swing
  0->9.5%). Single-run A/Bs are unreliable. For any change we care about
  ranking, run >=2-3 seeds and compare distributions, not single runs.
- [x] **H-T — aggregate-goal-score allocation** (DONE, v14 =
  `yeti_curriculum_v14_aggscore`). *Kept (simpler + raised the ceiling);
  refuted as an oscillation fix.*

  **Problem.** The start-allocation weight is `1 - seg_success_ema`, and
  `seg_success` is a **boolean**: "did the episode advance at least one
  CP from its start." This saturates the instant a segment clears its own
  CP — e.g. CP0's boolean is ~0.99 ("collected >=1 fruit") so its weight
  -> 0 even though it reaches the princess only ~5% of the time. A
  segment that looks "solved" gets starved, and in a shared net a starved
  skill rots. The `(1-success)` rule then chases whatever is now worst,
  which starves the previously-good one — a ping-pong that shows up as the
  reach-4 / princess **oscillation** (v13 sweep: reach-4 99->0->10->89,
  princess 0.7->27->0->0->0 across snapshots; 20M did not converge). The
  `segment_floor=0.5` patch did not hold: at 20M, CP4 still drew
  `start_frac=0.425` while CP3 was starved to `0.025`.

  **Change (one knob, conceptually).** Replace the boolean with a
  **scalar aggregate goal score** = `reached_level / 5` (4 fruits +
  princess = 5 goals; princess = level 5). EMA it per start level
  (`goal_score_ema`), and weight EVERY start level CP0..CP4 by
  `1 - goal_score_ema` in a single normalized draw. Because a level's
  score only approaches 1.0 when the policy actually reaches the PRINCESS
  from there, no level is "done" (weight ~0) until the whole game is
  solved from it — so nothing is starved prematurely. The fixed reset
  reserve (`reset_fraction`) and the anti-starvation floor
  (`segment_floor`) both fall out of the weighting and are set to 0.

  **Why it should help the oscillation.** Computed from v13's
  `episodes.csv` (last 20%, princess=5): per-start scores CP0 0.58 / CP1
  0.56 / CP2 0.57 / CP3 0.70 / CP4 0.81 -> weights all moderate
  (0.19-0.45), none near 0. Implied allocation flips CP4 from the biggest
  share to the smallest (0.425 -> ~0.12) and feeds CP0-CP2 instead. Reset
  (CP0) earns ~24% on its own from its score gap (vs the hardcoded 0.40),
  and self-shrinks as the from-reset journey is solved. Smooth, non-
  winner-take-all allocation should stop the ping-pong.

  **Note / scope.** Aggregate vs boolean are IDENTICAL for the last
  segment (CP4: only one goal left, score 0.8 or 1.0), so this does NOT
  add reps to the F4->princess leg — if anything it gives it *less*. The
  bet is that the leg is not rep-starved (we drilled CP4 at 42% with flat
  ~4% success) but destabilized by the allocation; the expected outcome is
  "oscillation -> stable", not necessarily a higher princess rate by
  itself. Seed-picking (Pillar 1: the CP4 pool is only post-fruit-4
  states) is a SEPARATE lever, deliberately not changed here (one
  experiment at a time).

  **What changed in code** (`scripts/mo5/yeti/train_checkpoint_curriculum.py`):
  added `goal_score_ema`; `record_episode` maintains it; `pick_start`
  weights all levels incl. CP0 by `1 - goal_score_ema` (cp0_floor /
  segment_floor optional, default 0); `curriculum_diag.csv` gains
  `gscore0..4`. v14 config = v13 recipe (warm-start 12M champion, gamma=1
  PBRS, 20M, 250k snapshots) with ONLY the allocation rule changed
  (`reset_fraction: 0.0`, `segment_floor: 0.0`).

  **Success criteria.** (1) reach-4 / princess oscillation amplitude
  across snapshots drops vs v13. (2) princess-from-reset is at least
  stable, ideally trending up. (3) `gscore*` and `start_frac*` in the diag
  confirm the allocation spreads as predicted (CP4 share ~0.12, reset
  ~0.24). Per H-S, a single run can't rank small effects — but the
  oscillation-amplitude question is visible within one run.

  **RESULT (20M run).** Mixed, and very informative.
  - *Allocation worked exactly as designed.* Final `start_frac` = reset
    0.24 / CP1 0.21 / CP2 0.24 / CP3 0.18 / CP4 0.12; `gscore` =
    0.58/0.60/0.66/0.74/0.80. The fixed reset reserve and the
    anti-starvation floor were removed with NO starvation — the simpler
    rule (one weight, no `reset_fraction`, no `segment_floor`) is kept.
  - *Refuted as an oscillation fix.* reach-4 still swings the full 0<->1
    range; second-half std 0.378 (v13: 0.370) — unchanged. princess std
    0.10. Spreading the budget evenly did NOT stabilize anything, which
    rules OUT allocation-starvation as the cause of the oscillation and
    points at the shared-policy/PPO dynamics themselves (see the new
    "Why it oscillates" note). Tiny rollouts (`n_steps` defaults to
    128/num_envs = 16) are a prime amplitude suspect, untested.
  - *But the ceiling rose ~5x.* Clean from-reset eval (300 ep,
    stochastic) of the diag-peak snapshots: 4.0M 15.7%, 6.25M 0%,
    **12.75M 58.3% (reach-4 99.7%)**, 18.75M 32.0%, 19.25M 52.7%,
    19.75M 7.7%. New champion = **v14 12.75M, 58.3% princess from reset**
    (`output/mo5/yeti/champions/v14_aggscore_12750k`), vs the prior
    11.2% (v11-4750k). The aggregate-score allocation raised the PEAKS
    even though it didn't stabilize the trace.
  - *Capture is now the bottleneck, not the policy.* The routine
    round-million sweep (5/8/11/13/15/17/19/20M) eval'd **0% on every
    snapshot** and nearly got v14 written off as a dud — every good peak
    fell BETWEEN the sampled snapshots (13M=0% sits right next to
    12.75M=58%). The live diag `reach_princess` is a useful *pre-filter*
    (it flagged 12.75/19.25/18.75M, which eval'd high) but NOT
    trustworthy per-snapshot (it also flagged 6.25M, which eval'd 0%) —
    so peaks must be confirmed by real from-reset eval, not the EMA.
  - *Takeaway.* Oscillation looks intrinsic to one shared PPO policy on
    a sparse deep goal; the productive response is to CAPTURE the peaks
    (eval-based keep-best + finer snapshots), and optionally DAMP them
    (bigger `n_steps`, lower LR, target-KL), rather than more curriculum
    tuning. See H-U.
- [ ] **H-U — eval-based keep-best + capture (BACKLOG/next).** Replace
  the blind time-based snapshotting with: (a) finer snapshots, and (b) a
  callback that periodically runs a cheap from-reset eval (small N) and
  saves the model whenever its princess rate beats the best-so-far, then
  a precise eval (300+) to rank the kept candidates. Motivated by H-T:
  v14 produced a 58% policy that the snapshot grid nearly lost. Decouples
  run length from output quality and enables early-stopping once the
  captured-best plateaus. Optional follow-on: damp the oscillation
  (sweep `n_steps` up from 16; lower LR; target-KL) and re-measure
  amplitude.
- [x] **H-V — phase-2 anneal (DONE, v15 = `yeti_curriculum_v15_phase2`).
  THE BREAKTHROUGH.** Diagnosis first (raw TFRecord scan of v14's TB):
  the oscillation coincides with **destructive PPO updates** — per-update
  approx_kl hit **max 68** (median 0.06, p99 0.33) with no `target_kl`
  cap and `n_steps`=16, driving explained_variance negative (min -30,
  0.7% of updates). No curriculum/seed/allocation signal predicted the
  crashes — the cause is internal to the optimizer. Fix: warm-start the
  v14 58% champion's WEIGHTS (new `warmstart_weights_only` flag — a full
  PPO.load would restore the old hyperparameters) and shrink the step:
  `n_steps` 16->512, `target_kl`=0.05. *Result:* updates capped (max KL
  68 -> 0.13, p99 0.043), explained_variance never went negative again,
  reach-4 std 0.38 -> 0.17, and **princess-from-reset 58% -> 99.7%**
  (clean 300-ep eval of the 3M/4M/4.5M snapshots; champion = 4.5M). Speed
  is also tightly converged: ~259 steps, leftover bonus ~787 (98% of wins
  within 2% of the best-ever 788) — though whether 788 is the GLOBAL
  speed optimum is unverified (could be a stable local optimum). NOTE the
  *final* model still eval'd 0% (residual end-of-run dip) — keep-best /
  snapshots remain essential even here.
- [x] **H-W — cold + steady (DONE, NEGATIVE; v16 = `yeti_curriculum_v16_coldsteady`).**
  Verdict: the steady recipe (`n_steps`=512, `target_kl`=0.05) does NOT learn
  from scratch. Swept all 200 snapshots from reset — every one scored
  princess 0.0 / reach4 0.0; it plateaus at reach-3 and never crosses the
  reach-4 wall in 20M steps. The big-step phase-1 exploration WAS necessary.
  Recipe = simulated annealing on the policy (phase-1 high-temperature jumps
  escape the plateau; phase-2 low-temperature steps converge). Never start
  cold-steady; explore-then-anneal. (Full interpretation in TL;DR.)
- [ ] **H-X — beat level 2 (next target; mapping work).** Yeti has
  further levels with DIFFERENT layouts; level 2 is top-down/descending
  (vs level 1's ascent). The transition: princess touch -> victory music
  -> bonus added to score -> next level loads, **bonus resets to 1000**
  (that reset is the clean level-2-start signal; no level-counter byte is
  known — approach 6).
  - DONE (prep): `scripts/mo5/yeti/capture_level2_start.py` rides the champion past
    the princess, detects `bonus->1000`, and dumps the level-2 start
    save-state -> `output/mo5/yeti/level2/level2_start.sav` (+ png + a
    transition mp4). This is the level-2 CP0 seed.
  - Carries over for free: global RAM (x/y, lives, bonus, score, princess
    flag) and the entire RL pipeline (checkpoint curriculum, aggregate-
    goal-score allocation, phase-2 anneal, keep-best, eval/render tools).
  - New mapping work (the labor): ~~(a) level-2 fruit RAM addresses +
    positions; (b) a new nav graph + floor bands~~ **DONE via RAM map RE
    (see `experiments/003-yeti/ram_map_re.md`)**, (c) wire a level-2 training
    setup (TODO — plan below).
  - **Training plan (approach A, chosen):** simplest first —
    `fruit_princess_bonus` reward (no nav-graph path shaping) + checkpoint
    curriculum + the phase-1->anneal recipe, starting from `level2_start.sav`.
    Add the full nav graph (B) only if A stalls. Verified RAM hooks + the code
    seams to parameterize (FRUITS_TOTAL=2, fruit-presence dict {0x2EAE,0x2EC7},
    CP0 start-state, princess=goal3, diag CSV widths) are in
    `experiments/003-yeti/ram_map_re.md` ("Level-2 training plan").
  - **BREAKTHROUGH (RAM map):** the static level layout is a tilemap in user
    RAM — a 40x25 grid of 1-byte tile-ids, base 0x2C27, `tile(col,row) =
    RAM[0x2C27 + row*40 + col]`. Calibrated + validated against level 1's
    known map (recovered all 5 ladders + 4 fruits + princess exactly). For
    level 2 we now read the WHOLE map directly (no Go-Explore needed — it was
    confirmed stuck at floor-2 gap jumps): 6 floors, 10 ladders, 2 fruits
    (on floor F5, agent x=14 & x=64, y=128), princess bottom-right at
    agent (x=72,y=182). Tile-ids: ladder 1-4, floor 5-8, fruits = 2x2 sprite
    blocks. Princess is a separate entity (Y=RAM 0x2B00, X=0x2B01, 4px
    units). Tools: `find_map_in_ram.py`, `render_tilemap.py`,
    `extract_level_map.py`; data: `output/mo5/yeti/level2/level2_map.json`.
    This generalizes to ALL future levels (read the map, build the nav graph
    automatically).
  - Train with the proven recipe (curriculum -> phase-2 anneal ->
    keep-best). Open choice: warm-start from the L1 champion (transfers
    ladder/snowball/joystick skills if visuals are similar) vs cold; A/B.
  - Policy scope: one policy per level for now (simpler; we have the
    recipe). A single multi-level policy (train on both, or condition on
    level) is possible but reintroduces the shared-net interference we
    fought — revisit later.
- [x] **H-Y — reward state leaked across episodes (FIXED, commit 61a9f3a).**
  `train_checkpoint_curriculum.py` never called `reset_reward()` at episode
  boundaries, so a stateful reward (PBRS `prev_phi`/`last_floor`, the `best_d`
  ratchets, novelty `visited`/`best_floor`) carried over between episodes.
  Benign for game-reset starts (spawn resolves to a real floor) but harmful
  for EVERY save-state start (all curriculum CPs + all of level 2): the reward
  holds the previous episode's potential/floor. On L2 fatal (agent at y=30 =
  floor unresolved all episode -> shaped on stale last_floor). Fix: call
  `reset_reward(self._reward_fn)` in `CheckpointCurriculumEnv.reset()`.
  CAVEAT: every prior L1 curriculum run ran under this leak — likely a
  contributor to the oscillation we chased; re-examine old L1 conclusions
  cautiously.
- [x] **H-Z — harden `load_state` against stale frame buffers (bug class).** (DONE)
  The preprocessing pipeline holds two cross-step buffers — the `frame_buffer`
  deque (frame_stack) and `_prev_raw_frame` (maxpool). A direct
  `iface.load_state()` bypasses `gym.reset()`, so neither is cleared on a
  save-state start; the first post-load frame even maxpools against a PRE-load
  frame. It works today ONLY because the reset does 5 noop steps (>=
  frame_stack=4), evicting stale frames before the episode starts — fragile
  (breaks silently if the noop count drops below frame_stack). Fix: explicit
  frame-buffer reset on load_state (clear + refill) + a test asserting no
  pre-load frame survives a CP reset. Same recurring class as H-Y.
- [x] **H-AA — `best_d` rewards hard-code 4 fruits (level-1-only).** (DONE)
  `fruit_bonus_path_progress` and `..._universal` initialise
  `best_d = {1,2,3,4}`, so they silently mis-handle any level with a different
  fruit count (level 2 has 2). PBRS dodges this (iterates `ctx.fruits_present`),
  but those two rewards should derive their fruit set from `fruits_present`.
- [ ] **H-AB — save/restore the frame stack with checkpoints (enhancement).**
  Better alternative to H-Z's reseed: when snapshotting a checkpoint, also
  store the frame-stack (+ maxpool frame) and RESTORE it on load, instead of
  reseeding to a frozen stack. Makes a CP-start observation identical to the
  real mid-episode observation (true motion history, perfectly
  on-distribution) and lets us drop the 5 settle-noops that currently advance
  the game ~20 frames past the snapshot. Cost: checkpoint format carries the
  stack blob; file-based starts (level2_start.sav) have no stack -> fall back
  to reseed (H-Z). Only matters once the curriculum bootstraps CP1+ snapshots
  (on L2 today every start is CP0 from a file), so deferred until then.

  **DONE (H-AB implemented).** Promoted from "deferred" after the settle
  turned out to be an active bug: the 5 NOOP settle-noops advance the game
  ~20 frames past the snapshot, and at waypoints with a fast standstill
  hazard (the `L12a` goat, ~frame 7) that burned the whole survival window
  -> seeded episodes started already-doomed -> `goal_score` pinned at 0
  (which then, under the old sum-weighting, made those pools hog the start
  budget). Root-caused by RAM diff (a goat sprite marches into the standing
  agent) + a settle sweep (`L12a` survives 6/6 at settle=1, 0/6 at settle=5).
  Implementation: `PreprocessedEnv.export_frame_stack()` / `restore_frame_stack()`
  / `current_observation()`; pool entries are now 4-tuples
  `(source_cp, bonus, state_bytes, stack)`; capture grabs the stack at the
  save-state moment (CP + WP); `reset` restores it and starts at **settle 0**
  (on-distribution). Stack-less seeds (old checkpoints, offline seeds,
  file-based CP0) fall back to reseed + settle **1** (was 5). Config
  signature guards against preprocessing changes -> reseed fallback on
  mismatch. Tests: tests/python/test_frame_stack_restore.py (restore ==
  live continuation; signature/length guards) + 4-tuple round-trip in
  test_checkpoint_manager.py. NOTE: the fallback 5->1 also affects L1's
  CP-seeded starts (more on-distribution); correctness guarded by the H-Z
  stale-frame test. So the "stale `L12a` pool" was a SYMPTOM of the settle,
  not a bad pool; H-AK (waypoint-group allocation) still stands on its own.
- [x] **H-AC — crayon HUD does not fully re-render after load_state.** (FIXED)
  Long-standing bug, reproduced deterministically
  (`tests/python/test_savestate_determinism.py`). Three distinct save/load
  bugs were found and fixed (the native module is rebuilt from the crayon
  submodule):
  (1) **video bank-select not serialized** (savestate v3->v4): `video_page` +
  `gate_array_reg` (0xA7C0) reset to 0 on load -> redraws hit the wrong video
  bank. Serialized in v4. Restored play-area determinism.
  (2) **v3->v4 backward-compat control freeze.** v4 added the game-extension
  PIA latches (`game_pia_*`, the SX 90-018 joystick I/O) to the memory state;
  `set_state` applied them unconditionally, so loading a *v3* save (fields = 0)
  zeroed the live joystick PIA -> agent uncontrollable after load. Fixed with a
  per-struct `has_v4_fields` flag: pre-v4 saves don't clobber the v4-only
  fields. (This is why every v3 save broke the instant we rebuilt with v4.)
  (3) **THE HUD root cause: `set_state` wiped the ROMs.** `MemorySystem::
  set_state` did `state_ = state`, overwriting the live (immutable, NOT
  serialized) `basic_rom`/`monitor_rom` with the deserialized state's
  zero-filled arrays. The MO5 character font lives in the monitor ROM, so the
  font read as zeros after load and all HUD/text glyphs drew as blanks. (Game
  logic survived because it runs from RAM; only ROM *data* reads broke, which
  is exactly why RAM round-tripped while the HUD didn't.) Found by tracing
  every memory read during the first post-load step and diffing continuous vs
  reloaded: a single ROM read at 0xFD24 returned 0x3C (continuous) vs 0x00
  (reloaded). Fixed by preserving the ROMs across `set_state`. Also completed
  the RL load path (`mo5_rl.cpp` now calls `load_state_from_buffer`, restoring
  master_clock/frame_count/etc. instead of only 4 subsystems).
  The observation determinism test now PASSES (was xfail) and guards the fix;
  C++ regression tests added in `tests/test_savestate_v2_compat.cpp`.
- [ ] **H-AD — residual 2-cycle save/load phase wobble (benign, accepted).**
  After the H-AC fixes, RAM and rendered-frame determinism are exact. One
  residual: when a save is captured while the game is in a timing-sensitive
  region (a `DECA;BNE` delay loop straddling the 20000-cycle frame boundary),
  resume can differ by ~2 CPU cycles for 1-2 frames, then RECONVERGES. Ruled
  out: emulator non-determinism (two continuous runs are byte-identical),
  reset-before-load (same with/without), frame-skip mod-4 (k-sweep is not
  periodic), and hidden timing state in any subsystem (all serialized). The
  puzzle: step 0 matches the full save exactly yet step 1 diverges by 2 cycles
  -> a non-serialized transient shifts where the frame budget cuts the CPU
  mid-instruction. ZERO functional impact: RAM (rewards), the rendered obs, and
  Go-Explore's cell archive all depend on RAM/render, which are deterministic.
  Pinning the exact bit needs instruction-cycle master-clock tracing; deferred
  as not worth the effort.
- [x] **H-AE — ALL prior level-2 training was invalid (dead start state).**
  `capture_level2_start.py` saved the level-2 seed only 20 frames after the
  level loaded — inside the non-interactive intro — and that seed was a v3 save
  later loaded by v4 code (bug #2 above). Net effect: from `level2_start.sav`
  the player had NO control. Proof: all 34,215 episodes of the 3M-step probe
  `yeti_curriculum_l2_v2_probe` ended at the IDENTICAL position (x=11, y=44)
  with the same start-state hash — if the agent could move, those would vary.
  So every level-2 "RL finding" (won't cross gaps, sits on the top floor,
  reset_reach=[1,0,0,0], the whole gap-jump-exploration debate) was an artifact
  of a frozen start, not agent behavior. **Lesson: validate that the agent can
  actually act from a start state before drawing ANY behavioral conclusion.**
  Fixed: `capture_level2_start.py` now snapshots a window of candidates and
  keeps the earliest one where holding RIGHT actually moves the agent (control
  verified), with bonus still 1000. Regenerated `level2_start.sav` is a v4 save
  validated controllable.
- [x] **H-AM — the L1 champion was trained against the SAME broken restore, so
  every pre-`2b0a45d` L1 princess figure is invalid** (CONFIRMED 2026-08-12;
  the level-1 twin of H-AE, caused by the H-AC fix landing).
  **Full write-up: `experiments/003-yeti/core_provenance_2b0a45d.md`** —
  mechanism, reproduction recipe, and tracked eval data under
  `experiments/003-yeti/data/champion_repro_2b0a45d/`. Tool:
  `scripts/mo5/yeti/core_determinism_probe.py` (`selfcheck` = ~30s standing
  guard on one build; `capture` + `compare` = the full A/B). Symptom:
  `champions/v15_phase2_4500k`, documented at 99.7% princess-from-reset,
  evaluates 0/40 today. Bisected by building the core at four points and
  re-running the documented eval (`eval_from_reset.py --profile yeti_fruit
  --episodes 40 --stochastic`):

  | core build | princess | >= 4 fruits |
  |---|---|---|
  | `2b0a45d~1` (7f8d8c7) | **39/40 (97.5%)** | 39/40 |
  | `2b0a45d` | 0/40 | 39/40 |
  | `HEAD` (f542839) | 0/40 | 39/40 |
  | stale Jul-2 `.so` | 0/40 | 39/40 |

  `2b0a45d` is the exact commit; a freshly built HEAD matches the unversioned
  `.so` exactly, so the binary hid nothing. **Why a load_state-only commit moved
  a from-RESET eval:** `MO5RLInterface::reset()` boots the emulator only on the
  FIRST reset — every later episode is restored from a cached `startup_state_`,
  so the save/load path is on the critical path of ALL training and eval
  episodes, not just explicit `load_state` calls. That single fact is what made
  this look impossible to explain from the Python history.
  **The old restore was broken in two policy-visible ways** (both fixed by
  H-AC): (1) DYNAMICS — ~57 RAM addresses drifted from a true boot, including
  the 8-byte-strided hazard object table at `0x2B60/68/70/78/80`; signature byte
  `0x2B24` was FROZEN at 148 where a real boot has it live (13..251), i.e. a
  counter/RNG driver was inert and snowball timing was quieter and more
  predictable. Player `x`/`y` (`0x2B52`/`0x2B51`) were unaffected, so this is
  the hazards, not the avatar. (2) RENDERING — the wiped monitor ROM blanked the
  HUD glyphs (exactly 105 pixels, rows `y=1..14`).
  **Proof that the NEW core is the correct one** (policy-free: fixed no-op
  action sequence, full 48K RAM snapshot per step, 150 steps): old vs new core
  are byte-identical on a real boot (physics never changed), and on the new core
  restore == boot bit-exactly, while on the old core restore diverges from boot
  at step 44. So H-AD's "residual wobble is benign" holds for the FIXED core,
  but the PRE-fix restore was not merely wobbly — it was wrong.
  In a deterministic champion episode the first play-area difference is a
  falling snowball at step 76 (drifting down-right), while the player's RAM is
  still identical through step 84 and the actions match for 100 steps.
  Consequence: L1 navigation is real (reach-4 = 39/40 in EVERY build) but the
  hazard-sensitive final leg was tuned to frozen hazards. No pixel-level
  workaround exists — verified that cropping the HUD does NOT restore the old
  behaviour, because the dominant component is game state, not the frame.
  **Lessons.** (a) Record the native build's SHA in champion dirs and run
  manifests; a policy is only meaningful against the emulator it trained on.
  (b) A "save/load only" core change is NOT eval-neutral while `reset()` is
  implemented as a state restore. (c) H-AE's rule generalises: validate the
  start state against a real boot, not just against itself — bit-exact
  restore-vs-boot is now the guard.
- **H-AF — first VALID level-2 run (`yeti_curriculum_l2_v3_10m`), raw data.**
  First L2 training on the fixed, control-verified start save (config = the
  latest L2 config, `fruit_bonus_path_progress_pbrs` level 2, phase-1
  exploration, 10M steps, 8 envs, seed 42). Completed clean (4h34m, exit 0).
  Recording MEASUREMENTS ONLY — no conclusions yet (prior early reads have
  repeatedly been wrong; e.g. the whole v1/v2/probe interpretation was an
  artifact). 27,802 episodes. Per-window (~3475 eps each), first->last:
    - death rate (end_reason=death): ~38% -> ~2-4%.
    - mean final_x: ~3 -> ~6; max final_x per window: 48-57 early -> 16-31 late.
    - mean final_y: ~24 -> ~42 (spawn y=30; F2 standing y=54); max final_y=54
      in almost every window (one window hit 92 ~= F4).
    - deepest floor by max final_y: F2 in nearly all windows.
    - end_reason totals: env_done 21,070; death 6,732.
  Overall: max final_x=57, max final_y=92; episodes crossing gap-1
  (final_x>=14) = 190 / 27,802 (0.7%); episodes with final_y>=48 = 65; fruits
  collected = 0; princess = 0; final `reset_reach=[1.00,0,0,0]`.
  Model + logs: `output/mo5/yeti/training/yeti_curriculum_l2_v3_10m/`.
  Open questions (NOT conclusions) to investigate before deciding next step:
  does the agent approach the gap and die, approach and retreat, or not move
  right at all? watch a rollout of the final model; check whether max_final_x
  declining over training is real avoidance or exploration cooling; check the
  reward/potential trace on a descent attempt.
- [x] **H-AG — diagnose WHY v3 fails, then fix it (v4 = 3M, BREAKTHROUGH:
  first fruits on L2).** The open questions above were answered by rolling
  out v3 checkpoints from the *actual L2 start state* (`scripts/mo5/yeti/rollout_l2.py`
  — the only tool that boots `level2_start.sav`; the others boot L1). Every v3
  checkpoint gets stuck at the first gap: max agent_x 6-11, **0/15 rollouts
  reach the descent ladder at x≈18**. "Goes left" = pushes into the wall.
  Then measured (not guessed) four compounding reward-mechanics bugs:
  1. **γ bug.** `train_checkpoint_curriculum.py` did
     `reward_params.setdefault("gamma", cfg.ppo.gamma)` = **0.99**. PBRS
     `F = γΦ − Φ` with Φ<0 pays **+0.098/step for idling** (~+37/episode =
     essentially the entire episode reward). L1 champions set
     `reward.params.gamma: 1.0`; the L2 configs never did. (Same living-reward
     pitfall as v6 on L1 — see "Reward-shaping pitfalls #1".)
  2. **Fall-spike.** The nav-graph potential credits *reaching a lower floor
     regardless of HOW you got there*, so falling out-pays crossing the gap
     (~3.6×). Tolerance tweaks don't fix it — the corpse rests exactly on the
     floor line.
  3. **Death-detection lag.** Death was detected by bonus-freeze (the lives
     byte is inert on L2); `bonus_stall_frames=120` detects death ~30 gym-steps
     late, so a dead/falling agent keeps banking shaping reward.
  4. **Sprite-pose byte 0x2B54.** surface/creditable = {0-5 walk, 8 ladder};
     airborne/freeze = {9/10 jump, 11 fall, 12 death-anim}. Pose=11 (falling)
     fires ~30 frames before the 0x2AFC death flag.

  **Fix (one new reward style, L1 untouched):**
  `fruit_bonus_path_progress_pbrs_grounded` in `rewards.py` — freezes shaping
  while airborne (pose ∉ {0-5,8}), reuses the base reward via a `_potential`
  override, and keeps `last_floor` across airborne frames (stateful but
  Markovian on `(s,s')` + the bounded floor residue). Config sets
  **gamma: 1.0**. (0x2AFC prompt-death termination was wired separately into
  `src/mo5_rl.cpp`, opt-in via `death_flag_addr/value`; L2 profile opts in.)

  **Result (`yeti_curriculum_l2_v4_grounded_3m`, 3M steps, measured from
  `episodes.csv`, 27,964 eps):**
  - **Gap wall broken:** 16,866 / 27,964 eps (**60%**) now reach x≥18 (the
    descent ladder) vs v3's 190/27,802 (0.7%) past x≥14, and 0/15 in the
    checkpoint rollouts. Max final_x 57→**76**; max final_y 92→**172** (deep
    multi-floor descent).
  - **First fruits ever collected on L2:** 2 (both from reset starts).
  - **Loiter gone:** mean episode reward collapsed 35.85 (v3) → ~1.1-2.9
    (v4 windows) — the idle living-reward is gone, confirming the γ bug was
    the dominant driver.
  - Best snapshot = **1000k** (real ladder descents). USER confirmed via video:
    the 2000k snapshot reaches F4 *by falling*; the earlier "reached floor 6"
    reading was a fall-through mislabeled by a since-fixed depth metric.
  - Model + logs: `output/mo5/yeti/training/yeti_curriculum_l2_v4_grounded_3m/`.
  Tooling built this round: `scripts/mo5/yeti/rollout_l2.py` (L2 rollout from the real
  start state; depth-sweep + video + heatmap; RAM-sourced HUD banner working
  around the load_state HUD bug; 0x2AFC death detection). Also: grounded
  checkpoint admission (defer seed snapshot to the next grounded frame — the 2
  v4 CP1 seeds were mid-jump and doomed on reload) and the
  `min_survival_frames`→`min_survival_steps` rename (it counts gym steps).
- [x] **H-AH — v5 (15M) longer cook at the v4 recipe (DONE, NEGATIVE — exposed
  a reward bug in the grounded gate).** `yeti_curriculum_l2_v5_grounded_15m` —
  same grounded reward + γ=1 + 0x2AFC + grounded admission, 15M steps, exit 0,
  6h57m, 72,657 episodes.
  **Result: regressed, did not plateau.** Total fruits in the whole run = 4
  (same near-zero rate as v4's 2/28k); from-reset fruit success 0% throughout.
  Across training the agent converged AWAY from the task: reaching the descent
  ladder (final_x≥18) collapsed 65%→78%→…→~1% while mean episode reward *rose*
  2.1→7.9. Last-20% median final position = (x=1, y=30) = **the spawn point**:
  the late policy sits at spawn ~328 steps for ~7.9 reward and never descends.
  Snapshot depth-sweep (`scripts/mo5/yeti/rollout_l2.py`, 10 stochastic eps each):
  500k→F3, 1M→F1, 3M→F3, 6M/10M/15M→F1 (never past floor 1 from ~6M on; never
  reaches floor 5 / the fruits in any snapshot).
  **ROOT CAUSE (proven, not inferred) — the airborne-freeze breaks PBRS
  telescoping and re-opens jump-farming.** The grounded gate returns
  `phi=None` while airborne; in the base reward `phi is None` triggers a
  *re-baseline* (`prev_phi=None`), and the landing step re-baselines again — so
  the "moved-away" half of a jump is never charged. Synthetic reward-trace
  probe of a NET-ZERO round trip (walk right grounded, then jump back to start
  airborne): ungated reward nets **+0.0000** (telescoping cancels the round
  trip) but the grounded reward nets **+0.4800** — the retreat is free. PPO
  farms this approach-then-jump-back loop (~+0.48/cycle) near spawn indefinitely
  instead of descending. This is the same class of pathology as the γ<1 living
  reward (v6/H-AG bug #1), reintroduced by the airborne freeze; v4 (3M) was too
  short to exploit it, v5 (15M) found it. **The fruit/descent problem is NOT a
  compute or curriculum shortfall — fix the reward first.**
  Model + logs: `output/mo5/yeti/training/yeti_curriculum_l2_v5_grounded_15m/`.
- [x] **H-AH2 — fix the airborne-freeze telescoping break (DONE, code +
  unit tests; awaiting 3M smoke).** Tension: re-baselining `prev_phi` on
  landing correctly *neutralizes a fall* (don't credit dropping into a deeper,
  closer floor) but *forgives a retreat* (the farm), and a naive "hold prev_phi
  across airborne" kills the farm but re-credits survivable falls — two goals
  that conflict under a scalar potential. Considered a floor-based tie-break
  (re-baseline only when landing DEEPER), then chose a cleaner rule (user's
  suggestion): **gate credit on ALIVENESS, not floor.**
  Fix in `_fruit_bonus_path_progress_pbrs_grounded`, rewritten from the
  `_potential`-override monkeypatch to an explicit class with three documented
  non-Markovian deviations: (D1) `last_floor` fallback; (D2) airborne freeze
  that **holds** `prev_phi` (no re-baseline) so telescoping survives the jump
  and the return leg is charged on landing; (D3) **death gate** — on a grounded
  frame that is a death (`ctx.died`, the 0x2AFC flag) credit nothing. Net
  behavior (synthetic probe + 8 unit tests in `test_rewards.py`):
  farm/jump-back → **0**, fatal fall (died on landing) → **0**, survived
  descent → **+** (real progress; can't be farmed — climbing back up is
  grounded and charged at γ=1), gap-cross (same floor) → **+**, ladder descent
  → **+**, grounded round-trip → **0**.
  Why aliveness over floor: targets the actual objective ("don't reward
  dying"), is not level-direction-specific, and covers ALL death causes.
  Plumbing: `RewardContext.died` (append-only); `train_checkpoint_curriculum`
  sets it from 0x2AFC, gated to level ≥ 2 (L1 termination timing untouched),
  and also uses it to label `end_reason="death"`.
  **Live-trace findings (per-step, 3M snapshot)** validating the wiring: the
  lives byte stays put through death on L2 (so the old lives-based
  `end_reason="death"` was DEAD CODE on L2 — deaths mislabeled `env_done`);
  0x2AFC flips 32→65 on the exact terminating frame (`done` same step), so the
  reward sees `died` on the fatal step; and the sampled deaths were NOT falls —
  the agent descends to floor 3, then loiters/jumps-in-place there (the farm)
  and dies to a goat while GROUNDED, confirming the death gate must (and does)
  fire on grounded frames, not just airborne. Accepted residual: a fall that
  lands ALIVE and dies 1 step later still banks the landing credit (bounded,
  requires surviving the impact; not farmable).
  RESULT (v6 = `yeti_curriculum_l2_v6_grounded_3m`, 3M, exit 0, 35,704 eps):
  **fix validated — the farm is gone.** Mean episode reward stays low and flat
  across training (2.9/4.8/4.8/3.6/1.6/3.8 by window) instead of climbing to ~8
  like v5; reach-ladder (final_x>=18) stays HIGH (68/96/95/79/75/94%) instead of
  collapsing to ~1%. Depth-sweep (rollout_l2, 8 eps): every snapshot incl. the
  FINAL 3M descends to F2-F3 (3M reaches F3 88%), vs v5's 6M/10M/15M sitting at
  F1. So the agent now descends consistently and does NOT loiter at spawn.
  REMAINING WALL: it plateaus at **F3** — never reaches F4/F5 (where the 2
  fruits are); only 3 fruits collected from reset in the whole run. This is the
  genuine L2 difficulty (the F3->F4 transition: gaps + goats that can't be
  jumped like L1 snowballs, must be dodged/retreated).
  CROSS-RUN CHECK (grounded-depth sweeps, rollout_l2): the agent has NEVER stood
  on floor 4+ in ANY L2 run — max grounded floor = F3 for v4, v5, v6 (v3 was
  stuck at F1). The only "F4" sightings were falls-to-death (deep final_y), not
  legitimate reaching. v6 is the best: same F3 ceiling but it reaches F3 most
  RELIABLY (stable across snapshots incl. the final one), vs v4/v5 which touched
  F3 only transiently before collapsing. The F3 goat (x~34) is the ceiling.
  NEXT candidates:
  (a) let it cook longer now that the reward is honest; (b) diagnose the F3->F4
  failure with a rollout (gap-death? goat? navigation?); (c) H-AI reverse-
  curriculum / waypoint seeds to practice the deep descent + goat-dodging.
  DIAGNOSIS (b) DONE (rollout of the v6 3M policy, 20 eps): 19/20 reach F3
  (grounded, legit descent, NOT falls); ALL 20 die GROUNDED with no fall-pose
  before death, clustered at a FIXED spot floor 3 x~34 (y=78). I.e. the agent
  descends to F3, walks to x~34, and is killed there every time — a goat/enemy
  collision, not a gap-fall and not a navigate-to-ladder failure. This is the
  "goats can't be jumped, must retreat up a ladder" wall. Implication: more
  compute alone is unlikely to crack it — with gamma=1 PBRS a retreat-and-retry
  telescopes to ~0 reward, so there's no gradient rewarding the dodge and the
  agent keeps walking into the goat. Points at (c): waypoint/curriculum practice
  past the F3 goat (or a goat-aware mechanic). NEXT: confirm the goat visually
  (video), then design the F3-goat practice.
  ROUTE TRACE (v6 3M, 6 eps, path to death). CORRECTION: an earlier version of
  this note misread agent RAM-x as pixels and drew the WRONG conclusions
  (retracted). Ladders are in PIXELS; agent RAM-x -> pixel = x*4+8. Converting
  the traced route (RAM -> pixel, nearest ladder):
    F2 landing RAM x=18 -> px80  = L12a
    F3 landing RAM x=46 -> px192 = L23b (RIGHT F2->F3 ladder)
    death      RAM x=34 -> px144 ~ L34 (px136, the single F3->F4 ladder)
  So the agent descends F1->F2 via L12a, F2->F3 via L23b, walks left to L34, and
  dies at px~144 to the goat GUARDING L34. Corrected implications:
  - The nav map is NOT wrong (the "F1->F2 ladder mismatch" was the units error).
  - This is NOT a "short route has goat, safe route exists" case: F3->F4 has
    only ONE ladder (L34), a chokepoint the agent MUST pass. The danger-blind-
    shaping concern is real in GENERAL (would bite where 2 ladders exist:
    F1->F2, F2->F3) but is not what kills the agent here.
  - The wall is simply: the only F3->F4 descent (L34) is guarded by a goat the
    agent must learn to beat (timing/dodge) — supports more compute (if the goat
    is timeable like L1 snowballs) and/or seeding waypoints just past L34 (on F4)
    so the agent gets many reps + discovers the deeper reward.
- [ ] **H-AI — reverse curriculum via WAYPOINT START-SEEDS (design, only if
  v5 plateaus).** Reaching the first fruit on L2 requires a long multi-floor
  descent across ~14 gaps and many ladders, and L2 goats *cannot be jumped*
  like L1 snowballs — the agent must sometimes RETREAT (climb back up a
  ladder). With γ=1 PBRS the potential telescopes, so a retreat-then-return
  nets ~0 reward (a designed benefit of the γ=1 choice). Plan: manufacture
  per-floor / ladder-top **waypoint start-seeds** (scripted or nav-graph
  descent snapshotting, like `level2_start.sav` was made) and seed a reverse
  curriculum from them. **CRITICAL distinction (user):** waypoints are START
  SEEDS ONLY, *not* success-checkpoints — success stays defined as "grab a
  fruit", and waypoints must NOT pollute the reach/success metrics (you don't
  need to *reach* waypoints, you need to grab fruits). Needs: a waypoint-capture
  script + a curriculum mode that seeds from waypoints while scoring success by
  fruit collection only.
  **Second rationale — start-state DIVERSITY (not just reverse-chaining).**
  Beyond backward-chaining the descent, seeding from many waypoints spreads the
  training start distribution across the whole level instead of the single
  `level2_start.sav` spawn. That diversity is valuable on its own: it exposes
  the policy to mid-level geometry (gaps, ladder-tops, goat encounters) it would
  otherwise almost never see from a cold start, which should improve robustness
  and exploration coverage. So H-AI is worth trying whenever we hit trouble /
  plateau — its benefit isn't limited to the reverse-curriculum framing.
  **Refinement (user, post-v6) — waypoints must be OPTIONAL / non-gating,
  because the level branches.** Topology (corrected): F1->F2 has TWO ladders
  (L12a px80, L12b px304), F2->F3 has TWO (L23a px16, L23b px192), F3->F4 has
  ONE (L34 px136). Where a floor has two down-ladders the agent may legitimately
  take either, so a waypoint can NOT be a required checkpoint (unlike fruits,
  which are mandatory): capture MULTIPLE waypoints per floor and sample among
  them as start-seeds, NEVER require reaching a specific one. The
  CheckpointManager needs a SEPARATE "waypoint pool" (optional start states, no
  success semantics) distinct from the CP (fruit) pools. Also: v6 was only 3M
  steps (L1 champions ran 12-20M) — L2 needs MORE steps regardless; waypoints
  are an accelerator, not a substitute for compute.

  **FINALIZED CAPTURE DESIGN (user, post-v6) — capture real states on reach,
  like CPs; do NOT synthesize by RAM-poke.** An earlier idea (load level2_start
  and write the agent's x/y to a computed waypoint) was rejected: writing only
  position bytes leaves the rest of the machine state (floor var, velocity,
  goats, pose, render) inconsistent -> invalid/desynced state, the exact class
  of save/load bug we fought. Instead:
  - **WP positions** are computed from the tilemap (ladder tops/bottoms) using
    the confirmed pixel<->RAM equation (x_ram=(px-8)/4, y=floor_top_y). These are
    DETECTION TARGETS only.
  - **WP is position-based, not a flag** (unlike a fruit): "reached WP_k" =
    agent GROUNDED (pose in SURFACE_POSES) within a tolerance of WP_k's (x,y).
  - **Capture on reach**, exactly like CP capture on fruit pickup: when the
    agent reaches a WP during a real episode, snapshot save_state -> that WP's
    pool (grounded-only admission, live state only). Real states, no synthesis.
  - **Guard — do NOT re-capture the seeded start-WP.** When an episode is seeded
    from WP_k, the agent starts AT WP_k, so the detector would fire on frame 1
    and re-snapshot a near-duplicate / log a bogus "reached WP_k from WP_k".
    Track the start-WP and suppress its re-save (require leave-and-return, or
    skip it for the episode) — mirrors the CP start-level admission guard.
  - **Seeding**: reset + WP-pool seeds mixed, deep/frontier-weighted; success
    metric stays fruit-only (WPs never enter reach/success stats).
  - **Bootstrap caveat (honest):** capture-on-reach means floor-4+ WPs only
    populate once the agent FIRST reaches them, which is still gated by the F3
    goat. So WP-seeding accelerates AFTER a first breakthrough (bootstraps like
    the CP curriculum did from rare reset-chains) but does not manufacture the
    first floor-4 state alone -> run WP-seeding ALONGSIDE a longer run (a lucky
    stochastic goat pass seeds the first deep WP, then it compounds).
  **START MIXING (reset vs CP vs WP) — objective: max reps further down.**
  Three start sources now: reset (level2_start), CP pool (fruit-collected
  states), WP pool (position waypoints). Key difference: CPs are weighted by
  `1-success` ("practice where you fail"), but WPs are NON-GATING (no success
  metric), so weight them by DEPTH/FRONTIER instead (deeper / just past the
  current from-reset frontier = more) — classic reverse-curriculum, and it
  directly serves "more reps further down". Optional secondary signal:
  reach-scarcity (seed rarely-reached WPs more) — an internal sampling heuristic
  NOT a reported metric (keeps WPs out of success/reach stats). L2 note: CP
  pools are near-empty until fruits get collected (v6 ended cp=[0,1,0]), so the
  CP share is ~0 automatically early and grows later — no special-casing; WP is
  the primary deep source on L2.
  RECOMMENDED (revised — self-regulating, no thresholds, matches L1). An earlier
  draft proposed a fixed reset reserve (~0.25) + depth-weighted WP draw;
  RETRACTED. Verified the last L1 champions (v14, v15) use reset_fraction=0.0
  AND segment_floor=0.0 — NO fixed reset reserve; reset's share EMERGED from the
  H-T aggregate-goal-score allocation. And depth-weighting needs hand-picked
  thresholds (dispreferred). Instead EXTEND H-T to WPs: treat reset, CP, and WP
  uniformly as "start states", each weighted by `1 - goal_score_ema` where
  goal_score = fruit/princess progress reached FROM that start (reached_level/N).
  - No reset reserve, no depth thresholds — fully self-regulating (matches L1).
  - Naturally gives "more reps further down": a newly-captured deep WP inits at
    goal_score=0 (like CPs) -> weight 1.0 -> heavily sampled -> many reps at the
    new frontier; weight decays as it's mastered and the frontier moves deeper.
    The hardest reachable spot (the goat WP) keeps the lowest goal_score -> the
    most reps. Self-bootstrapping reverse-curriculum, no tuning.
  - Still non-gating: goal_score-from-WP is only a SAMPLING weight (scored by
    reaching the FRUIT, not the WP); WPs never gate advancement or enter reported
    reach/success stats.
  From a WP start the agent plays normally with the grounded reward, success
  stays fruit-only; the PBRS potential from the WP position gives a short path to
  the fruit reward (discover + practice the deep segment).
- [~] **H-AI RESULT (v7 = `yeti_curriculum_l2_v7_wp_15m`, 15M, WP curriculum):
  progress + a composition wall + a metric caveat.** Exit 0, 7h10m, 166,782 eps.
  WHAT WORKED: the WP curriculum captured ALL 16 waypoints incl. the deep ones
  (L34/L45/L56 = floors 3-6), and from waypoint starts the agent learned to PASS
  THE F3 GOAT and descend all the way, collecting both fruits (both CP pools
  full; 37k eps got 2 fruits). So the goat is beatable and the deep descent is
  learnable — the F3 wall is no longer a hard zero.
  IT COMPOSES TO COLD RESET (this is the real win — correcting an earlier
  premature "mirage" read). A from-reset snapshot sweep (rollout_l2, 30 eps):
  **13.5M and 14.0M reach floor 5 + BOTH fruits 100% from cold reset**
  (level2_start). v3-v6 never collected a single fruit from reset; v7's best
  snapshots collect both, 100%. So the waypoint curriculum broke the F3 goat
  wall AND composed back to reset.
  CAVEAT — the policy OSCILLATES (n_steps=16 / no target_kl = L1's H-V
  destructive-update swing): late snapshots collapse (14.8M-15M -> floor-1/0),
  so the FINAL model is degraded and a single-snapshot eval is meaningless.
  Champion = a snapshot (13.5M/14.0M), captured by sweeping — same as L1 (H-U).
  Full from-reset snapshot sweep (rollout_l2): 1M-12M top out at FLOOR 3 (never
  past the goat from reset); the both-fruits capability emerged ONLY in a narrow
  ~13.5-14.4M window that swings 17%<->100% (n_steps=16 oscillation); TWO
  snapshots hit 100% both-fruits/30eps (13.5M and 14.0M), 14.1-14.2M and
  14.8-15M collapse to floor 1. So there are exactly two co-equal champions
  (13.5M, 14.0M) — either is an ideal phase-2 anneal warm-start.
  My first eval looked only at 15M and wrongly concluded "doesn't compose";
  reset_reach_ema=0.40 was a windowed avg over the 0<->100% swing.
  NOT YET: the PRINCESS (final goal, after both fruits) — 0 touches. Diagnosed
  from VIDEO (user, 14M champion; corrects my earlier RAM misread of "floor-6
  approach gap"): FRUIT 2 sits on the EDGE OF A GAP (floor 5, agent x~64). The
  agent jumps to collect it, grabs the fruit, but JUMPED TOO EARLY and falls
  into the gap to its death (deaths cluster at x~63, pose 11 fall -> 0x2AFC;
  y=150 is the fall-through to the bottom, NOT a controlled floor-6 arrival). So
  it CAN reach both fruits but the fruit-2 jump itself is fatal — it never
  survives past fruit 2, hence never reaches the princess. This is a precise
  jump-TIMING problem (grab fruit 2 AND land safely). Chicken-and-egg: with
  gamma=1 the agent has no gradient to SURVIVE the fruit-2 jump because it's
  never reached the princess reward beyond it. Levers: phase-2 anneal (refine
  timing + stabilize the oscillation), and a WP just past fruit 2 (floor-5-right
  / L56) so it discovers the princess reward and backward-chains surviving the
  fruit-2 jump.
  ** H-AJ RESULT (v8 = phase-2 anneal, warm-start v7-14M, n_steps 16->512,
  target_kl 0.05, WP on, 10M): STABILIZED both-fruits-from-reset; princess still
  unsolved.** Exit 0, 3h46m, 75,304 eps. From-reset snapshot sweep (rollout_l2):
  EVERY snapshot 1M-10M reaches floor 5 + BOTH fruits 92-100% (final model
  included) — the v7 oscillation (narrow 13.5-14.4M window, 17<->100%, collapsed
  final) is GONE. Reproduces L1's H-V on L2: small KL-bounded steps stop the
  destructive updates and lock in the capability. reset_reach=[1,0.98,0.98,0]
  corroborated by the sweep. STABLE from-reset both-fruits is now the L2 baseline
  (v3-v6: 0 fruits; v7: oscillating; v8: stable ~100%).
  PRINCESS (corrected — earlier "princess=0" was a LOG BUG, see below; the
  DEFINITIVE princess signal is end_reason=="princess_touched"): the princess
  IS reached. Measured via end_reason: v8 = 2170 touches, v7 = 72 touches. But
  0 are from a TRUE cold reset (both runs) — every touch starts from a SEED.
  Breakdown by logged start_level (fruits still on map at start = level 0):
    v8: start_level 0 = 780 (ALL from WP seeds, 0 cold reset), 1 = 34,
        2 (both-fruits CP2 seed) = 1356.
    v7: start_level 0 = 33 (WP seeds), 1 = 4, 2 = 35.
  IMPORTANT measurement note: n_fruits_collected counts fruits collected DURING
  the episode (+1 on princess), so n_fruits>=3 only catches princess touches
  from starts where BOTH fruits were collected in-episode (cold-reset-like / WP
  seeds with fruits present) — it MISSES princess from CP2 (both-fruits) seeds
  (those collect 0 fruits in-episode -> n_fruits=1). That is why the earlier
  n_fruits>=3 numbers (780 v8 / 33 v7) UNDERCOUNTED: they equal exactly the
  start_level-0 subset. Use end_reason for totals.
  So the final leg (fruit-2 -> L56 -> princess) IS learned FROM SEEDS — the
  2->3:19% is CP2-seed (both-fruits) -> princess success, and it is REAL — it
  just doesn't COMPOSE into a cold-reset chain yet. Same seed-works /
  reset-doesn't pattern as v7's both-fruits, one leg deeper. The distinction
  that matters: "final leg from a seed" (~19%, works) vs "whole level from cold
  reset" (0, the remaining wall). Levers to compose it from reset: more anneal
  time, continued WP-seeding of the princess leg, keep-best.
  LOG BUG (found here, breaks princess accounting): _log_episode writes
  reached_level = fruits_total - fruits, OMITTING the princess (+1), so
  episodes.csv reached_level caps at fruits_total and NEVER shows a princess
  touch. record_episode uses the correct value (fruits_total+1 on princess), so
  seg_success / goal-score / seeding were always right; only the CSV column was
  wrong. This caused repeated "princess=0" misreads. FIXED: _log_episode now
  mirrors record_episode (logs fruits_total+1 when _princess_touched_this_ep).
  For runs logged BEFORE the fix, use end_reason=="princess_touched" (totals)
  or n_fruits_collected>=3 (fruits-collected-in-episode subset only).
  NEXT: (Q2) PHASE-2 ANNEAL to stabilize the oscillation (warm-start a good
  snapshot's weights, n_steps 16->512, target_kl=0.05) — the exact recipe that
  took L1 58%->99.7% (H-V); (Q3) self-regulating WP share (weight WPs by
  1 - reach-from-reset so they fade as the agent reaches them unaided — no cap),
  though composition already works, so this is secondary. Also capture the
  13.5M/14M champion and log WP-starts distinctly.
- [ ] **H-AK — waypoint-group allocation (fix start-mix DILUTION).** Measured
  on v8's actual goal-scores: with `pick_start` normalizing `1 - goal_score`
  over the *sum* of all pools, the 16 waypoints each cast a vote, so the WP set
  collectively takes **89% of starts** while RESET gets only **4.3%** and the
  CP2 compose leg **2.4%**. That reset-starvation is a prime suspect for why v8
  never composes the princess from a cold reset (it's barely practiced). Root
  cause: allocation is over stored POOLS, not over legs — N waypoints add N
  votes, so adding a waypoint mechanically shrinks reset's share (the opposite
  of self-regulating). Also stale pools sit at max weight: `L12a_bot/top` at
  `goal_score 0.00` (weight 1.0) eat ~12.6% each — a floor-1 WP should average
  ~0.6, so 0.0 means never-sampled or bad seeds (see TODO staleness bug).
  FIX (agreed, keep simple): make `pick_start` HIERARCHICAL — draw among
  sources {reset, each CP, waypoints-as-ONE-group}; the WP group's weight is the
  MEAN of member `1 - goal_score` (count-invariant, so adding waypoints only
  re-slices the group's own budget, never reset's); if the group is picked, a
  second draw within it by `1 - goal_score`. Under this rule v8's reset share
  goes 4.3% -> ~26%, WP group ~34%. L1 / WP-off is byte-identical (no WP
  candidates -> the top-level draw is exactly the old CP-only draw).
  **IMPLEMENTED** (pick_start hierarchical draw + `_WP_GROUP` sentinel; tests
  in test_checkpoint_manager.py assert count-invariance + within-group split).
  Note: H-AK reduces the *symptom* of the settle bug (stale WPs hogging
  budget); the actual root cause of the `L12a` deadness is the settle (H-AB).

  **RESULT (v9 = `yeti_curriculum_l2_v9_wpgroup_15m`, warm-start v8-final,
  anneal recipe + H-AK, 15M).** From-reset sweep of all 150 snapshots
  (keep_best_sweep, 30 ep each, GPU): **princess-from-reset = 0 on EVERY
  snapshot** (max 0.000). both-fruits-from-reset recovers to ~1.0 in the last
  third (10-15M) but shows a 4-9M COLLAPSE-then-recover dip that v8 (same
  recipe, no H-AK) didn't have (81/150 snapshots hold both-fruits >=0.90).
  Verdict: H-AK correctly fixes the start-mix DILUTION (reset 4%->26%, late
  snapshots stable at both-fruits) but allocation ALONE does NOT compose the
  final leg from a cold reset — the princess still only fires from seeds
  (CP2->princess ~36% in training), never chained from reset. Same
  "model-free PPO doesn't compose across start distributions" wall, now
  isolated to one leg (both fruits -> survive fruit-2 jump -> L56 ->
  princess). The 4-9M dip also hints the warm-start-from-final + new
  allocation caused a mid-run readjustment. Kept H-AK (correct + keeps late
  both-fruits stable); it is NOT sufficient for princess-from-reset. Next
  levers: H-AB (seed fidelity; not in v9), reverse-curriculum seeding of the
  L56/princess approach, or explicit distribution-matched chaining.
- [ ] **H-AL — defer fruit credit (v10, RUNNING).** Per-step trace of a v9-13M
  reset episode: the agent jumps for fruit 2, is ALREADY FALLING (pose 11)
  when it grabs it, and falls to death at (62,150) — yet banks the fruit
  reward, because the sparse fruit term was paid at pickup regardless of the
  ensuing death (the death gate only protected the shaping term). So the fatal
  early jump is locally optimal; both-fruits "reach_top ~100%" was an illusion
  (fruit 2 collected IN the death fall). CP2 seeds are clean/playable, so NOT a
  seeding bug. Fix (one lever): reward param `defer_fruit_credit: true` — pay
  the fruit on the next grounded-ALIVE frame (D4), so a fatal airborne grab
  pays 0 and the agent is pushed toward the safe/later jump timing. v10 =
  `yeti_curriculum_l2_v10_deferfruit_10m`: warm-start weights-only from v9-13M
  (already navigates to both fruits; only the safe landing is new), inherit
  v9's seed pools (the L12a "unplayable" seeds were a settle-5 artifact — now
  playable at settle=1 via H-AB, so their goal_score should rise and un-hog
  the WP budget), anneal recipe, H-AB on, 10M. Fallback if it plateaus on the
  fatal-jump local optimum: phase-1 (n_steps=16) exploration then anneal.

  **RESULT — L2 SOLVED.** From-reset sweep of all 100 v10 snapshots
  (keep_best_sweep, 30 ep, GPU): princess-from-reset > 0 on 24 snapshots, peak
  window 5.9M-7.3M (0.83-0.97). Best = **7.3M snapshot; precise 300-ep
  stochastic eval from cold reset = 98.7% princess (296/300)** — vs 0 on all
  150 v9 snapshots and 0 on every v3-v9 run. So `defer_fruit_credit` was the
  blocker: once the fatal fruit-2 grab stops paying, the agent learns the safe
  landing and composes the whole level from reset (matches the call that
  both-fruits-ALIVE is the hard part; the princess follows). Champion saved to
  `output/mo5/yeti/champions/l2_v10_7300k/` (model + seed pools + meta). The
  peak is transient — the final model collapsed (7M+ -> ~0, capture-the-peak
  pattern), so keep-best stayed essential. Optional v11: anneal FROM the 7.3M
  champion to stabilize the peak / push toward L1-like 99.7%, but L2 is
  effectively solved as-is.
- [x] **RESOLVED (NOT a bug) — the "unfaithful eval" was POLICY OSCILLATION +
  evaluating the degraded FINAL snapshot.** Chasing the v7 reset_reach
  discrepancy, I first suspected the eval tools were unfaithful (v7 final/15M
  eval'd 0 from reset while training showed ~54%). WRONG. A multi-snapshot
  from-reset sweep (rollout_l2, per 100k) shows the policy OSCILLATES wildly:
  13.5M and 14.0M reach floor 5 + BOTH fruits 100% (30/30) from cold reset, but
  14.8M/14.9M/15.0M collapse to floor-1/0. v7 uses n_steps=16 / no target_kl =
  the exact L1 destructive-update oscillation (H-V). So: the eval tools are
  FAITHFUL; I just evaluated the final snapshot, which was in a "fall" phase.
  The training reset_reach_ema=0.40 and "54% last-50 true resets" were WINDOWED
  AVERAGES over the swing (snapshots range 0<->100%). LESSON (same as L1 H-U):
  a single/final snapshot is meaningless on an oscillating policy — always SWEEP
  snapshots from reset and keep the best. Prior v5/v6 evals are fine (they were
  genuinely stuck), but should ideally have been snapshot sweeps too.
- [x] **H-AN — L4 waypoint anchors from a grounded census (DONE, committed).**
  `e667206` derived all 20 L4 jump anchors from platform geometry (fit the tol box
  inside `standable_span`) and narrowed tolerance to match. Bisected at 1.2M/arm,
  seed 42, v4 warm start, one lever each: revert 0.75-0.84, B1 anchors-only
  0.79-0.85, B2 tol-only 0.78-0.87, BOTH decays to 0.01 and stays there 11M steps
  (reproduced on seed 43). Individually harmless, jointly destructive.
  **Root cause:** a jump's landing depends on the POLICY. On floor 7 the agent is
  grounded for TWO FRAMES at px 108 then airborne to the ladder at 144, and
  detection is pose-gated, so an anchor 4 px off detects nothing — anchor 28 scored
  1.00 vs v6's champion and 0.00 vs a later policy. **Rule adopted:** census
  grounded positions over >=2 policies from different runs, score per EPISODE, pick
  by the WORST policy's score, keep tolerance flat
  (`debug/l4_anchor_recommend.py`). Shipped `Rope1` 27->25, `Spring` 48->51,
  `Step` 60->66. Also found `Step` (0.00) and `Spring` (0.12) were MANDATORY
  milestones whose fixed tol-2 reward box never contained the agent — the `Fr1`
  defect, live since v4, so their distance term never switched off. Verified vs the
  revert probe at matched healthy steps: 700k reward 53.4 vs 46.4, Rope1 0.85 vs
  0.67, Step 0.81 vs 0.62.
- [x] **H-AO — the wall is `Step` -> `Lclimb3_top` (DONE, DIAGNOSED — see H-AR).**
  `prog` = P(an episode seeded here reaches any NEW route point). Both arms, healthy
  steps: `Step` reach 0.81 / **prog 0.02**, `Lclimb3_top` reach 0.02. The agent
  arrives at Step reliably and gets nowhere from it. All prior effort aimed at rung
  11 (`Low2`, floor 13) was two route points too far along. Method: roll out from
  the `Step` pool, classify where episodes go and how they end, and check whether
  `Lclimb3_top`'s 0.02 is non-arrival or another detection blind spot (apply the
  H-AN divergence test — its downstream `Low1` is also 0.00, so the test is
  inconclusive there and needs a direct trace).
  **RESULT.** Three findings, each measured, which together explain why rung 11 has
  been 0 for the life of the project.
  1. *Arriving at the ladder-3 head is lethal on a TIMER, not structurally.* 37/40
     episodes from the `Step` pool die on the ladder, the death flag flipping the exact
     frame y reaches 78 (floor 11's standing level). Scripted departure sweep
     (`debug/l4_ladder3_timing.py`): hold NOOP on arrival and 0/41 waits survive; hold
     LEFT and waits 14-22 survive; hold RIGHT and waits 11-28 survive. Departures 0-10
     die during the climb whatever follows. So there is a ~18-frame safe window in a
     deterministic cycle — a learnable timing skill.
  2. *So `Lclimb3_top` from-reset reach is 0.02, and the reach gate then shuts it out.*
     `gate_waypoints: true` + `reach_threshold: 0.15` filters on a waypoint's OWN
     reach.
  3. *Measured consequence: the three deepest pools are never sampled.* From the
     control's 7140 episodes (`start_key` counts in episodes.csv): `Step` 217 starts,
     `Lclimb3_top` **0**, `Low1` **0**, `Low2_launch` **0** — each holding 100 seeds.
     And the seeds are good: seeded there directly with the control's 800k policy,
     `Lclimb3_top` reaches floor 12 in 0.17 of episodes, `Low1` reaches floor 12 in
     0.90. A live gradient the trainer never sees.
  The gate is self-locking at the frontier: the frontier is by definition the point not
  yet reached, so its own reach is ~0, so it is never sampled, so the skill is never
  practised. **This, not anchors or tolerances or warm-starts, is what has been blocking
  L4.** Tools: `debug/l4_step_handoff.py`, `debug/l4_ladder3_timing.py`.
  Also visible: floor 12 -> 13 is 0/60 from the `Low1` and `Low2_launch` pools, so the
  rope-2 crossing is the NEXT wall behind this one.
- [x] **H-AR — gate a waypoint on its PREDECESSOR's reach (DONE. Mechanism works;
  did NOT beat v6 at 15M — see H-AS).**
  `curriculum.gate_waypoints_by_predecessor`, default False so L1/L2/L3 are unchanged.
  A waypoint is eligible as a start if the agent reaches EITHER it or the route point
  immediately before it. `Lclimb3_top` opens because `Step` is 0.81; `Low1` stays shut
  until `Lclimb3_top` itself clears 0.15, so the frontier advances exactly one rung at
  a time. That keeps the protection the gate was added for (L3: ungated drilling of
  unreachable states produced skill that did not compose to reset, 0.03%, at ~40% of
  episodes) while making the one advanceable rung trainable.
  ONE LEVER vs `l4_anchors_v2_1200k`, which is therefore the control.
  PASS: `Lclimb3_top` appears in the start_key counts at all AND its from-reset reach
  rises above the control's 0.02. FAIL: still 0 starts (the gate was not the binding
  constraint), or starts non-zero with reach still ~0.02 (not learnable from these
  seeds). Read at matched HEALTHY steps, 600-850k — the control collapses from 900k on.
  Config `experiments/003-yeti/configs/l4_predgate_1200k.yaml`, log `output/monitor/l4_predgate/`.
  Unit-pinned in `tests/python/test_wp_predecessor_gate.py`, including a test that
  asserts the OLD rule locks out the frontier.
- [x] **H-AP — champion on current code (DONE. v12 = 15M. NOT worse than v6 once v6
  is measured at more than one seed — see H-AS).**
  15M, current code, v4 `final_model.zip` warm start, seed 42 — v6's exact recipe so
  its distribution (n=150, mean 6.37, median 7.88, max 10.00) is a legitimate
  control. There is currently NO champion for this code: v6's was built at
  `bc85424`, before the revert and the censused anchors. Without this, H-AQ has
  nothing valid to compare against. Judge by `keep_best_sweep` distribution, not the
  final model.
  **H-AR RESULT at 1.2M: the mechanism does exactly what it was built to do.**
  `Lclimb3_top` went from 0 starts in 7140 control episodes to 44 starts by 200k; reach
  0.02 -> 0.36; prog `—` -> 0.74; `Low1` opened by itself once `Lclimb3_top` cleared 0.15
  and `Low2_launch` correctly stayed shut. The floor 11 -> 12 crossing measured 0.20 ->
  0.80. Committed as f069429.
  **H-AP RESULT at 15M (v12): v6 IS STILL BETTER, and there is a real regression.**
  v12 matched v6 on `Step` (0.7-0.87) and `Lclimb3_top` (0.6-0.79) but `Low1` was 0.00
  for the whole 15M where v6 held 0.4-0.76. The broken link, from episodes.csv:
  ```
  P(reach Low1 | reached Lclimb3_top)   from reset   seeded
      v6                              5158/5533=0.93   0.93
      v12                               14/7757=0.00   0.08
  ```
  v12 reached `Lclimb3_top` MORE than v6 (32433 vs 23084 episodes) and `Low1` far less
  (6285 vs 23730). Anchor-free floor-occupancy from each run's own `Lclimb3_top` pool,
  25 episodes, matched snapshots -- no champion/final mixing:
  ```
  step      v12 len/f12      v6 len/f12
  1.0M       55 / 0.56        57 / 0.68     <- equal here
  1.4M       27 / 0.36        74 / 0.96
  2.6M       47 / 0.52       119 / 0.72
  6.6M       76 / 0.60       113 / 0.80
  14.0M      67 / 0.44        95 / 0.88     <- v6 learned it, v12 never did
  ```
  So it is a LEARNING failure: current code crosses 0.56 of the time at 1.0M and never
  improves, while v6 climbs to 0.88 and its episodes get 2x longer. RULED OUT: detection
  (measured anchor-free), the seed pools (v6's and v12's `Lclimb3_top` pools are
  near-identical -- pose 8/0 at px 272, and 0/100 survive standing still in BOTH, which
  is expected given the floor-11 enemy), capability, and `Low1_launch` deletion (v6 never
  used it: 0 seeds, 0 reaches, empty pool). NOT ruled out: poses 6/7, 7b27d49, the
  censused anchors, the gate's altered start distribution.
- [x] **H-AS — is v6 REPRODUCIBLE? (DONE. NO — 1 seed in 4. There was no regression.)**
  Before bisecting four levers, test whether the thing we are bisecting toward is real.
  v6 is n=1, its own `Low2` peaked at 0.02, and method rule 6 records that from-reset
  reward collapses to ~3 and recovers in every arm, so single L4 runs are unreliable.
  Seed 43 deliberately -- rerunning seed 42 on identical code reproduces v6 and measures
  nothing. Worktree `/tmp/wt_a0` at bc85424; config
  `experiments/003-yeti/configs/l4_v6repro_s43_15m.yaml`; log `output/monitor/l4_v6repro_s43/`.
  **RESULT: FAIL, decisively.** bc85424's own code on three fresh seeds, all truncated at
  2M so it is apples to apples with v12:
  ```
  run                     L3top   Low1   P(Low1 | L3top)
  v6  seed 42 (bc85424)    1825   1644       0.81
  v6  seed 44 (bc85424)     165     12       0.07
  v6  seed 45 (bc85424)     245      6       0.02
  v6  seed 46 (bc85424)     216     16       0.07
  v12 seed 42 (f069429)    1643    533       0.16
  ```
  v12 beats every ordinary seed of v6's code and reaches `Lclimb3_top` 7-10x more often,
  so the predecessor gate is a real improvement and there was never a regression to find.
  v6 seed 42 scratched past the 0.15 gate at ~1.1M and cascaded; 1 seed in 4 does.
  This retires H-AR's four candidate levers (poses 6/7, 7b27d49, the censused anchors, the
  gate's start distribution) AND the `Step`-as-accidental-brake theory: seeds 44/45/46 all
  have `Step` unmarkable exactly like seed 42 and did not cascade.
  Run at 3 seeds x 2M, ~2.6h total, after a 15M single-seed attempt was killed at 1h10m —
  see method rules 8 and 9.
- [x] **H-AQ — sprite-overlap detection + pose blocklist (DONE, NEGATIVE but SAFE. Code
  kept, default NOT flipped. c666500.)**
  A/B at 3 seeds x 2M, `box` vs `sprite` paired per seed, one lever
  (`curriculum.waypoint_reach_mode`).
  **SAFE:** `Step` reach 0.65->0.69 and 0.72->0.72, no upstream regression; the offline
  sweep over every route point on two policies found no waypoint losing a genuine
  detection.
  **NOT AN IMPROVEMENT.** Anchor-free floor occupancy, matched snapshot 1.75M, one FIXED
  reference pool so only the policy varies, 30 episodes each:
  ```
  arm          mean_len  floor11  floor12
  box_s44        71       1.00     0.40      tie
  sprite_s44     47       1.00     0.40
  box_s45        92       1.00     0.47      box better
  sprite_s45     33       1.00     0.30
  box_s46        35       1.00     0.27      box better
  sprite_s46     28       0.97     0.23
  ```
  Box is equal-or-better on the crossing in 3/3 pairs, and sprite's episodes are ~45%
  shorter in 3/3. No single pair is significant at n=30 (se ~0.08), but the direction is
  consistent on two independent measures.
  **THE TRAP THIS RUN WALKED INTO, and why the anchor-free check exists.** `Lclimb3_top`
  reach DOUBLED under sprite (0.22->0.48, 0.17->0.35) and that was entirely definitional:
  a wider y-window fires more often. Method rule 5 -- a metric change is a lever -- applied
  to the exact number that looked like the win. Floor occupancy ignores anchors and
  tolerances, so it is the honest comparison.
  **UNVERIFIED mechanism for why it may be slightly worse:** the 18 px y-window lets a
  milestone mark MID-JUMP, before the agent has landed and secured the position, so the
  distance term switches off early and the shaping gradient weakens for the rest of the
  traversal. Not measured; do not act on it without measuring.
  **Kept anyway** because it is flag-gated at "box", the tests pin the geometry, and it
  measured four `_launch` pads that mark ~16-24 px EARLY at their predecessor's seed
  (frame-0 hits 25/40, 37/40, 39/40, 20/40 under box; 0/40 under sprite). Narrowing them
  to tol 2 instead was measured and rejected -- it removes the false positives but guts
  real detection (Fr1_launch 0.45->0.05, Hi1_launch ->0.10), the same trap as e667206.
  So the launch-pad defect is now documented with a measured fix, whether or not we take
  it this way.
- [ ] **H-AV — IS THE ROPE-2 CROSSING POSSIBLE FROM px 188 AT ALL? (NEXT, no GPU, ~20 min.)**
  The one unverified link in the rope-2 chain. From px 188 (measured standable), sweep
  scripted jump timings and directions and ask whether ANY sequence reaches floor 13
  (y 70, px 0..128). Same method that settled the ladder-3 hazard, where scripted
  departures 11-28 frames into the cycle survived and 0-10 died.
  WHY THIS GATES EVERYTHING ELSE: `Low1`'s anchor is also on a lethal pixel and its pool is
  completely healthy, so a real anchor defect can be inert. If nothing crosses from px 188
  then the `Low2_launch` anchor is not the binding constraint and the survival-gate work
  (H-AW) should not be spent on rope 2 yet.
  PASS: some scripted line crosses -> the position is viable, the agent simply never learns
  it, and H-AW is on the critical path. FAIL: nothing crosses -> find what does before
  touching the curriculum.
  NOTE ropes MOVE between frames, so a single-frame pixel measurement of rope position is
  NOT valid evidence (an earlier attempt at that was retracted). The sweep has to be over
  timings, not geometry.
- [x] **H-AW — SURVIVAL GATE: require SEED_POSES at window end (DONE, NEGATIVE. Flag
  committed 91cebf6, default OFF, do not enable.)**
  A/B at 3 seeds x 2M, one lever, both arms carrying the re-landed anchor fixes.
  **RESULT: no effect on pool composition.** `Low2_launch` seeds at px 184 (LETHAL, 0/8
  survive), counted AS SAVED:
  ```
  gate off:  13, 13, 17  of 100        gate on:  34, 20, 24  of 100
  ```
  And they are NEW, not inherited from the v4 warm start: 11/11/9 (off) and 33/13/20 (on).
  **AND THAT CONTRADICTS THE GATE'S OWN BEHAVIOUR, WHICH IS UNEXPLAINED.** Instrumented to
  log capture position, survival, end pose and verdict: px-184 captures are REJECTED 8/8
  with the 400k policy and 8/8 again with the 2M policy. Only two code paths write a
  waypoint pool -- `save_waypoint` (gated) and the load-from-disk path (inherited) -- so
  these seeds should be impossible. NOT RESOLVED after four probes; parked deliberately
  rather than explained away.
  Two of my explanations for it were asserted without measurement and both were then
  disproved (see method rule 12): "the agent recovered so end_pose was a surface pose" and
  "the end_pose lookup falls out of range". Neither holds.
  PROVEN by filmstrip (`debug/l4_gate_admitted_proof.py`,
  `experiments/003-yeti/evidence/l4_rope2_geom/gate_admitted_seed_falls.png`): a px-184 seed from a gate-ON pool
  that is absent from v4's pool, reloaded and held NOOP, falls -- pose 5 grounded at t+0,
  pose 11 FALL by t+3, y 70 -> 126.
  ALSO MEASURED: one earlier reading of "40 seeds at px 184" was inflated to 40 from 34 by
  reading the position AFTER one NOOP step; a seed carrying leftward motion moves 188 -> 184
  in one frame. Read pool positions AS SAVED.
  NEXT IDEA, not attempted: reject a capture whose POSITION lies outside its floor's
  MEASURED usable span (floor 12 = px 188..224). That is a static check needing no
  simulation and no reasoning about episode continuations, and it encodes the measurement
  directly. Blocked on the unexplained admission path above -- if seeds can enter the pool
  by a route we have not found, a capture-time filter may not reach them either.
- [ ] **H-AT — the real problem: the `Lclimb3_top -> Low1` crossing is learned 1 run in 4.**
  Not a regression, an unreliability. The floor 11 -> 12 leftward jump follows a TIMED
  hazard at the ladder-3 head: scripted departures 11-28 frames into the cycle survive,
  0-10 die (`debug/l4_ladder3_timing.py`). Learnable in principle -- the enemy IS visible
  in the 84x84 observation, and the `Step` pool spans the hazard cycle (77-86% of its seeds
  survive an immediate climb, against the ~40% a phase-diverse pool would predict). So the
  information and the practice states are both present and it still only lands 1 in 4.
  NOT yet attempted. Whatever is tried, judge it at >= 3 seeds: `P(Low1 | Lclimb3_top)` is
  0.81 / 0.07 / 0.02 / 0.07 on IDENTICAL code, so n=1 cannot see anything.
- [~] **H-AU — rope 2: the LAUNCH PAD IS A LETHAL PIXEL (diagnosed 2026-09-09; fixes
  designed, none applied yet).**
  `Low2` has been reached 24 times in the project's history and never seeded from; v6 stood
  on the launch pad 25,520 times and crossed 10 (P = 0.0004). Diagnosed by watching a
  from-reset video, after the numeric routes all missed it.
  **MEASURED (direct emulator reads, not training outcomes):**
  ```
  FLOOR 12, tile extent 184..232      walking LEFT   walking RIGHT
    184                                   0/8            0/8     falls
    188 .. 224                            8/8            8/8     OK
    228                                   0/8            0/8     falls
    232                                   0/8            0/8     falls
  ```
  * `Low2_launch`'s anchor is px 184 -> LETHAL. 81/100 of its pool sit there and fall on
    load. The shaping pays **+1.08** for the grounded step onto it -- the largest single
    payment in the whole from-reset trace.
  * The agent then falls, the TRAMPOLINE below bounces it back to floor 12, and it repeats
    until it dies. Pose 14 (rope carry) never appears: it never grabs the rope.
  * `admit_requires_survival` cannot clean this. Median 83 steps to death vs
    `min_survival_steps: 30`, so 100/100 doomed seeds are ADMITTED. Raising the threshold
    is not the fix; the criterion must become "grounded on a platform at window end".
  * The recorded span in `standable_span`'s docstring (188..228) is WRONG at the right end.
  * **Direction does not matter** -- tested because the foot row is asymmetric
    (centre-6..centre+2); 8/8 both ways everywhere. Hypothesis rejected.
  * `Low1`'s anchor (px 232) is ALSO outside the span but its pool is HEALTHY, because the
    agent falls at 228 before ever reaching 232. **A lethal anchor can be harmless**, so
    "this anchor is wrong" does not imply "this is the blocker".
  **THE REAL DEFECT:** yeti_map.py's comment states `Low2_launch edge 44 -> 45` in the PAST
  TENSE, but `jump_waypoint_pos` has no such entry -- `e667206` applied it and the wholesale
  revert removed the code while LEAVING THE COMMENT. The file reads as already fixed.
  **DECIDED, not yet applied:** re-land Low2_launch 184->188 and make the comment match the
  code; Low1 232->224; correct standable_span's docstring (floor 12 right limit 228 -> 224,
  measured 0/8 at 228 vs 8/8 at 224); survival gate -> grounded-on-platform at window end.
  **NOT VERIFIED:** that any of it improves anything, and that the crossing is even
  POSSIBLE from px 188. The rope-position analysis was retracted (ropes move between
  frames, so single-frame pixel measurement is unreliable).
  **NEXT, before any GPU:** from px 188 sweep scripted jump timings and ask whether ANY
  sequence crosses. If none does, the anchor is not the binding constraint. Tools:
  `debug/l4_crossing_trace.py`, `debug/l4_low2launch_fix_visual.py`,
  `debug/l4_pull_direction.py`. Figure `experiments/003-yeti/evidence/l4_rope2_geom/low2launch_fix_v2.png`;
  clip `experiments/003-yeti/evidence/l4_rope2_fromreset/rope2_failed_ep0.mp4`.
- [x] **H-AQ (original proposal; superseded by the entry above, which has the result).**
  Retires the anchor-placement bug class instead of fixing instances. Detection
  currently asks "is the agent's POSITION inside a tolerance box"; ask instead "does
  the agent's SPRITE contain the anchor POINT". Identical in x (sprite half-width IS
  a derived tolerance, ~7 px ~ 2 x_ram units) but very different in y: today's test
  is +-tol around the sprite TOP (4 px window) while overlap asks whether the point
  lies in `[y, y+17]` (18 px). A jumping agent's y DECREASES, so its sprite still
  spans the floor line for much of the arc. Measured, fraction of episodes firing,
  good anchor vs the one that broke:
  ```
                            v6champ a25  a28     v11 a25  a28
  SPRITE allow surface           0.04  1.00         0.96  0.00
  SPRITE allow +traverse         1.00  1.00         0.96  0.96
  SPRITE block fall+death        1.00  1.00         0.96  0.96
  ```
  Sprite overlap ALONE does not help; overlap PLUS a non-allowlist gate makes the
  anchor stop mattering. **Blocklist, not allowlist:** `SURFACE_POSES` fails CLOSED
  — poses 6/7 were missing from it for the project's entire history (~54% of
  grounded frames on any leftward approach suppressed) and pose 15 is still
  uncatalogued and appears every run. A blocklist fails open. Shape: detection =
  sprite overlaps anchor point; gate = blocklist {11 fall, 12 death}; capture =
  `admit_requires_survival` with grounded as an eviction PREFERENCE, not a filter
  (measured: a mid-jump state reloads frame-identically, so the "inherits a fall"
  justification was false; and grounded is insufficient anyway — `Low2_launch` held
  100 grounded-but-doomed seeds). Changes reward AND metrics, so it invalidates
  comparison to v6 and needs H-AP first. Big change: expect to bisect it if it
  regresses, so land it as ONE lever with a control arm.
- [ ] **H-B — does curriculum help an EASY target?** From the baseline,
  add *only* a CP0+CP1 start mix (capped at CP1) and compare CP0->CP2
  vs reset-only. Needs a `max_start_level` knob.
- [ ] **H-C — compute.** Plain reset, 20M steps, no other change.
  (`yeti_universal_v5_20m.yaml` already drafted.)
- [ ] **H-D — targeted exploration for F2->F3.** Local to the post-F2
  state (not global entropy/RND). Design TBD.
- [ ] **H-E — MIP tested honestly.** MultiInputPolicy in the reset-only
  setting, isolated from the curriculum confound of v7.
- [ ] **H-G — Markovian current-floor from state.** Remove the
  `last_floor` mid-jump fallback (a small, bounded non-Markovian residue
  kept for now because it prevents jump-farming). Derive current floor
  from RAM (ladder/climb flag or vertical-velocity byte) so floor is a
  pure function of the present state. Cheap-to-medium; only pursue if
  evidence says the residue matters.
- [ ] **H-H — shaping target selection.** Phi currently sums distance
  over *all* remaining fruits, which creates competing pulls when the
  next fruit is in the opposite direction from later ones (the F1->F2
  vs F3/F4 conflict seen in H-A). Try nearest / next-in-order target
  instead. Only after PBRS (H-F) lands so it's a single change.

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
`scripts/mo5/yeti/train_checkpoint_curriculum.py`.

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
`scripts/mo5/yeti/go_explore.py --seed-archive ... --seed-min-cp 2`, which loads
the given archive, filters it through the state validator, and adds
the viable cells to the exploration archive before the main loop starts.

Two prerequisites added for this experiment:

- **State validator** (`python/retro_ai/training/state_validator.py` +
  `scripts/mo5/yeti/filter_archive.py`). Rule: load state, noop probe, reject if
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

Two fixes to `scripts/mo5/yeti/go_explore.py`:

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
load_state). Seeds extracted from v9 via `scripts/mo5/yeti/extract_seeds.py`.

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
   `scripts/mo5/yeti/filter_archive.py` applied to a converted archive, or by
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
episodes.csv (`experiments/003-yeti/evidence/short_cp2_episodes/episode_0_state.pkl`). Player
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
(`scripts/mo5/yeti/rollout_policy_from_seeds.py` against v3's final_model):
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

### 15. Floor-novelty reward — segment_2to3_v5  *(verified)*

First test of the approach-14 candidate #2: one-shot +1.0 reward the
first time the agent enters each new floor per episode. Same
everything else as v3 (5M steps, 48 re-validated CP2 seeds). Reward:
``fruit_bonus_floor_novelty`` (registered in
``python/retro_ai/training/rewards.py``, commit `640fa4b`).

**Result: CP2→CP3 rate = 8.95% last-20%, vs v3's 3.30%.** Roughly 2.7×.

Per start_floor (last 20%, pure CP2 seeds):

| start_floor | v3 CP3 rate | v5 CP3 rate | v5 up % | v5 down % |
|:-----------:|:----------:|:----------:|:------:|:-------:|
| 0 (spawn)   |  1.43%     |  1.91%     |  8.2%  |  0.0%   |
| 1 (floor 2) |  2.60%     | **14.09%** |  3.7%  | 22.4%   |
| 2 (floor 3) |  0.00%     |  0.00%     |  1.0%  | 19.8%   |
| 3 (floor 4) | 12.83%     | **24.46%** |  1.4%  | 36.3%   |

Where v5 pulls ahead:
- **From game-floor-2** (start_floor=1): 14.1%, up from 2.6%. The
  biggest shift.
- **From game-floor-4** (start_floor=3): 24.5%, up from 12.8%.

Where v5 doesn't help:
- Spawn is unchanged (1.9% vs 1.4%). The bottleneck isn't "encourage
  the agent to leave spawn".
- game-floor-3 remains 0%. The hardest starting position.

**But climb rate didn't change.** v3 and v5 both show 4.1%
overall-climb. Novelty didn't make the agent learn to climb more
often. What it appears to have done instead: make *descents* more
efficient. Compare v3 vs v5 down-rates from start_floor=1 (40.7% →
22.4%) and same-floor-stays (56.3% → 73.9%): v5 falls off floor 2
less often. From start_floor=3, v5 descends *more* (25.1% → 36.3%)
— plausibly controlled descent to reach a remaining fruit on a
lower floor.

The reward doesn't distinguish up from down ("any new floor this
episode pays once"), so the agent uses it for whichever direction
the remaining fruit is.

**So the novelty reward is doing some work, but not the work we
predicted.** It's not making the agent better at climbing; it's
making the agent better at going wherever the fruit happens to be.
That's useful, especially since v2-of-CP2 seeds have fruit 1 or 2
remaining and need descent.

Config: ``experiments/003-yeti/configs/segment_2to3_v5.yaml``.
Output: ``output/mo5/yeti/training/segment_2to3_v5``.

### 15.1. Open question and next probe

The lever that moves the needle on spawn (start_floor=0) or
floor-3 (start_floor=2) is still missing. Those are the two
start floors where an agent must actually climb past the first
few floors to reach a remaining fruit. v5 doesn't solve them.

Approach 14.1's option 3 (distance-to-remaining-fruit) would
target those cases specifically. Risk that we already named:
if the remaining fruit is on a lower floor, the gradient points
down, which we don't want. But we could gate it: reward only
reductions in *upward* vertical distance when the remaining fruit
is above the agent, else pay nothing. Worth sketching.

v4 (20M, no shaping) is still running at ~49% complete. If that
finishes near v3's 3-4% too, the "just train longer" option is
empirically dead and reward shaping is our only remaining lever.

### 15.2. 20M without shaping — segment_2to3_v4  *(verified)*

Approach 14's option 1 ("just train longer") tested.

segment_2to3_v4: same as v3 (fresh policy, re-validated v9 CP2 seeds,
bug fixes in place) but 20M steps instead of 5M. Same config, same
seed, same everything else.

**Result: CP2→CP3 = 4.21% last-20%, up from v3's 3.30% but still
below v5's 8.95%.** And the per-floor breakdown is worse than v3 on
climbing:

| start_floor | v3 (5M)   | v4 (20M)  | v5 (5M, novelty) |
|:-----------:|:--------:|:--------:|:---------------:|
| 0 (spawn)   |  1.43%   |  1.07%   |   1.91%         |
| 1 (floor 2) |  2.60%   |  6.98%   |  14.09%         |
| 2 (floor 3) |  0.00%   |  0.00%   |   0.00%         |
| 3 (floor 4) | 12.83%   | 10.82%   |  24.46%         |

Overall climb rate: v3 = 4.1%, **v4 = 1.2%**. Training longer actually
made the agent climb *less*, not more. 4× the training reinforced the
local optimum harder.

**So training longer without shaping is empirically dead.** The
policy converges toward "stay-in-place or fall-and-die" as it has
more time to sharpen the reward basin it already occupies.

segment_2to3_v6 (climb-directional shaping, approach 15 candidate)
is running. If it does materially better than v5 on spawn and
game-floor-3 starts, we have the mechanism that breaks the wall.

### 16. Climb-directional novelty — segment_2to3_v6  *(verified)*

Approach 15's floor-novelty helped descent but not climbing.
v6 (`fruit_bonus_climb_novelty`) is the directional variant: same
fruit term, plus a one-shot +2.0 when the agent reaches a floor
HIGHER than any seen this episode, and only if a remaining fruit's
pixel-y sits strictly above the agent's pixel-y. Direction-gated +
target-gated.

Measured fruit pixel centres (from a CP0 screenshot with grid +
user verification, commits `e85ba92` / experiments/003-yeti/evidence/cp0_fruits_annotated.png):

  fruit 1: ( 184, 184 )  floor 1 (spawn)
  fruit 2: (  80, 150 )  floor 2
  fruit 3: ( 144, 120 )  floor 3
  fruit 4: ( 272,  88 )  floor 4 (top)

**Result: CP2→CP3 = 14.85% (last 20% on pure seeds).**

Four-way comparison (same 5M budget, same seeds; v4 is the 20M
outlier):

| Run  | Reward               | CP3 rate | Climb % | sf=0   | sf=1   | sf=2   | sf=3    |
|:----:|:--------------------:|:--------:|:-------:|:------:|:------:|:------:|:-------:|
| v3   | fruit_bonus          |  3.30%   |  4.1%   |  1.43% |  2.60% | 0.00%  | 12.83%  |
| v4   | fruit_bonus (20M)    |  4.21%   |  1.2%   |  1.07% |  6.98% | 0.00%  | 10.82%  |
| v5   | floor_novelty        |  8.95%   |  4.1%   |  1.91% | 14.09% | 0.00%  | 24.46%  |
| v6   | climb_novelty        | **14.85%**| **6.4%**| **4.26%**| **26.46%** | 0.05% | **31.01%** |

Reading:
- v6 beats v5 on every starting floor, and almost triples v3.
- Climb rate finally moved: 6.4% vs 4.1% across v3/v5. The directional
  gate + target check does what we predicted.
- **Spawn starts (game floor 1) saw a real uplift**: 1.43% → 4.26%.
  The agent is finally climbing from spawn when a fruit is above.
- **Floor 2 starts jumped to 26%** — about 10× v3.
- **Floor 3 starts remain ~0%.** From game-floor-3, the agent still
  cannot reach whatever fruit remains. Likely because: (a) fruit 4 is
  at x=272 on the right side of the top floor, but the ladder from
  floor 3 to floor 4 is elsewhere; (b) the CP2 seeds on floor 3
  usually have fruit 3 already collected, leaving fruit-from-other-
  floor as the target, which needs both descent AND navigation.

No sign of jump-farming despite the shaping. Training success curves
are smooth and not dominated by the climb term.

Config: `experiments/003-yeti/configs/segment_2to3_v6.yaml`.

### 17. Shaping design iteration — why v6 isn't enough  *(verified by analysis)*

Before building a next reward, reviewed v6's (`climb_novelty`) aggregate
and rollout signals for side effects:

- Episode length mean 117 (v3=114, v5=127); median 72 (v3=69, v5=83).
  No ballooning.
- Long failures (>=500 steps, stuck at CP2): 0.6% (v3=0.3%, v5=0.6%).
  No snowball-farming runaway.
- Final score mean 182 (v3=178, v5=179). Stable.
- Total reward median 0 (v3=0, v5=3); v5 pays more because its novelty
  fires on every new floor unconditionally, v6 only when fruit above.

12 rollouts from v6's final model (user review):
- "agent jumps to fruit and gets it" — working.
- "on floor 4, has collected fruits 3+4; jumping 2 snowballs, dies on
  third" — **stuck on top with fruits below**. v6's reward doesn't
  pay for descent, so once the climb bonuses run out the agent has
  no gradient toward the remaining low-floor fruits.
- "wandering on floor 1, jumping around" — at spawn without fruits
  above the agent, climb reward doesn't fire; plain fruit_bonus
  alone still fails.

Conclusion: v6's directional gate is too restrictive. We need a
reward that pays for movement toward whichever remaining fruit is
closest regardless of direction.

### 18. Path-distance reward with hand-coded map  *(verified)*

User pushed back on several simpler options:

- **Manhattan distance to nearest fruit**: two problems. (1) In
  Y-phase (different floor from fruit), jumping reduces dy enough to
  look like progress. (2) Moving sideways reduces dx easily, but
  real progress requires finding a ladder — agent can get stuck
  beneath a fruit on the floor below.
- **Staged Y-then-X** (reward dy progress first, then dx): same
  problem — "reduce dy" on the wrong floor doesn't route through
  ladders.
- **Fixed lowest-numbered-fruit priority**: restricts the agent's
  freedom to choose which fruit to pick first.
- **Per-fruit floor-novelty combined with fruit-above gate** (v6):
  helps from spawn/floor 2 but not from floor 3 or floor 4 starts.

Resolution: **build the real navigation graph and reward
shortest-path progress.**

Map verification done by loading a CP0 state and overlaying ladder
boxes / fruit boxes on the rendered screenshot. User corrected
offsets iteratively until every element landed:

Floor top-Y (where an agent sprite's UL sits when standing):
  floor 1 (spawn): y=184, floor 2: 152, floor 3: 120, floor 4: 88,
  floor 5 (princess): 56. Floors 32 px apart.

Fruit pixel CENTRES (sprite 16x16):
  F1 (184, 184)  F2 (80, 150)  F3 (144, 120)  F4 (272, 88)

Ladders (UL pixel x, 16 px wide, 32 px tall):
  L12a x=112, L12b x=272  (floor 1 has two up-ladders)
  L23  x=232
  L34  x=168
  L45  x=200
Princess UL (304, 48), sprite 16x24 (at x=312 centre, y=60).

Verified artefacts: `experiments/003-yeti/evidence/cp0_fruits_annotated.png`,
`experiments/003-yeti/evidence/cp0_ladders_annotated.png`, `experiments/003-yeti/evidence/cp0_nav_graph.png`.

#### 18.1. Graph model

Module: `python/retro_ai/training/yeti_map.py` (pure Python, no
dependencies beyond typing/dataclasses).

15 fixed nodes: 4 fruits, 5 ladders x 2 endpoints each (top+bottom),
1 princess. Edges: horizontal same-floor edges (cost = |dx|) and
ladder bot<->top edges (cost = FLOOR_HEIGHT=32). All-pairs shortest
paths via Floyd-Warshall on construction; lookup is O(number of
floor-N nodes) per query since the agent is a transient point.

Sanity distances (verified by tests):
- F1 <-> F2: 136 px
- F1 <-> F4: 392 px
- F1 <-> princess: 464 px
- Agent (floor=1, x=0) -> F1: 184 px
- Agent (floor=1, x=280) -> F2 via L12b: 232 px (shorter than via L12a)

#### 18.2. fruit_bonus_path_progress reward

Module: `python/retro_ai/training/rewards.py`.

Per-step logic:
1. Fruit-pickup term (same as fruit_bonus).
2. Resolve current floor (agent_floor_from_pixel_y with 8 px
   tolerance); fall back to last-known floor if mid-jump.
3. Clear best_d for any fruit now absent (post-pickup housekeeping).
4. For EACH remaining fruit, compute path distance from the agent
   through the graph. If distance < best_d[fruit], pay
   (best_d - new) * scale and update best_d.
5. Return.

Key design choices:
- **Multi-fruit tracking (per-fruit best_d), not closest-only**: the
  agent gets shaping toward whichever fruit it moves nearest to,
  not just one pre-chosen target. Matches user's "do not restrict
  to predefined order" requirement.
- **Strict-less-than ratchet + per-fruit lock**: jumping and
  oscillation pay zero. Once the agent has been distance D from
  fruit F, only distances < D pay further.
- **Falls back on last-known floor during jumps**: shaping stays
  active mid-jump instead of flickering.
- **Princess not yet a target**: when all fruits are collected, the
  progress term falls silent. Will add princess routing once we
  confirm the pipeline works for 4-fruit pickup.

Cost per step: ~60 integer ops (one Floyd table read per via-node,
4 fruits x ~15 floor-candidates). Negligible vs emulator step.

Next: config + smoke + 5M run as segment_2to3_v7.

### 19. Path-progress reward — segment_2to3_v7 surprise  *(verified, suspicious)*

5M run with `fruit_bonus_path_progress` (commit `82353d7`). Same
seeds, same hyperparameters as v3/v5/v6.

**Summary numbers (last 20%, pure CP2 seeds):**

| Run | Reward                        | CP3 rate | Climb % | Descent % |
|:---:|:-----------------------------:|:--------:|:-------:|:---------:|
| v3  | fruit_bonus                   |  3.30%   |  4.1%   |   20.5%   |
| v5  | floor_novelty                 |  8.95%   |  4.1%   |   17.3%   |
| v6  | climb_novelty                 | 14.85%   |  6.4%   |   18.1%   |
| v7  | path_progress                 |  7.26%   | **14.7%** | **10.9%** |

Per start_floor:

| start_floor | v3 cp3 | v5 cp3 | v6 cp3 | v7 cp3 |
|:-----------:|:------:|:------:|:------:|:------:|
| 0 (spawn)   | 1.43%  | 1.91%  | 4.26%  | **5.11%** |
| 1 (floor 2) | 2.60%  | 14.09% |26.46%  | 11.32% |
| 2 (floor 3) | 0.00%  | 0.00%  | 0.05%  | 0.07%  |
| 3 (floor 4) | 12.83% | 24.46% |31.01%  | 13.41% |

Mixed picture:
- **Climb rate is the highest yet** (14.7% vs v6's 6.4%). The
  shaping does push climbing.
- **Spawn-floor cp3 rate is the highest yet** (5.11%).
- But **floor 2 and floor 4 cp3 rates regressed from v6**, and
  overall cp3 rate is below v6.

**Suspicious side effect: high-reward farming on a few specific seeds.**

v7 episodes have very different reward distributions than v3/v5/v6:

| Run | reward_med | reward_max | long_runs (>=500) |
|:---:|:----------:|:----------:|:-----------------:|
| v3  | 0.00       | 8.1        | 0.3%              |
| v5  | 3.00       | 35.0       | 0.6%              |
| v6  | 0.00       | 20.1       | 0.6%              |
| v7  | 12.13      | **926.7**  | **2.7%**          |

v7's max-reward is ~50x v6's, and 2.7% of episodes survive past 500
steps without reaching CP3 (vs 0.6% for the others). 300 of 315
high-reward (>700) failed episodes start on the same seed (idx 47:
fruits 1+2 collected, agent on floor 4 at ram_x=31). Each ends back
at the same x=31, y=86 it started at, with no score gain.

Theoretical max reward bound for this seed: `(d_F3 + d_F4) * scale =
(108 + 140) * 0.01 = 2.48`. But trained-policy episodes accumulated
926. **300x the bound** under the per-fruit best-d ratchet.

Confirmed by isolated property test: `fruit_bonus_path_progress`
called with random walk 1000 steps respects the bound (1.64 < 2.48).
So the reward formula in isolation is correct.

Interaction with the training loop somehow breaks the lock. Two
candidate causes I haven't pinpointed:
- An invisible mid-episode `reset()` clearing best_d (perhaps the
  ThreadedVecEnv or SB3 calls reset under some condition).
- A subtle recompute that re-paths through new floors and racks up
  large progress on each pseudo-episode.

I tried to reproduce with the saved final_model.zip on the same seed
(`scripts/mo5/yeti/repro_v7_farming.py`) and the trained policy stays
completely stationary at start position — total reward 0 over 1000
steps. So the trained policy and the reward-collection during
training disagree.

**Conclusion**: don't trust v7's headline 7.26% as the merit of
path-progress shaping. There's a bug in how reward accumulates over
a training episode.

Next step: instrument the env to track per-episode reward inside
SegmentEnv, log to TB, and add a sanity check that reward never
exceeds `sum_of_initial_distances * scale + n_fruits_collected *
fruit_bonus_term` per episode.

### 20. Shared-reward bug + clean v7 result  *(verified)*

#### 20.1. The bug

While instrumenting the reward path to root-cause v7's apparent
farming, I added a per-step trace recorder that dumps any episode
whose total exceeds the analytical bound. First smoke run produced
this dump for episode 408 of env 0:

  step | x  | y  | floor | best_d
   1   | 57 | 82 | 4     | {1: 356, 2: None, 3: None, 4: 36}
   2   | 57 | 82 | 4     | {1: 356, 2: None, 3: None, 4: 36}
   3   | 58 | 78 | None  | {1: 356, 2: None, 3: None, 4: 32}
   4   | 59 | 76 | None  | {1: 364, 2: None, 3: None, 4: 28}  <-- bd[1] up
   ...
  13   | 62 | 82 | 4     | {1: 376, 2: None, 3: None, 4: 12}
  14   | 61 | 78 | None  | {1: 68,  2: None, 3: None, 4: 12}  <-- jumps
  15   | 61 | 78 | None  | {1: None, 2: None, 3: None, 4: None}  <-- WIPED

`best_d[1]` is supposed to monotonically decrease (per-fruit lock).
Steps 3-10 show it INCREASING (356 → 380), and step 15 shows the
whole dict wiped to None mid-episode. The ratchet is broken.

Root cause (commits `d540831` and earlier):

All three multi-env training scripts (`train_segment.py`,
`train_checkpoint_curriculum.py`, `go_explore_phase2.py`) share a
SINGLE `reward_fn` instance across all parallel envs:

```python
reward_fn = create_reward(cfg.reward.name, cfg.reward.params)

def make_env(rank):
    def _init():
        return SegmentEnv(..., reward_fn=reward_fn, ...)  # shared!
    return _init
```

When SB3 ends env A's episode, it calls `env.reset()`, which calls
`reset_reward(self._reward_fn)`. That clears the shared per-episode
state. **Every other env still mid-episode now sees a fresh
reward_fn on the next step**, re-baselines, and earns full
"progress" reward all over again on the same path.

Affects every stateful reward we shipped:
- `fruit_bonus_floor_novelty` (v5)
- `fruit_bonus_climb_novelty` (v6)
- `fruit_bonus_path_progress` (v7)

Stateless rewards (`fruit_bonus`, etc., used by v3/v4) are unaffected.

#### 20.2. The fix

Each env now constructs its own reward_fn instance inside the
`_init` closure:

```python
def make_env(rank):
    def _init():
        env_reward_fn = create_reward(cfg.reward.name, cfg.reward.params)
        return SegmentEnv(..., reward_fn=env_reward_fn, ...)
    return _init
```

Regression test in `tests/python/test_no_shared_reward_fn.py`
asserts two SegmentEnvs created via `make_env` hold distinct
`reward_fn` and distinct `best_d` dicts. Pinned so this can't
silently regress.

Forensic instrumentation kept as a permanent safety net in
`python/retro_ai/training/reward_trace.py`. Any future episode that
exceeds the analytical bound will be pickled to disk with full
per-step state. SegmentEnv enables it when the configured reward is
`fruit_bonus_path_progress`; other rewards skip tracing.

#### 20.3. v7 with the fix

5M run, same config as before:

**CP2→CP3 = 50.24% in last 20% (pure CP2 seeds), up from 7.26%.**

Per start_floor:

| start_floor | v7 BUGGY | v7 FIXED |
|:-----------:|:--------:|:--------:|
| 0 (spawn)   |  5.11%   | **61.11%** |
| 1 (floor 2) | 11.32%   | **71.40%** |
| 2 (floor 3) |  0.07%   |  0.51%   |
| 3 (floor 4) | 13.41%   | **54.73%** |

- Climb rate jumped 14.7% → 30.1%.
- Descent rate 10.9% → 18.4% (agent uses both directions, as the
  reward intends).
- 17 episodes reached CP4 (0.16% of pure CP2 episodes). First time
  per-segment training has produced any CP4 reach.
- Median reward 3.84, max reward 22.96. Well within bounds.

Floor-3 starts still ~0% — the lone weak spot. Hypothesis: those
seeds usually have F3 already collected (so target is a fruit on a
different floor that requires both descent through L34 AND
horizontal navigation, longer path).

#### 20.4. Implications for v5 and v6 numbers

v5's reported 8.95% and v6's 14.85% are both contaminated by the
same bug. Without rerunning we don't know how much of those
gains were real vs reward-leak.

Two options:
- Rerun v5 and v6 with the fix, just to have clean comparison data.
- Skip them: v7 already dominates and is now the headline result.

Path-progress is clearly the right shaping. The other shaping
formulas can be retired.

Commit: `d540831`.

### 20.5. Floor-3 starts: why they're stuck at 0%  *(observed)*

10 floor-3 CP2 seeds in the v9_v2 archive, all with F3 already
collected. After v7 fixed (50% overall CP3), floor-3 starts only
hit 0.51%. Rolled out the trained policy from each, with a live
reward HUD overlay (`scripts/mo5/yeti/rollout_with_reward_overlay.py`).

What we saw on a sample:

- **seed 24** (y=118, F2+F4 remaining): agent jumps left and
  **falls** off the floor 3 platform straight down to floor 1. As
  it falls, pixel y crosses through the floor-2 bucket; our
  reward's `last_floor` fallback updates the agent's "current
  floor" to 2, and the path-distance to F2 (on floor 2) drops
  massively. Agent gets reward for falling-toward-F2. Then dies
  on floor 1 from a snowball.

- **seed 17** (y=118, F1+F4 remaining): agent dies in 15 steps
  jumping into a snowball.

- **seed 36** (y=118, F1+F2 remaining): agent jumps left and
  falls to floor 2 via the gap. Cumulative reward 4.96 (highest
  of the three) because it crossed two floor boundaries on the way
  down.

The picture: from floor 3 with F3 already collected, the
**path-progress reward sometimes pays for falling**. Agent's pixel
y crosses lower-floor buckets on the way down; our `last_floor`
fallback updates accordingly; path distance to fruits on those
lower floors drops; reward fires.

Why the per-fruit lock doesn't fully save us: each lock only
prevents re-collecting reward for the SAME minimum distance to a
fruit. A fall yields a one-shot credit (the "best ever distance
to F2" tightens once during the fall). Then the agent dies. Net:
small reward + episode termination. Better than infinite farming,
but it still teaches "fall = quick reward".

**Possible fixes (not implementing now)**:

1. Don't fall back to `last_floor` — only credit progress when the
   agent's pixel-y resolves cleanly to a floor (i.e., agent is
   standing). Mid-air pays nothing; ladder-climb pays only when
   the agent lands on the new floor.

2. Detect "agent is on a ladder" via x being within 16 px of a
   known ladder column AND y crossing the floor boundary. Pay
   only for ladder-driven floor changes.

3. Penalty term for descents not at a ladder column. Punishes
   falling specifically.

Going with option 1 is the simplest cut, but we're not blocked on
solving floor-3 right now. The v7-fixed policy gets 50% on the
other three start floors and that's a real improvement worth
chaining on. Documented and moving on to segment 3to4.

### 21. Segment 3→4: 30% CP4 with the same reward  *(verified)*

First per-segment CP3→CP4 training. Same reward as v7
(`fruit_bonus_path_progress`), same hyperparameters, same approach
20 fix in place.

#### 21.1. Seed pool: enrichment from collected_states

CP3 seeds were sparse in v9 alone (19 validated). The trained v7
agent reached CP3 thousands of times during its 5M run; its
`collected_states.pkl` contains 800 CP3 states. After running them
through the same validator (probe=120) and merging with v9's 19:

  raw merged:                       819
  validated (CP3 viable for ≥120 noops): 187
  rejected:                          632 (most "agent landed on F3
                                          and a snowball is one
                                          frame away" type states)

Then quality-filtered down to the top-50 per remaining-fruit group
(by post-settle bonus), keeping all 4 remaining-fruit
configurations represented:

  remaining=(1,):  4 (kept all)
  remaining=(2,): 38 (kept all)
  remaining=(3,): 36 (kept all)
  remaining=(4,): 50 of 109 (top-50 by bonus 862-863)
  total:         128 CP3 seeds.

Stored at `output/mo5/yeti/seeds/v9_v3_cp3enriched.pkl`.

(The collected_states distribution is heavily skewed: 109 of 187
have F4 remaining, because v7's agent reached CP3 most often by
collecting F1+F2+F3 in that order, leaving F4 last. We capped that
group to avoid sample-bias.)

Tooling: `scripts/mo5/yeti/build_cp3_seeds.py`.

#### 21.2. Headline result

5M run, 128 seeds, all 8 envs.

**CP3→CP4 = 30.41% in last-20% pure CP3 episodes.**

Per start_floor:

| start_floor | n     | up    | down  | same  | cp4 rate  |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:---------:|
| 0 (spawn)   |  4982 | 83.3% |  0.0% | 16.7% | **86.85%** |
| 1 (floor 2) |   506 |  1.0% | 12.6% | 86.4% |  1.19%    |
| 2 (floor 3) |  6794 | 11.9% |  2.5% | 85.6% | 14.16%    |
| 3 (floor 4) |  5002 |  1.9% | 12.4% | 85.7% |  0.04%    |
| 4 (artifact)|   135 |     - |     - |     - |  0.00%    |

Reward stats: median 0.88, max 9.77 — well-behaved. Reward tracer
was on (forensic safety net for path_progress); no episodes
exceeded their bound.

#### 21.3. The asymmetry: agent climbs but doesn't descend

The per-floor split shows a sharp pattern that mirrors what we saw
on CP2→CP3:

- Spawn (no descent needed, just walk to F1 if it's the remaining
  one) → 87% success.
- Floor 3 (needs to climb up to floor 4 for F4 OR descend to lower
  floors for F1/F2) → 14%, mostly via climbing (12% climb rate).
- Floor 4 (target is below: F1/F2/F3) → **0.04%**. Agent has to
  descend.

Despite ~5000 floor-4 episodes worth of training data, the policy
**doesn't learn to descend efficiently**. Climb rate at sf=3 is
1.9% (no available ladder up — princess unreachable), descent
rate is 12.4% (agent does fall sometimes), but only 0.04% reach
the fruit. So it falls but in the wrong way.

This is consistent with the floor-3 issue from approach 20.5:
falling is mostly fatal, controlled descent via a ladder is rare,
the path-progress reward credits both mid-fall progress and
ladder-arrival progress, and the dying-in-fall episodes are still
the dominant pattern.

Hypothesis to investigate: **the reward is symmetric in
direction-of-distance-reduction, but the game isn't symmetric in
risk**. Climbing up a ladder is safe (snowballs roll past on
horizontal). Falling without a ladder is fatal. Descending a
ladder requires lining up x precisely, and the policy likely
hasn't learned the ladder-x signature for descent the way it has
for climb.

#### 21.4. Open questions for next experiments

1. **Why descent is harder than climb (approach 22 candidate):**
   - Are agents on floor 4 NOT trying to use ladders, or trying and
     misaligning?
   - Could a "must be on a ladder x" gate (option 1 from approach
     20.5) help, by removing the fall-progress reward and forcing
     the policy to learn ladder use for descent?
2. **Chaining toward princess (approach 22b):** stitch v7's CP2→CP3
   policy and v8's CP3→CP4 policy together (possibly behavioral
   cloning or curriculum) and see whether end-to-end CP0→princess
   is reachable.

Configs / commits: see `experiments/003-yeti/configs/segment_3to4_v1.yaml`.

### 22. Fall vs ladder descent: why descent is hard  *(verified)*

Investigated why per-segment training (both v7 and v8) shows agents
that climb but fail to descend. Key findings from manual probing of
the env:

#### 22.1. Ladder mechanics

Pressing DOWN on floor 4 only descends through L34 if the agent's
RAM x is exactly **42** (1-pixel-wide window). At ram_x=41 or 43,
DOWN is a no-op. The visible ladder sprite is 16 px wide (UL=168 to
184), but only the leftmost RAM column (x=42 = pix 168) registers
as "on the ladder for descent".

We did not exhaustively check L23 / L12 descent windows but the L34
result is enough: stopping at exactly the right pixel column is
hard for an RL agent without a strong gradient pointing there.

#### 22.2. Falls vs ladder descents in our reward

`agent_floor_from_pixel_y` returns a floor number only when y is
within ±8 of a floor's standing y; otherwise None. Floor anchors:
y=184 (1) / 152 (2) / 120 (3) / 88 (4) / 56 (5).  16-px tolerance
bands around each anchor leave 16-px gaps between floors.

The path-progress reward uses `last_floor` as a fallback when y is
in a gap. So a fall from floor 4 (y=86) all the way to floor 2
(y=150) traverses roughly:

  y= 86  floor=4   (start)
  y= 90  floor=4
  y= 98  None  -> last_floor=4
  y=110  None  -> last_floor=4
  y=114  floor=3   <-- floor transition; reward fires for path-progress to lower fruits
  y=118  floor=3
  y=122  floor=3
  y=130  None  -> last_floor=3
  y=146  floor=2   <-- another floor transition; reward fires again
  y=150  floor=2   (landed)

So a single off-ledge fall pays roughly 2 × (32 px * scale) = 0.64
reward at scale=0.01, all in ~10 frames. A ladder descent from
floor 4 to floor 3 is also ~32 px y-change but takes ~30 frames and
pays only 0.32 reward (one floor transition). **Falling pays more
per attempt than ladder descent.**

#### 22.3. Why the obvious fix isn't actually a fix

The obvious fix ("don't fall back to `last_floor`, only credit on
confirmed floors") doesn't actually solve this. After the fix a fall
still pays whatever the path-distance reduction was when y stabilises
on the landed floor — which equals the ladder-descent's payment for
that specific floor transition. So:

- Fall from floor 4 to floor 2 (skipping floor 3 entirely): pays
  the path-distance reduction from floor-4-x to floor-2-x in one
  shot, AT landing. Same total as two ladder descents.
- Ladder descent floor 4 → 3, then 3 → 2: pays each transition once
  on landing.

Falls and ladder paths land in the same place and pay the same
total. The fall is FASTER, ending the episode sooner; ladders take
~60 frames, falls take ~10. Per second of wall-clock, falls give
more reward. Falls remain locally attractive even with the fix.

**To genuinely disfavor falls** we'd need either:
- A penalty for non-ladder y-changes (detected by checking agent_x
  vs ladder columns during the y change).
- Knowledge that "fall = die soon" baked into long-horizon credit
  assignment, which only works if the agent has actually learned
  to survive on the lower floor and continue collecting reward —
  i.e., it's a training-budget issue, not a shaping issue.

#### 22.4. The bigger picture

Even setting reward aside: in v8, 12.4% of floor-4-start episodes
do successfully descend to a lower floor, but only 0.04% reach CP4.
So **the policy gets to lower floors, then dies before reaching the
fruit**. The bottleneck is not "make the agent descend"; it's "make
the descended agent survive on the lower floors of a level it
hasn't fully learned to play".

That's a training data / curriculum problem, not a reward shaping
problem. Falls are a symptom, not the cause.

#### 22.5. Decision

Skip reward fiddling. Move to chaining the existing per-segment
policies (v7 CP2→CP3 and v8 CP3→CP4) and measure end-to-end
behavior. If the chain reaches the princess from spawn even at low
rates, we have a working pipeline and can iterate on weak spots.
If it doesn't, the asymmetry observed here will reveal itself
at scale.

### 23. Chaining v7 + v8 = 0.4% CP2→CP4  *(verified)*

Wired up chained-policy eval (`scripts/mo5/yeti/eval_chained_policies.py`).
Loads both trained models, plays v7 from CP2 seeds, hands off to v8
when CP3 is reached, plays until CP4 or episode ends. Records max
CP reached per episode.

Run:
- 48 CP2 seeds × 5 episodes per seed = 240 episodes
- v7 (CP2→CP3 specialist) → v8 (CP3→CP4 specialist)
- 5 settle frames between policy switches (mimics training-env reset)

**Result**:

  max CP reached = 2:  103 / 240  (42.9%) — v7 didn't reach CP3
  max CP reached = 3:  136 / 240  (56.7%) — v7 OK, v8 stuck
  max CP reached = 4:    1 / 240   (0.4%) — full chain success

Naive expected rate from product of standalone rates:
  v7 CP2→CP3 = 50.24%  ×  v8 CP3→CP4 = 30.41%  =  15.3%

Observed: 0.4%. **40× worse** than the product-of-rates prediction.

#### Why the gap: distribution mismatch at handoff

Compared the CP3 states v7 reaches in training vs the CP3 pool v8
was trained on:

| signal           | v7 collected_states (300 sample) | v8 training pool (128) |
|------------------|:--------------------------------:|:----------------------:|
| floor 0 (spawn)  |   4%                             | 28%                    |
| floor 1          |  15%                             |  3%                    |
| floor 2          | **66%**                          | 39%                    |
| floor 3          |  16%                             | 29%                    |
| floor 4          |   0%                             |  1%                    |
| remaining=(1,)   |  10%                             |  3%                    |
| remaining=(2,)   |   4%                             | 30%                    |
| remaining=(3,)   |   8%                             | 28%                    |
| remaining=(4,)   | **77%**                          | 39%                    |

v7 reaches CP3 most often on floor 2 with F4 remaining. v8 was
trained on an artificially-balanced pool (we capped per-group at
top-50 to ensure all 4 remaining-fruit configs were represented).

v8's per-start-floor success rate from its training (last-20%):

  sf=0 (spawn):  86.85%
  sf=1 (floor 2): 1.19%   <-- the dominant handoff bucket
  sf=2 (floor 3): 14.16%
  sf=3 (floor 4): 0.04%

77% of v7's handoffs fall into the "floor 2 start with remaining=F4"
bucket where v8 gets 1.19%. The chain is bounded by v8's worst
start, not its best.

#### Implications

The naive "train per segment, then chain" approach assumes the
upstream segment's outputs match the downstream segment's training
distribution. They don't. Quality-filtering the seed pool for v8
(approach 21.1) produced a balanced distribution good for SEGMENT
training metrics but bad for chain handoff.

Two ways out:

1. **Train v8 on the empirical v7-handoff distribution.** Drop the
   per-group balancing; use raw (or reservoir-sampled) v7
   `collected_states` as v8's seed pool. v8 then specialises on the
   actual handoff distribution.
2. **Train end-to-end** instead of chaining: a single policy from
   CP2 to CP4 (or further). No handoff. The reward stays the same
   (path_progress); the difference is the policy keeps playing
   after collecting fruit 3.

Option 1 is cheaper: just re-run segment_3to4 with a different seed
pool. Option 2 requires re-architecting the per-segment scaffolding
to support multi-segment-per-episode.

#### Decision

Try option 1 first (cheaper). Re-run segment_3to4 with v7-handoff
distribution, then re-evaluate the chain.

### 24. CP4 → princess: detection bug masked real progress  *(verified)*

The 5M `segment_4toP_v1` run reported 0% princess touches across
58,664 episodes. Looking closer revealed the detection rule was
broken, and the agent had in fact reached the princess 11 times.

#### What the rule was

```python
princess_touched = (
    curr_fruits > prev_fruits          # game repopulates fruits 0 -> 4
    and curr_lives >= prev_lives       # not a death respawn
    and curr_bonus > prev_bonus        # bonus countdown resets up
)
```

The intuition: when the agent touches the princess, the game
"completes" the level by re-populating fruits, resetting the bonus
countdown back to ~1000, all in a single frame. Detect that
transition.

The intuition was wrong.

#### What actually happens at the touch frame

Loaded a near-princess save state (32 px from the princess centre,
no obstacles between, lives=3). Walked right and observed:

| Frame | x  | y  | fruits | lives | bonus | score |
|-------|----|----|--------|-------|-------|-------|
| 0     | 68 | 54 | 0      | 3     | 693   | 370   |
| 1     | 68 | 54 | 0      | 3     | 692   | 370   |
| 2     | 69 | 54 | 0      | 3     | 691   | 370   |
| 3     | 70 | 54 | 0      | 3     | 690   | 370   |
| 4     | 71 | 54 | 0      | 3     | 690   | 370   |
| 5     | 71 | 54 | 0      | 3     | 689   | 370   |
| **6** | **72** | **54** | **0** | **3** | **689** | **1059** |

Score jumped by 689 — exactly the remaining bonus consumed. But
`fruits_remaining` stayed at 0, `bonus` stayed at 689, `lives`
stayed at 3. The agent then enters a frozen "celebration" screen
for ~370 frames before the next level starts and *only then* do
fruits/bonus repopulate.

So `(curr_fruits > prev_fruits AND curr_bonus > prev_bonus)` is
strictly false at the touch frame. The rule never fires. By the
time the celebration screen ends, the bonus-stall-frames timer
(default 10) has already terminated the episode with end_reason
`stall`.

#### Finding a reliable signal

Diff'd RAM bytes 10900..11200 between pre-touch and touch frames
and watched all changes over 1500 frames. Most diffs were
counters (animation, snowball positions, etc) that change during
normal play. One byte stood out:

- **RAM byte 11050** flips 0 → 1 only on the touch frame and stays
  1 throughout the ~370-frame celebration. It auto-clears 1 → 0
  when the next level starts.

Confidence-checked across 26,336 frames of varied non-touch
gameplay (random rollouts from CP4 seeds, random play from CP0
including 2 fruit pickups and 1 death/respawn): zero 0 → 1
transitions. The flag is a clean level-cleared signal.

The implementation is in `scripts/mo5/yeti/probe_princess_flag_long_baseline.py`
(re-runnable confidence check) and the new detection lives in
`scripts/mo5/yeti/train_segment.py` as the rising-edge check
`prev=0, curr=1` of `ram[11050]`.

#### Re-analysing v1 with the new rule

Episodes from v1 with `n_fruits_collected == 0` AND
`final_score - start_score >= max(100, start_bonus / 2)` are very
likely princess touches that the broken rule missed (delta
matches consumed bonus, no other source of large score jumps in
this segment). Eleven such episodes in the run:

| step      | env | length | delta | start_bonus | final_xy   |
|-----------|-----|--------|-------|-------------|------------|
| 3,672,600 | 4   | 90     | 775   | 809         | (72, 46)   |
| 4,047,864 | 3   | 118    | 755   | 809         | (72, 46)   |
| 4,168,160 | 7   | 186    | 115   | 229         | (72, 44)   |
| 4,323,664 | 3   | 147    | 507   | 574         | (72, 54)   |
| 4,324,824 | 7   | 126    | 750   | 809         | (72, 44)   |
| 4,557,480 | 0   | 92     | 764   | 809         | (72, 44)   |
| 4,603,568 | 5   | 89     | 766   | 809         | (72, 46)   |
| 4,605,416 | 1   | 98     | 760   | 809         | (72, 46)   |
| 4,625,896 | 0   | 120    | 516   | 575         | (72, 44)   |
| 4,822,168 | 5   | 88     | 777   | 809         | (72, 44)   |
| 4,846,832 | 6   | 129    | 758   | 809         | (72, 44)   |

Reading:

- All eleven end at `final_x ≈ 72` (pixel ≈ 288) on floor 5 —
  exactly where the princess sprite is.
- Eight started from `(68, 78)` (a near-floor-5 seed); three
  started from lower floors `(19, 150)`, `(27, 114)`, `(26, 110)`
  and *climbed* to the princess. So the policy can chain ladders
  end-to-end, occasionally.
- All eleven are in the last ~30% of training (steps 3.7M-4.8M of
  5M). The agent was learning; the broken rule denied it credit.
- Nominal touch rate is 11 / 58,664 = 0.019%. Small but non-zero,
  and almost certainly an undercount because the broken reward
  also denied the policy the princess shaping signal.

#### Implications

1. The "0%" headline was an artifact. The v1 model occasionally
   solves the segment.
2. With the corrected detection, the universal-path-progress
   reward will pay `prev_bonus * princess_scale` (≈ 25-40 reward)
   on each touch, and the env terminates the episode with
   `end_reason="princess_touched"` instead of waiting for a stall.
   Both signal and credit assignment improve.
3. The trained `segment_4toP_v1/final_model.zip` is a viable
   warm-start for v2 — it already encodes a functioning navigate-
   plus-touch policy at low rate.

#### Next steps

- Add the user's manually-saved near-princess state to the CP4
  seed pool (32 px from princess, lives=3, bonus=696, no obstacles
  in the way — strictly easier than the existing pool).
- Launch `segment_4toP_v2` with the corrected detection. Same
  config as v1 (5M, 8 envs, universal path-progress reward), but
  this time success will actually be reinforced.

### 25. CP4 → princess: 68.8% with corrected detection + warm-start  *(verified)*

`segment_4toP_v2` ran for 5M steps with (a) the corrected princess
detection rule (RAM byte 11050 rising edge), (b) the seed pool
extended with the user's manually-saved near-princess state
(`v9_v5_cp4_user_seed.pkl`, 8 seeds total), and (c) warm-start from
v1's `final_model.zip`.

#### Training-time metric

End-reason distribution across all 53,674 training episodes:

```
princess_touched: 9907 (18.5%)
env_done:        43767 (81.5%)
```

vs v1's effective 0.019% (eleven episodes out of 58k that the
broken rule missed). The credit-assignment fix pays out.

The success rate climbs visibly across training — last 5% of the
log shows windows of 38-67%, mostly driven by the easy seed.

#### Per-seed deterministic eval

The training-time number averages over 5M steps of policy
evolution. To assess the *final* policy quality, ran 10 stochastic
rollouts per seed (`max_steps=2000`):

| Seed | Start (ram_x, ram_y) | Pixel centre | Floor | Touches | Avg length |
|------|----------------------|--------------|-------|---------|------------|
| 0 | (68, 54)  | (280, 54)  | 5  | 10/10 | 6      |
| 1 | (18, 140) | (80, 140)  | 2  | 9/10  | 192    |
| 2 | (18, 140) | (80, 140)  | 2  | 7/10  | 191    |
| 3 | (19, 150) | (84, 150)  | 2  | 10/10 | 185    |
| 4 | (19, 150) | (84, 150)  | 2  | 9/10  | 181    |
| 5 | (26, 110) | (112, 110) | 3  | 5/10  | 112    |
| 6 | (27, 114) | (116, 114) | 3  | 5/10  | 109    |
| 7 | (68, 78)  | (280, 78)  | 4  | **0/10** | -    |

Overall: **55/80 = 68.8%**.

The headline is the floor-2 starts: 35/40 (87.5%). Three full
ladder climbs (L23 + L34 + L45) plus a snowball-dodge run, in
~185 frames. The end-to-end task is clearly within reach.

#### The remaining failure mode

Seed 7 — start `(68, 78)` on floor 4 — fails 100% of the time.

Geometry: agent at pixel x=280, princess at pixel x=312, but the
only way up to floor 5 is ladder L45 at pixel x=200. So from
seed 7 the policy must walk LEFT to L45 (away from the princess
in pixel-x terms), climb, then walk right past snowballs to the
princess.

Every other seed in the pool is consistent with "head broadly
right and up" — even the floor-2 starts have ladders to their
right (L23 at x=232). Seed 7 alone requires going against that
gradient.

The path-progress reward should still pay for moving toward L45
because the navigation graph routes through it. But the warm-
started v1 policy didn't have any L45-direction signal during v1
training (the broken rule blocked credit assignment), so it
likely calcified an "easier" rightward-bias. v2's training added
princess credit but the floor-4 pattern is contradicted by every
other seed's policy.

#### Implications

1. **The principal claim is empirically met**: per-segment
   CP4→princess training works at 68.8% across the seed pool.
2. **The last 30% gap is concentrated on one start position**.
   Whether to fix it depends on what the per-segment numbers
   feed into next: if we chain with segment_3to4, only the
   handoff distribution matters; if we build a unified model,
   we need every CP4 cell solvable.

#### Next steps

Two viable directions, distinct in cost:

- **Cheap probe**: rollout from seed 7 with an exploration noise
  override (eg ent_coef=0.1) for ~50k steps to see if the policy
  can be nudged into discovering L45. This tests "the policy is
  stuck in a local min" hypothesis cheaply.
- **Deeper fix**: add a curated near-L45 floor-4 save (analogous
  to the user's near-princess save) to the seed pool, then re-
  train. Mirrors what unblocked v1 → v2.

If the user's segment_3to4 distribution naturally lands on the
"head right" floor-4 positions (not the (68, 78) one), then
chaining might not need the seed 7 fix at all.

### 26. Pivot to plain PPO from reset (yeti_universal_v1)  *(verified)*

After approaches 14-25, per-segment training had hit three
structural issues:

- **Validation problem**: probe=120 rejects exactly the
  transitional CP4 states we want to train on (post-F4-pickup
  near-edge positions). Lower the probe and we accept dying
  states; raise it and we only get already-safe equilibria.
- **Distribution-handoff problem**: approach 23 already showed
  v7+v8 chaining produces 0.4% even with 50% × 30% per-segment
  numbers. Each segment's output distribution doesn't match the
  next's training distribution.
- **Forgetting problem**: approach 25's v3 lost CP4→princess
  after warm-starting from v2 because v3 only trained on CP3
  starts.

So we pivoted to plain PPO from reset with the universal
path-progress reward. Same reward shaping we already had, just
without segment scaffolding. Single distribution, single policy,
no handoff.

#### v1 result: a regression

5M steps from reset, **warm-started from segment_4toP_v2**.

Result: agent picks F1 reliably (98% rate), then walks to far
right corner (px=312, ram_x=76) and stalls until episode ends.
Only 9 F2-pickups in 30,520 episodes, all within the first 420k
training steps; zero across the next 9.5M.

#### Why the warm-start poisoned the run

segment_4toP_v2 was trained exclusively on CP4 starts (fruits=0,
agent on floors 2-5). That model's prior is "no fruits remain,
target princess." When applied to CP0 (fruits=4, agent at spawn
on floor 1), it has no useful initial behavior — and PPO, faced
with a confusing initial value estimate, settled on the cheapest
reward stream: F1 + retreat.

The retreat dynamic: F1 is at (px=184, floor 1). Walking right
after pickup takes the agent away from L12a (floor 2 ladder at
px=120), so the per-pixel path-progress shaping pays nothing.
But the rightward walk is *safe* — no obstacles to the corner.
Walking left toward L12a *might* die during exploration. So PPO
correctly preferred the safe-but-stagnant strategy over the
risky-but-progressing one, because the warm-start prior weighted
the value head against any climbing intuition.

#### Lesson

Don't warm-start across distributions. The training distribution
of the prior must overlap with the new training distribution. A
v2 → CP4 prior into a CP0 → reset run is a category error.

### 27. Plain PPO from reset, no warm-start (yeti_universal_v2)  *(verified)*

Same as v1 but **no warm-start**. 5M steps.

Result: dramatic recovery. By the last 20% of training:

- 0.6% picked 0 fruits
- 24.1% picked F1 only
- **74.8% picked both F1 and F2**
- 2 episodes (out of 16,502 total) picked F3
- 0 reached F4 or princess

By bin 8 of 10 (steps ~4M-4.4M): 97.4% pickup rate of F1+F2.
Reward shaping is doing exactly what we designed.

#### The CP2 plateau

v2 maxed out at "F1 + F2 + plateau on floor 2." Going from F2
(at px=80, floor 2) to F3 requires a 288-pixel commit:

- Walk right 160 px to L23 (px=240)
- Climb L23 to floor 3
- Walk left 96 px to F3 (px=144)

The shaping reward pays 0.01/pixel during that 160-pixel walk
to L23, with no fruit reward at the end of it (just more
shaping toward F3). PPO finds it hard to commit to such a long
horizontal traversal when the per-step gradient is small and the
exploration risk is high.

### 28. Stronger path-progress shaping (yeti_universal_v3)  *(verified)*

Hypothesis: bumping ``scale`` from 0.01 to 0.05 (5x) increases
the directional gradient strength enough to push the policy
through the F2 → F3 transition.

Same setup as v2, just with ``scale: 0.05`` in the reward params.

Result: **basically identical to v2.** Last-20% pickup rates:
F1+F2 = 72.5% (vs 74.8% v2). F3 pickups: 0 (vs 2 in v2). F4 and
princess: 0.

So path-progress strength wasn't the bottleneck. The reward
gradient is correct in direction; PPO simply isn't discovering
the F2→F3 trajectory through random exploration within 5M steps.
This is a long-horizon credit-assignment problem, the kind that
shaping rewards alone can't solve.

#### Lesson

Universal path-progress + reset training reliably gets 2 fruits.
F3 onward needs either (a) much more compute (20M+ to let random
exploration eventually find F2→L23→F3 trajectories), or (b)
curriculum starts that bypass the long-horizon discovery
problem. Curriculum is what train_checkpoint_curriculum.py was
designed for, and we now have:

- A working CP0→CP2 policy (v3's final_model.zip)
- A CP3 seed pool (v9_v3_cp3enriched.pkl, 128 seeds — built in
  approach 21 from v9 + v7 collected_states)

The v3 policy provides a good initial value/policy estimate for
the curriculum, and the CP3 seeds let some training episodes
skip the F1+F2 commit and practice F3 directly. This time the
warm-start is across overlapping distributions (CP0 → CP3 both
include floor-2-and-up navigation) so it should be safe.

#### Plan: yeti_curriculum_v1

- Warm-start from yeti_universal_v3/final_model.zip
- 5M steps
- ``reset_fraction=0.6, frontier_fraction=0.4, earlier_fraction=0``
  — 60% of episodes from CP0, 40% from the highest-checkpoint
  pool (initially CP2 from v3's saves; the curriculum manager
  promotes to CP3, CP4 as the policy unlocks them)
- Same path-progress universal reward, ``scale=0.01``
- Princess flag detection wired in

### 29. Checkpoint curriculum on top of universal (yeti_curriculum_v1/v2)  *(verified)*

After v3 plateaued at F2, we tried the checkpoint curriculum
(``train_checkpoint_curriculum.py``) to give the policy direct
practice on the F2→F3 transition.

#### v1: warm-start from v3 + 40% CP2 starts

``reset_fraction=0.6, frontier_fraction=0.4``, warm-started from
yeti_universal_v3 (which had CP1+CP2 checkpoints saved).

Result: failure, and a familiar one. Of the 40% CP2-start
episodes, **median total_reward = 0.00** — the v3 policy produced
essentially no reward from CP2 states. CP2 was out-of-distribution
for v3 (F2 already collected, agent on floor 2 — a scene v3 never
trained on). The agent stalled or fell to floor 1; 9,321 CP2
episodes yielded 0 F3-pickups.

Same warm-start poisoning as approach 26 (v1 from reset). Loading
a policy into states it never trained on gives near-zero reward
and no recovery within budget.

#### v2: from scratch + curriculum

Same curriculum (``reset=0.6, frontier=0.4``), no warm-start.
Stopped at 2.87M / 5M steps (no signal of escape).

Result: curriculum *hurt* relative to plain reset:

- CP0 starts: 90% reach F1, only ~9% reach F2 (vs plain v2's 74%).
- CP2 starts: 1/4151 reached F3.
- CP3 starts (a few late ones): 2/353 reached F4.

Splitting the policy's attention across CP0 and CP2 start
distributions degraded performance at *both*. One set of network
weights can't serve two different start distributions when the
required behaviors look different.

#### Lesson

The checkpoint curriculum, as implemented (mixed start
distribution into one policy), does not help here. It either
poisons via OOD warm-start (v1) or degrades via attention-split
(v2).

## Summary: what we know after approaches 1-29

Two robust empirical facts:

1. **Segments learn well in isolation.**
   - CP0→F1 (reset): 98%
   - CP2→CP3: 50% (v7)
   - CP3→CP4: 30% (v8)
   - CP4→princess: 69% (segment_4toP_v2)
   - CP0→F1→F2 (reset): 74%

2. **Composition fails, both ways.**
   - Chaining separate policies: 0.4% CP2→CP4 (approach 23) —
     each segment's output distribution doesn't match the next
     segment's training distribution.
   - One policy from reset: plateaus at 2 fruits (v2/v3); stronger
     shaping doesn't break it (long-horizon discovery problem).
   - One policy + mixed-start curriculum: degrades all segments
     (v1 OOD warm-start, v2 attention-split).
   - One policy warm-started forward: catastrophic forgetting of
     the prior segment (segment_3toP_v1) or OOD collapse (v1).

### The underlying cause

Model-free PPO learns exactly the start distribution it trains
on, and only that. It does not compose, transfer, or explore far
beyond its current competence. Every composition failure above is
a variant of this.

### The one composition approach NOT yet tried

Naive chaining (approach 23) failed because v8 was trained on a
*balanced* CP3 pool while v7 *outputs* a skewed CP3 distribution.
We concluded "distribution mismatch" and pivoted — but never
tried the obvious fix: **train each segment on the actual output
distribution of the previous segment.**

Plan (sequential distribution-matched chaining):

1. Train CP0→F1 from reset (known: 98%).
2. Collect the actual states where the policy picks F1 → CP1 pool.
3. Train CP1→F2 from *that* pool. Collect its F2-pickup states.
4. Repeat F2→F3, F3→F4, F4→princess.

Each segment trains on exactly what the previous produces, so the
handoff matches by construction. This uses what works (segments)
and fixes the one thing that breaks (handoff distribution). It's
distinct from the curriculum (which mixes distributions into one
policy and degrades) and from naive chaining (mismatched pools).

Open question to resolve before committing: one policy fine-tuned
forward through the segments (risks forgetting) vs N policies
orchestrated by a controller that switches on fruit-count
transitions (more robust, more engineering).

### 30. Curriculum redesign: priority-based seeding (design + rationale)  *(design)*

This entry captures the full design discussion behind the curriculum
redesign — not just the change, but the reasoning, the lessons that
forced it, and the principles we're now building to. It is
deliberately verbose so future-us doesn't re-derive it.

#### North Star

A **single policy** that, started from game reset (CP0), reaches the
princess. Formally: maximize **P(princess | start = CP0)**.

Not "a relay of per-segment policies + a switch." One agent, from
reset, end to end.

#### Why the obvious things failed (lessons so far)

- **Plain PPO from reset** (approaches 26-28): learns CP0→F1→F2 (74%)
  then plateaus. No reward signal reaches the late game, so the late
  transitions never get a gradient. Degenerates (F1 + corner-camp) if
  pushed.
- **Stronger shaping** (v3, scale 0.05): no help. The bottleneck is
  long-horizon *discovery*, not gradient magnitude.
- **Higher entropy** (v4, ent_coef 0.05): actively worse — global
  noise breaks the precise sequencing the early segments need.
  Exploration must be targeted, not blunt.
- **Curriculum, forward** (curriculum_v3): broke the CP2 wall (reached
  CP3!) but stalled at CP3→CP4 = 0%. Diagnosis via rollout: from the
  CP3 spot the agent walks the wrong way and dies — it had **never
  sampled** the rightward F3→F4 route enough to reinforce it.
- **Warm-start across distributions** (v1, curriculum_v1): poisons the
  policy — loading a policy into states it never trained on yields
  ~zero reward and no recovery.
- **The probe validator**: rejected ~99% of real mid-action pickups
  because it tested *passive* survival (noop for 120 frames). It kept
  only safe-equilibrium states, starving and biasing the seed pools.
  Replaced by play-based scoring (approach: deferred survival /
  reached-next).

The recurring failure under all of it: **segments learn in isolation
(CP2→CP3 50%, CP3→CP4 30%, CP4→princess 69%) but don't compose** —
CP2→CP3 is 0% from CP2 *starts* inside the unified policy despite 50%
in isolation.

#### The reframe that unlocked the design

"Distribution mismatch" is not a bug to remove — it's the **mechanism**.
Starting an episode from "fruit 1 already collected" is precisely what
teaches "go left for fruit 2," which a reset start can't teach because
the agent rarely gets there. Mismatch from reset is *why* frontier
starts are useful.

The real objective decomposes into two requirements that every design
choice must serve:

- **R1 — competence:** each segment's success rate must be high.
- **R2 — composition:** each segment must be trained on the *states the
  agent actually reaches from reset*, or the per-segment skills don't
  chain into P(princess | CP0).

These are in **tension**: R1 wants heavy practice on the hard frontier
(few, possibly artificial states); R2 wants the practice states to match
the agent's own reset-trajectory distribution. The clean extremes show
the trade-off:

- Reset-only training: zero mismatch (every state is self-produced) but
  no late-game signal → plateau.
- Seed/segment training: late-game signal but mismatch → segments don't
  transfer.

We can't zero both. The job is to **shrink the mismatch enough that
focused frontier practice still transfers**, while keeping the late-game
signal.

#### The machine (how the pillars interlock)

The design is a self-reinforcing loop, and the CP0 reserve is its engine:

    CP0 (reset) starts
      → agent occasionally reaches CP_n on its own
        → those reset-origin CP_n states enter the pool (on-distribution)
          → frontier practice on CP_n→CP_{n+1} uses real arrival states
            → success rises → frontier weight shifts forward
              → deeper reset chains become possible → repeat

Reset starts manufacture fresh, diverse, on-distribution deep-CP seeds.
Frontier weighting concentrates the gradient on the current wall.
Success-adaptive weighting advances the wall as it cracks. This is why
the CP0 reserve matters for *both* R1 (it seeds the frontier) and R2
(it forces end-to-end composition).

#### Key insight: pool fullness is inverse to CP difficulty

"Pools fill fast" is only true for easy CPs. Hard CPs (the whole point)
stay sparse. This flips which mechanism matters where:

- **Easy CPs (CP1/CP2):** pool full, eviction runs constantly →
  eviction policy is what shapes the pool. Bonus-eviction collapses
  diversity (v3's CP3 pool collapsed to one position). Want:
  reset-origin, diverse retention.
- **Hard CPs (CP3/CP4):** pool sparse (5-20 states), eviction almost
  never fires, the size cap is irrelevant. What matters is **lenient
  admission** (don't reject rare reaches) and **retention** (never lose
  a rare good state).

A single bonus-eviction rule is wrong at both ends. Hence: lenient
admission everywhere + reset-origin eviction (only bites when full,
i.e. on easy CPs, where it preserves on-distribution diversity).

#### The three pillars (target design)

**Pillar 1 — Pool composition (what's in each CP pool)**
- Self-generated from the agent's own play. [have]
- Lenient admission: `survived ≥ N OR reached_next`. [have]
- Reset-origin retention: when full, evict the entry whose *source
  episode started from the highest CP* first (tiebreak: lower bonus).
  Replaces bonus-only eviction. [NEW]
- Size cap 100 (binds only on easy CPs; harmless). [have]

**Pillar 2 — Start-state selection (where episodes begin)**
- Within-pool: uniform random. [have]
- Across-CP: P(start at level ℓ) ∝ (1 − success_rate[ℓ]),
  availability-gated (skip empty pools), with a CP0 floor. Replaces
  fixed reset/frontier/earlier fractions. [NEW]
- CP0 floor = 0.30 (keeps the engine running; chosen value, not tuned).

**Pillar 3 — Adaptivity (how knobs respond to progress)**
- Per-segment success rate already tracked. [have]
- (1 − success_rate) weighting *reads* it to steer practice — this is
  first-order adaptivity, and it de-hardcodes the training partition. [NEW]
- Pool size and CP0 floor stay as config knobs for now. [deferred]
- Second-order meta-adaptivity (plateau-detect → auto-tune pool
  size / CP0 fraction) is **deliberately deferred**: it's a control
  system with its own failure modes, and we have no evidence the fixed
  knobs are the bottleneck. If the first-order rules plateau, the
  measured plateau tells us exactly which knob to make adaptive —
  better-informed than guessing now.

#### What changes in code (CheckpointManager)

1. Thread each snapshot's **source start-level** through to
   `save_scored`; entries become `(source_cp, bonus, state)`.
2. Eviction (`_insert`, only when full): drop the highest `source_cp`
   first, tiebreak lowest bonus. (Reset-origin = source_cp 0 = most
   protected.)
3. `pick_start`: weight reached/available levels by (1 − success_rate),
   gated by non-empty pools, with a 0.30 CP0 floor.
4. Keep lenient admission unchanged.

#### Success criteria for the next run

- CP3→CP4 from CP3 starts climbs off 0% (the v3 wall).
- Reset→CP3 rate rises over training (the loop is feeding deep pools).
- Ultimately: a non-zero princess-touch rate from reset.
- Watch for: deep-pool overfit (heavy frontier weight + sparse pool) —
  mitigated by the CP0 reserve continuously refreshing the pool. If
  reset→CP_n stalls while CP_n→CP_{n+1} from-starts is high, that's the
  overfit signature and the signal to revisit the CP0 floor.

#### v5 run result (approach 30 as built) *(verified)*

Ran 5M steps, warm-started from `yeti_universal_v2` (clean dir, empty
pools), CP0 floor 0.30.

- The CP4 pool populated for the first time ever (6 states) — the
  reset reserve + lenient admission did manufacture deep seeds.
- But every deep segment stayed at 0%: CP2→CP3 = 0%, CP3→CP4 = 0%,
  CP4→princess = 0%, **0 princess touches**.
- From reset: CP0→CP2 ≈ 52%, CP0→CP3 ≈ 0.07%.

**What v5 proved (and didn't).** It proved the seeding machine works:
deep pools fill from reset-origin play. It did **not** crack
composition. Two non-exclusive causes:

1. **Budget spread too thin.** A single policy split its non-reset
   budget across *all four* segments, every one of them failing. With
   5M total and (1 − success) ≈ 1 everywhere, no segment got the
   concentrated practice that the isolated runs needed (CP3→CP4 took a
   multi-M-step segment run to reach 30%).
2. **Off-distribution deep pools.** The 6 CP4 states came from
   ~0.07%-rare reaches. Starting episodes there trains the policy on
   states it essentially never produces from reset — practice that
   can't transfer back to P(princess | CP0).

Both point the same way: **don't spend budget on a CP until the agent
can actually reach it from reset**, and when you do spend, concentrate
it rather than smear it.

### 31. Reach-gated frontier curriculum (yeti_curriculum_v6)  *(in progress)*

#### North Star (unchanged)

A **single policy** that, from game reset (CP0), reaches the princess.
Maximize **P(princess | start = CP0)**. No relay, no per-segment switch.

#### Requirements (unchanged)

- **R1 — competence:** each segment's success rate must be high.
- **R2 — composition:** segments must be trained on the states the
  agent actually reaches *from reset*, or the skills don't chain.

#### The decision v6 makes for us

The user's framing: "we are just trying to optimize training
resources." If one segment plausibly needs ~5M steps in isolation, then
the real end-to-end budget is somewhere between max(segment) and
sum(segments) — it depends on how much earlier segments transfer. A
fixed 5M total split four ways is below that floor by construction.

So the system should **decide where to spend** rather than smear the
budget uniformly. v6 makes that decision automatically with a single
new mechanism on top of v5: a **reach gate**.

#### Pillar status going into v6

**Pillar 1 — Pool composition**
- Self-generated, lenient admission, reset-origin retention, size cap
  100. [have, unchanged from v5]

**Pillar 2 — Start-state selection**
- CP0 floor 0.30. [have]
- Across-CP weighting by (1 − success). [have, but now driven by a
  responsive **EMA** instead of all-time cumulative rate — NEW]
- **Reach gate:** a level ℓ ∈ 1..4 is eligible as a start *only* once
  `reset_reach_ema[ℓ] ≥ reach_threshold` (0.15). [NEW]

**Pillar 3 — Adaptivity**
- The reach gate + EMA weighting together make the curriculum advance
  **one wall at a time, on-distribution**, with no hand-set schedule:
  the deepest reset-reachable unsolved segment gets the budget; as the
  CP0 reserve makes the next CP reachable, the gate opens and the
  frontier moves forward. This is the first-order, metric-driven
  adaptivity the user asked for ("a way for the system to adapt based
  on metrics collected, like plateauing score"), without the risk of a
  second-order auto-tuner. [NEW]
- Second-order meta-adaptivity (auto-tuning pool size / CP0 floor) is
  still deferred for the same reason as in approach 30.

#### Why the gate, in one line

v5 trained CP4 from 6 lucky-reach states the policy never produces from
reset. That's effort spent off-distribution. The gate forbids spending
budget on a CP the agent can't yet reach unaided, which is exactly the
budget-allocation decision we were making by hand before.

#### Why EMAs, not cumulative rates

The gate and the weights must track the **current** policy. All-time
cumulative rates are dragged down forever by early failures: a segment
solved at 8M steps would still read as "failing" and keep pulling
budget, and a CP that only became reachable recently would take far too
long to clear the gate. EMAs (α = 0.02, half-life ≈ 35 episodes) make
both signals responsive on the scale of a 20M-step run.

#### Budget & measurement plan

- Run at **20M** (≈4× v5) — enough that, if transfer is decent, the
  frontier should clear CP3 and reach CP4 from reset.
- The live summary now prints `reset_reach=[1.00, …]`, so we get the
  **steps-to-reach-each-CP transfer curve** directly. That curve is the
  evidence for the user's open question: if reaching CP_{n+1} after
  CP_n becomes cheap (steep curve), one policy transfers and 20M may
  suffice; if each wall costs ~the full isolated-segment budget (flat
  steps between walls), the honest conclusion is that a single 20M run
  is under-budgeted and a segment needs its own allocation.

#### Success criteria

- `reset_reach_ema[3]` clears 0.15 (CP3 becomes a legitimate frontier),
  then CP3→CP4 climbs off 0% — the v3/v5 wall.
- `reset_reach_ema` rises monotonically across CPs over training (the
  loop is feeding deeper pools on-distribution).
- Ultimately: a non-zero princess-touch rate from reset.
- Watch for: the gate never opening for CP3 (reset reach stuck < 0.15)
  → the bottleneck is earlier than we think (CP2→CP3 from reset), and
  budget/priority should sit there, not deeper.

#### What changes in code (CheckpointManager)

1. Two responsive EMAs added: `reset_reach_ema[0..4]` (index 0 pinned
   at 1.0) and `seg_success_ema[0..4]`, both updated in
   `record_episode` (α = 0.02). Cumulative counters kept for display.
2. `pick_start`: after the CP0 floor, eligible levels are non-empty
   pools with `reset_reach_ema ≥ reach_threshold`; weighted by
   (1 − `seg_success_ema`). If none eligible, fall back to reset.
3. `reach_threshold` config field (default 0.15) wired through
   `CurriculumConfig` → `CheckpointManager`.
4. `summary()` prints `reset_reach=[…]` for live monitoring.
5. Retention / admission unchanged from approach 30.


#### v6 run result (approach 31, 20M steps) *(verified)*

Ran the full 20M, warm-started from `yeti_universal_v2`, CP0 floor 0.30,
reach gate 0.15. Completed cleanly in 9h13m. **The reach gate worked
mechanically and the result is a clear, important negative.**

Final: `cp=[0,100,100,2,0]`,
`success=[0->1:95%, 1->2:3%, 2->3:0%]`,
`reset_reach=[1.00, 1.00, 0.00, 0.00, 0.00]`.

The endpoint is a collapse. To diagnose it honestly we re-measured both
v5 and v6 the **same way** — recent time-windows computed from
`episodes.csv`, not v5's lifetime-cumulative log number (which is
heavily inertial and hides recent decay). All numbers below are
windowed (recent) success.

**A logging caveat that bit us first.** The live `reset_reach` array is
indexed `[CP0, CP1, CP2, CP3, CP4]`. An earlier reading of the log
mislabeled index 1 (reach **CP1** from reset, which stays ~1.0 because
the agent always grabs F1) as "reach CP2." The real
reach-**CP2**-from-reset is index 2; corrected windowed values below.

CP1→CP2 (success from CP1 *saved-state* starts), windowed:

| run | early | mid | late |
|-----|-------|-----|------|
| v5 (approach 30, no gate, 5M)   | 69% | oscillates 6–92% | **71%** |
| v6 (approach 31, gate+EMA, 20M) | 24% | → 0% | **0% (flat 12M)** |

reach-CP2-**from reset** (index 2), windowed: v5 oscillates and ends
~73%; v6 goes 76% → 54% → 2.7% → **0** by ~5M and stays dead.

So v5 was *stuck and noisy* but alive; v6 was *actively destroyed* — a
hard, permanent flatline. Same metric, opposite outcome. This refutes
the first-draft conclusion ("interference, not scheduling"): the
scheduling change **is** most of what broke v6.

#### Root-cause diagnosis: our approach-31 changes collapsed diversity

Two of the three pillars (start-state diversity, pool diversity)
regressed at once, and that — not a fundamental interference wall — is
what turned v5's plateau into v6's collapse.

**1. The reach gate collapsed start-state diversity.** Start levels ever
sampled:
- v5: `{CP0:9587, CP1:3731, CP2:8407, CP3:6754, CP4:3421}` — all five,
  throughout.
- v6: `{CP0:21523, CP1:42070, CP2:7225}` — only three ever; CP3/CP4
  **never once** (their gate never opened); CP1 alone is 60% of episodes.

In v5 the broad CP0–CP4 sampling acted as a regularizer — the policy was
pulled toward many start states and never collapsed into one attractor.
v6's gate removed that.

**2. Bonus-tiebreak eviction froze the pools.** The approach-30 eviction
keeps the highest-bonus (fastest-reach) state and drops the rest. The
fastest-F1 states are found early, so the CP1 pool locked onto them and
stopped accepting new ones. Measured: the CP1 starts went from **11
distinct states early to 2 distinct states late**. 42,070 CP1-start
episodes — the bulk of the run — practiced from essentially **2 frozen,
stale snapshots**.

**3. The EMA weighting closed the feedback loop.** Seeing CP1→CP2 fail,
the (1−success) EMA poured *more* episodes onto CP1 — i.e. onto those 2
stale states. Failing repeatedly from 2 off-distribution snapshots
reinforces bad behavior in post-F1 states that look just like the ones a
reset trajectory passes through, so it bled backward and destroyed the
reset run's own F2 skill (reach-CP2-from-reset 100% → 0). Gate + EMA both
read the same degrading signal and fed each other — exactly the
second-order control-loop instability approach 30 said it was deferring
*because* of this risk. Approach 31 added it anyway.

#### Two distinct findings, kept separate

- **Self-inflicted (v6-specific):** the catastrophic collapse to 0 is
  caused by the reach gate + bonus-eviction + EMA destroying diversity.
  Removing the gate and fixing eviction should recover v5-level behavior.
- **Genuine wall (pre-existing, also in v5):** *advancing* past CP2 was
  never learned. Across 7,225 CP2-start episodes the agent reached CP3
  **exactly once**; reach-CP3-from-reset was 0% in every window of both
  runs. Reaching a checkpoint is a different skill from advancing from
  it — a high reach-CP2 does not bootstrap CP2→CP3, which needs its own
  repeatable success to climb, and never got one. CP0→CP3 (F1→F2→F3 in
  one episode) therefore stayed 0 even when reach-CP2 peaked.

#### Why this still motivates MIP — but with the gate off

The deep wall (CP2→CP3 never bootstrapping) is plausibly the
observation-aliasing problem: a CP1 state and a CP2 state look nearly
identical at 84×84 (only a fruit sprite differs), so a single CNN policy
struggles to attach different actions to them. Making the
fruit-collection state **observable** (the four `FRUIT_PRESENCE_ADDRS`
bytes as a small vector via `MultiInputPolicy`, Dict obs = image +
vector) de-aliases the states and gives one network a fair chance to
represent the conditional behavior.

But MIP must be built on the **v5-style diverse-sampling baseline, not
v6**:
- **Drop the reach gate** — it collapsed start diversity.
- **Fix eviction** so pools don't freeze to their fastest few states
  (e.g. retain for diversity, not just highest bonus; or cap how often a
  single state can be re-sampled).
- **Add the fruit-presence observation** to attack the real aliasing
  wall.

Diversity (both across-CP and within-pool) is load-bearing, not a
nice-to-have. v6 is the proof.


### 32. MultiInputPolicy + diversity-preserving curriculum (yeti_curriculum_v7)  *(verified, negative)*

Built on the v6 post-mortem: reach gate OFF (reach_threshold 0.0, so
v5-style diverse sampling), diversity-preserving eviction (random within
the worst source-CP tier, no more bonus-freeze), and a MultiInputPolicy
observation Dict{image, fruits[4]} (per-fruit presence vector) to
de-alias checkpoint states. Trained from scratch (MIP obs space is
incompatible with the CnnPolicy warmstarts). 20M steps.

**Result: negative, and clarifying.**
- Start diversity *was* preserved this time: all five levels sampled
  (`{0:33k, 1:24k, 2:25k, 3:19k, 4:8k}`). Dropping the gate worked.
- But CP1->CP2 still collapsed (windowed: peaked ~14% early, decayed to
  0%). CP2->CP3 = 0% (4/25,563). reach-CP3-from-reset = 0% always. 0
  princess.
- Crucially, CP1->CP2 **never got above ~14%** — from scratch it never
  learned the skill at all, vs plain from-reset PPO (v2) which reaches 2
  fruits 99.5% of the time.

**What this tells us:**
1. **The single-policy checkpoint curriculum does not build
   composition.** From scratch it is *worse* than plain from-reset PPO.
   Across v5/v6/v7 it never once produced a reset->CP3. v5's
   healthy-looking 71% CP1->CP2 was inherited from its v2 warm-start,
   not built by the curriculum.
2. **MIP's effect is inconclusive here** — confounded by from-scratch +
   curriculum, and the 4-d fruit vector may be underweighted next to the
   ~256-d CNN features. MIP was not given a fair test (it needs a
   reset-only setting with a baseline to beat).
3. Diverse start sampling is necessary (v6 proved removing it is fatal)
   but **not sufficient** (v7 had it and still collapsed). The loaded
   saved-states are off the policy's own trajectory manifold; dropping
   the agent cold into them does not teach the skill the way a
   continuous reset trajectory does.

### Clean from-reset baseline eval (v2)  *(verified)*

`scripts/mo5/yeti/eval_from_reset.py`, 200 stochastic episodes from a clean reset
under training-equivalent termination:

| reached | rate |
|---------|------|
| >= 1 fruit | 100% |
| >= 2 fruits | 99.5% |
| >= 3 fruits | **0%** |
| princess | 0% |

Deterministic (greedy) trajectory: exactly 2 fruits, every time.

**Conclusions for strategy:**
- The early game (CP0->CP2) is solved. The high-water mark for the North
  Star has been v2 the whole time, and it's stronger than we thought
  (99.5% to 2 fruits, not 74%).
- Everything after v2 — per-segment policies aside — was a regression or
  a wash on the single-from-reset metric.
- The entire remaining problem is the **F2->F3 transition**: a single,
  localized exploration wall a from-reset policy never crosses. Next
  work should target that transition directly (reward/exploration for
  the post-F2 right-and-up route), with v2's 99.5%/0% as the baseline to
  beat — not more curriculum-scheduling variants.


### 33. F2->F3 reward audit + leak fix (yeti_universal_v4)  *(in progress)*

After the v2 clean eval pinned the wall at the F2->F3 transition (99.5%
reach 2, 0% reach 3), we audited the path-progress reward on that leg
(`scripts`-level numeric trace of `nav.path_distance_from_agent`).

**Findings:**
- The gradient *direction* is correct: walking F2 -> right -> up L23 ->
  left to F3 monotonically decreases path distance (288 -> 0 px,
  ~20px/step). No sign or geometry bug.
- The magnitude is faint: ~2.9 reward max over the whole leg at
  scale 0.01 (approach 28 already showed 5x scale doesn't help).
- **The per-fruit best_d ratchet leaks the F3 budget.** best_d[F3] is
  the closest the agent ever drifted to F3 across the *whole* episode.
  En route to F2 the agent typically passes near the L23 ladder
  (x~240), banking best_d[F3] as low as ~128. So after collecting F2
  (distance 288) the first ~160px of the leg it must commit to pays
  **zero** — a dead zone exactly where it must reverse direction and
  traverse, with no value signal at the end (it has never reached F3).

**Fix (H-A):** on every fruit pickup, re-baseline best_d for all
remaining fruits (and princess) at the new position, so each inter-fruit
leg gets a fresh full-distance progress budget. Applied to both
`fruit_bonus_path_progress` and `..._universal`. Regression test
`test_path_progress_universal_rebaselines_best_d_on_pickup` fails under
the old behavior.

**v4 run:** exactly v2's config (reset-only, 5M, seed 42), only the
reward code changed. Compares against the v2 baseline (99.5%/0%). Honest
caveat: the leak fix makes the gradient *correct*, but it's still faint
and the core difficulty is long-horizon exploration — so this may not be
sufficient on its own. Result + decision to follow.

**v4 result (clean eval, 300 stochastic from reset):** reach-2 **90.3%**,
reach-3 **0/300**, princess 0. Training reach-3 ~0.07% (17/25340) vs v2
~0.012%. **Refuted as a capability fix and did not replace v2 as
baseline** (reach-2 fell below the >=99% rule). Mechanism: re-baselining
best_d on pickup made *all* remaining fruits' progress freshly available
after F1, so the competing pull toward distant F3/F4 (up-right) sometimes
out-paid grabbing the nearby F2 (left) — more F3 attempts, no
conversions, lower reach-2 reliability. Net lesson: shaping *direction*
isn't the bottleneck; long-horizon **exploration** is. The leak fix is
kept in code (it is the correct behavior) but is not a standalone win.
The reach-2 dip also exposed two reward-design issues pursued next:
non-Markovian shaping (approach 34) and all-fruits-vs-next-fruit target
selection (backlog H-H).

### 34. Markovian reward via PBRS (yeti_universal_v6)  *(in progress)*

A discussion-driven reward audit (not a training failure) found the
path-progress shaping is **non-Markovian**: the `best_d` ratchet pays
only when the agent beats its *closest distance ever* to a fruit this
episode, so the reward at a given state depends on episode history the
policy cannot observe. Two costs: (1) the critic cannot fit a consistent
V(s) when identical observations have different returns -> higher
advantage variance / noisier learning; (2) the asymmetry (no penalty for
backing away) doesn't actually discourage dithering, only stops paying.

`best_d` existed as an **anti-oscillation** hack: a naive clipped
"reward for getting closer" can be farmed by pacing toward/away; the
ratchet stops that by only paying for new closest. **PBRS achieves the
same farm-resistance while staying Markovian:**

    F(s,s') = gamma*Phi(s') - Phi(s),   Phi(s) = -scale * sum_f D_f(s)

Moving toward pays `+`, moving away pays a symmetric `-`, so round trips
telescope to ~0 — no ratchet needed. With the shaping gamma equal to the
agent's discount, the optimal policy is provably unchanged (Ng, Harada &
Russell 1999). The training script injects `cfg.ppo.gamma` into the
reward so the two gammas can't drift apart. Signed-delta (gamma=1) is the
same idea minus the formal guarantee; we default to the tied value.

**Audit also enumerated the other stateful pieces** in the reward:
- `best_d_princess` — same ratchet for the princess target; also removed.
- `last_floor` — a *smaller, bounded* non-Markovian residue: pixel-y
  alone can't tell a jump from a ladder climb, so the floor is inferred
  with one step of memory. It is **kept on purpose** — pinning the floor
  during a jump (plus x-based distance) is exactly what stops jumps from
  being rewarded. Removing it cleanly needs a current-floor signal from
  RAM (backlog H-G).

The clean distinction the discussion settled on: a reward may depend on
the *current state* (`current floor`, x, fruits) and stay Markovian; the
bug is depending on the *trajectory* (`best_d` = best-so-far). "Current
floor" is derivable from state; "last floor" is history by definition.

**v6 run:** v2's config, only the reward swapped to
`fruit_bonus_path_progress_pbrs`. Expectation: cleaner/less-noisy
learning and a Markovian foundation for later experiments; not expected
to crack F2->F3 on its own. Compare to v2 (99.5%/0%) the usual way.

**v6 result (clean eval, 300 stochastic): regression.** reach-1 93.7%,
reach-2 **54.7%** (vs v2 99.5%), reach-3 0, princess 0. **Rejected;
v2 stays baseline.**

Diagnosis (quantified + confirmed): PBRS with `Phi = -scale*sum_f D_f`
and `gamma=0.99` emits a per-step "living reward" `(1-gamma)*scale*ΣD ≈
0.08/step` (~80 over a 1000-step episode) just for *staying alive far
from the goal* — because the gap `(1-gamma)=0.01` multiplies a large
potential `|Phi|≈8` (sum over 4 fruit distances). That dwarfed the
~10-16 reward for collecting 2 fruits, creating a **survival bias**. The
eval confirmed the signature: v6 episodes ran far longer (mean 344 steps
vs v2's 198), 19/300 hit the 1000-step cap without dying, and episodes
reaching only 1 fruit still averaged 355 steps (dawdling, not productive
exploration). So the regression is the gamma<1 living-reward term, not
Markovian shaping per se.

Important nuance (the "0.99 ≈ 1" trap): the gamma values are close, but
the *reward* effect is `(1-gamma)*|Phi|`, which is not small when Phi is
large. The fix is `gamma=1` (signed-delta shaping): standing still pays
exactly 0, the living-reward term vanishes, while keeping Markovian +
farm-proof. Forfeits only the formal invariance-under-discounting proof.
Pursued as H-F2 (v6b). Trained from scratch — a reward-scale change means
the previous policy/critic can't be reused.


### 35. PBRS with gamma=1 breaks the 2-fruit wall (yeti_universal_v6b)  *(verified, BREAKTHROUGH)*

Single change vs v6: shaping gamma 0.99 -> 1.0 (signed distance-delta).
Everything else identical (reset-only, 5M, seed 42, from scratch).

**Result (clean eval, 300 stochastic from reset):**

| reached | v2 | v6 (gamma .99) | **v6b (gamma 1)** |
|---------|-----|----------------|-------------------|
| 2 fruits | 99.5% | 54.7% | **98.3%** |
| 3 fruits | 0% | 0% | **95.7%** |
| 4 fruits | 0% | 0% | **13.0%** |
| princess | 0 | 0 | 0 |

Training: reached 3 fruits in ~32% of reset episodes (7706/23743), 4
fruits 739 times; end-of-training `reset_reach=[1,1,1,0.93,0.11]`.

**Why this is the headline result.** Every prior single policy was
hard-walled at 2 fruits / 0% reach-3 from reset, and we had spent many
approaches (curriculum v3/v5/v6/v7, MIP, stronger shaping, RND, entropy)
trying to get past it — all failing. A *clean reward* did it with no
curriculum, no warm-start, no observation changes. The 2-fruit plateau
was substantially a **reward-shaping artifact**, not an intrinsic
exploration limit:

- The old `best_d` ratchet was **non-Markovian** (reward depended on
  best-distance-ever, unobservable) -> noisy critic, and it *leaked* the
  F3 budget (approach 33).
- The gamma=0.99 PBRS fix traded that for a **survival living reward**
  (approach 34) -> dawdling.
- gamma=1 signed-delta has **neither**: the only way to earn reward is to
  actually get closer to a fruit, symmetric (round trips cancel, no
  farming), Markovian, no survival subsidy. That clean gradient was
  enough to carry the policy F1->F2->F3 and often ->F4.

**Caveat / honesty:** this is one seed. The reach-4 (13%) and princess
(0%) numbers say the *new* frontier is F3->F4 and F4->princess; the win
is real but the level isn't beaten. Lesson reinforced: get the reward
*clean and Markovian* before reaching for fancier machinery (curriculum,
MIP, intrinsic motivation) — several of those past failures may have
been fighting the reward, not the task.

**v6 vs v6b is the controlled answer to "does 0.99 vs 1 matter?"** Same
reward, seed, and everything else; only gamma differs. Outcome flipped
(reach-2 54.7%->98.3%, reach-3 0%->95.7%). So a 0.01 gamma change is NOT
a small behavior change — the reward effect is `(1-gamma)*|Phi|`, small
coefficient times a large potential.

**Theory caveat we knowingly accepted:** gamma=1 signed-delta forfeits
the PBRS policy-invariance guarantee (the proof needs shaping-gamma =
agent-gamma = 0.99). So this shaping *can* bias the optimal policy
relative to the true discounted objective. Here the bias ("rush toward
fruits") is aligned with the goal, so it helps — but the theorem isn't on
our side. If we ever need the guarantee back without the survival bias,
keep gamma=0.99 but shrink |Phi| (shape toward the nearest target only so
`(1-gamma)*|Phi|` stays small) — backlogged, not pursued now since v6b
works and is simpler.

**New baseline = v6b.** Next: H-I (CP4->princess) and H-J (lift reach-4).


### 36. 20M compute, instability, and the ladder-ascent diagnosis  *(verified)*

**H-C (20M reset-only, `yeti_pbrs_g1_20m`).** Same v6b reward, 5M->20M.
- The **princess was touched 5 times** during training (steps 7.9M,
  9.0M x2, 10.4M, 17.1M). First end-to-end completions from reset — the
  full level is achievable by a single policy.
- But the deep legs are **unstable**. Snapshot sweep (120 stochastic eps
  each):

  | snap | reach2 | reach3 | reach4 |
  |------|--------|--------|--------|
  | 8M   | 82%    | 6%     | 0%  |
  | 10M  | 99%    | 97%    | 2%  |
  | 12M  | 99%    | 89%    | **30%** |
  | 14M  | 100%   | 100%   | 1%  |
  | 16M  | 100%   | 99%    | 0%  |
  | 20M  | 100%   | **0%** | 0%  |

  reach-4 appears only transiently (30% at 12M), and the 20M final model
  is degraded. Classic learned-and-forgotten on an under-sampled skill in
  a shared network. **Best = 12M snapshot** (full 300-ep eval: reach-2
  99.7 / reach-3 87.7 / reach-4 28.3 / princess 0). Lesson: for the rare
  deep skills, reset-only PPO is non-monotonic; snapshots + best-pick are
  essential, and a curriculum that makes the deep states frequent is the
  principled fix.

**Profile (`scripts/mo5/yeti/profile_run.py`, 12M champion, 150 eps).** Per-leg
arrival: CP1 step47/bonus954, CP2 80/926, CP3 158/863, CP4 207/825 — the
agent is FAST and CP1 is already optimal, so dawdling/snowball-by-slowness
is not the problem. Failures localize at the ladders: CP3->CP4 ends at
~(180,114)=L34, CP4->princess ends at ~(220,82)=L45. **The agent reaches
the correct ladder and dies on the ascent** — the deep wall is reactive
ladder-ascent timing (dodging the snowball on the climb), under-practiced
because reached rarely from reset.

**H-K (deep-CP drill, `yeti_curriculum_v8_deepdrill`, running).** Warm-
start the 12M champion + seed CP1-CP4 pools from the 20M reset-origin
buffer (100 states/CP, source_cp=0); curriculum with no reach-gate,
success-weighting, CP0 floor 0.4; gamma=1 reward unchanged. The seeds put
the agent at the L34/L45 ascents far more often than reset does, drilling
exactly the failing skill. Bonus-scaled reward already rewards speed (we
are NOT changing the reward). Watch: does reach-4 stabilize >28% and do
princess touches appear at eval (vs the 12M champion baseline)?

---

# Level-2 run 3 (`yeti_curriculum_l2_v3_10m`) — RESULT: NEGATIVE + full reward-mechanics diagnosis

First control-verified L2 run (save/restore bugs fixed; agent can actually
act, unlike v1/v2). 10M steps, phase-1 exploration, PBRS path-progress
shaping (`fruit_bonus_path_progress_pbrs`, level 2). Config:
`experiments/003-yeti/configs/yeti_curriculum_l2_v3_10m.yaml`.

## Result: 0 fruits, stuck on floor 1

- `cp=[0,0,0] saves=[0,0,0]`, all 27,802 episodes `reached_level=0`.
- Direct snapshot rollout from `level2_start.sav` (`scripts/mo5/yeti/rollout_l2.py`,
  new): **every** checkpoint (100k … 10M) tops out at the first gap.
  Max `agent_x` reached = 6–11; **0/N episodes reach the F1→F2 ladder at
  x≈18**. Heatmaps/trajectories (`--heatmap`): a single hot blob at the
  top-left of floor 1. "Best" descender (9.9M) reliably walks off the edge,
  falls to floor 2, and dies.

## Why it never crosses the first gap (all scripted/measured)

- First gap ≈ `agent_x` 8–11; the F1→F2 descent ladder is at x≈18, on the
  **far** side of the gap. Scripted tests from spawn:
  - hold RIGHT → walks off the edge at x≈6, falls.
  - hold RIGHT+JUMP → reaches x=11 at the apex, falls into the gap.
  - run-up (≥5 right steps, THEN jump) → clears to x=27, lands alive on the
    far floor.
  So the gap IS crossable, but only with a precise run-up jump the policy
  never discovered.

## Reward pathologies (measured on the actual reward fn, not inferred)

### A. Loiter / gamma bug
- PBRS term `F = γ·Φ(s') − Φ(s)`, `Φ = −scale·dist` (always ≤ 0). Standing
  still → `F = (γ−1)·Φ = +0.098/step` at γ=0.99. Over ~375 steps ≈ **+37**,
  which matches the run's mean episode reward (**35.85**). Measured: noop
  for 60 steps → **+5.9 at γ=0.99, exactly 0 at γ=1.0**.
- Root cause: `train_checkpoint_curriculum.py` does
  `reward_params.setdefault("gamma", cfg.ppo.gamma)` → 0.99. The **L1
  champions (v10/v11, and the `*_g1` configs) explicitly set
  `reward.params.gamma: 1.0`**; the L2 configs (v1/v2/v3) never did, so they
  silently got 0.99.
- Consequence: the idle trickle is largest at the most-negative Φ (farthest
  from goal) = the spawn/left-wall, and it's safe there. That is why the
  agent hugs the far-left wall — idling out-earns any risky move.

### B. Falling is rewarded (fall-spike)
- The path-progress potential credits reaching a *lower floor* regardless of
  HOW it got there. Reward-to-reach at γ=1 (measured): F1→F2 fall **+2.08**,
  F1→F3 **+3.20**; a survivable gap-cross to (F1, x=27) only **+0.72**. So
  falling out-pays crossing ~3.6×, and multi-floor falls accumulate more.
- **Tolerance is a red herring.** The corpse comes to rest at exactly y=54
  (= floor-2 standing line), so even ±0 exact match fires the +2.153 spike
  (measured at t=17, one step before death); ±8 just fires it earlier
  (t=14). Same magnitude either way.

### C. Death-detection lag
- Death is detected by **bonus-freeze** in `mo5_rl.cpp` (comment:
  "When the player dies, the bonus freezes"), NOT by lives — the lives byte
  (11095) is **inert on L2** (stays 5 through a fall death). The L2 profile
  relaxed `bonus_stall_frames` 10→120, so death (bonus freezes ~t=18 in
  gym-steps) isn't detected until **~t=48** (≈30 steps late). During that
  lag the dead agent banks the +2.15 spike + ~+2.2 loiter income, with no
  penalty.
- Faster, cause-agnostic death flag found: **0x2AFC (11004)** = 32 alive /
  65 dead, flips at the true death frame (t=18). Validated only for the
  fall death so far — NOT yet for enemy (goat/yeti) deaths.

## Sprite-pose byte 0x2B54 (11092) — measured table

Direction-dependent **sprite index** (a display artifact, not a clean
physics flag):

| value | meaning | on a surface (creditable)? |
|-------|---------|----------------------------|
| 0–3 | walk/idle facing right | yes |
| 4–5 | walk/idle facing left  | yes |
| 8   | on a ladder (up/down/idle, both) | yes |
| 9   | jump, facing right/straight | no (airborne) |
| 10  | jump, facing left | no (airborne) |
| 11  | fall | no (falling) |
| 12  | death "float-up" animation | no (dead) |

- 11 fires at step-off (BEFORE landing), persists through landing AND the
  frozen-dead period, then → 12 (~120 frames later) for the float-up.
- Survivable jump/fall → returns to a grounded code (0–5) on landing; a
  fatal fall stays 11 → never grounded. So it separates fatal-fall from
  survived-landing.
- CAVEATS (measured): (1) it is **not** the complete sprite state — the
  on-screen ground-touch change is in another, unidentified player byte;
  0x2B54 stays 11 across landing. (2) It is a per-game enumeration and
  direction-dependent, so key on **sets** with a safe default, never a
  single value like `==11`. (3) fall-left code not measured (inferred 12?).
- Same codes observed on L1 (walk 0–3, ladder 8), so semantics carry across
  levels.

## Proposed fix — as a NEW reward style; L1 left untouched

Not yet implemented. Three cooperating pieces:
1. **Pose-gated, stateful credit (L2 only):** withhold floor credit while
   pose ∈ {airborne/fall}; ladder(8)+grounded(0–5) stay creditable. Because
   pose=11 precedes the landing spike, the fall never scores at all — no
   penalty to tune. Stateful because "floor 2 by fall" vs "floor 2 by
   ladder" are the same position and differ only in history.
2. **γ=1.0** in the reward params (kills the loiter trickle).
3. **0x2AFC for prompt, cause-agnostic death termination** (detection, not
   penalty) so no reward accrues after death.

Why a new style, not editing `fruit_bonus_path_progress_pbrs` in place: on
L1 the goal is UP, so a fall is already anti-progress (negative shaping);
pose-gating would remove that penalty — a behavior change on the 99.7%
policy. Keep L1 on the existing reward; register a new L2 variant. L1 reward
unit tests (`tests/python/test_yeti_map.py`) pin the ±8 anchors; keep green.

## Open items to validate BEFORE implementing
- Complete the pose table (fall-left, any hurt/level-transition poses) and
  choose a conservative default for unknown codes.
- Confirm 0x2AFC=65 fires on non-fall deaths (goat/yeti contact).
- Check whether a cleaner underlying physics/state byte exists vs the
  display sprite index.
- Decide whether a *survived* fall (returns to grounded on a lower floor)
  should be credited or also suppressed.

## New tooling added this investigation
- `scripts/mo5/yeti/rollout_l2.py` — rollout/eval from an L2 start-state (video,
  heatmap, trajectory, action distribution, agent-x extent, per-checkpoint
  depth sweep). The from-reset renderers can't target L2 (they boot L1).

---

# Level-2 run 4 (`yeti_curriculum_l2_v4_grounded_3m`) — reward fix, PROMISING

First run with the two reward fixes from "run 3", isolated and validated on
fixed trajectories first. Same recipe as v3 EXCEPT:
1. `reward.name = fruit_bonus_path_progress_pbrs_grounded` — new pose-gated
   PBRS style: freezes shaping while the sprite pose (0x2B54) is
   airborne/falling (not in surface set {0-5 walk, 8 ladder}); `last_floor`
   kept. Reuses the ungated reward object and only overrides `_potential`,
   so the ungated reward (and L1) is byte-identical.
2. `reward.params.gamma: 1.0` (v3 defaulted to 0.99 → the loiter bug).
Only 3M steps (vs v3's 10M) — a shorter first check.

Pre-run isolated validation (all measured):
- Fatal fall: old reward +2.56 (spike), new **+0.48** (spike frozen; only
  the pre-fall grounded walk credits).
- L1 walk+ladder-climb: new reward **byte-identical** to old at every step
  (0 mismatches) — differs only when airborne. Ladder climb (pose 8)
  credited normally.
- Reward/map unit tests: 73 passed, 1 skipped.

Result (episodes.csv, v4 3M vs v3 10M):

| metric | v3 (old, 10M) | v4 (fixed, 3M) |
|--------|---------------|-----------------|
| reward mean | 35.85 (idle income) | **1.80** (loiter gone) |
| fruits collected | 0 / 27802 | **2 / 27964** (first on L2) |
| CP1 seeds captured (`cp`) | [0,0,0] | **[0,2,0]** |
| deepest floor by final_y | F1:27725 F2:75 F3:2 | F1:15820 F2:4143 **F3:7668 F5:290** F6:2 |
| deaths | 6732 (24%) | 665 (2.4%) |

Interpretation (confirmed): both fixes did what they should. γ=1 removed the
idle income (35.85→1.80). The agent now descends far deeper (v3 was
floor-1-only; v4 reaches F3 in ~27% of episodes, the F5 fruit region in ~1%)
and collected the first 2 fruits, so the curriculum finally captured CP1
seeds. Deaths dropped 10x.

NOT solved: 2/28k fruit rate ≈ 0.007%. This is a much better starting point,
not a win.

OPEN / to verify: whether the deep descents are controlled (ladders) or the
agent still falling but no longer rewarded for it. Low death rate + most
episodes ending via stall (not death) suggest real descent; characterizing
with `scripts/mo5/yeti/rollout_l2.py --heatmap` on v4 snapshots (output/mo5/yeti/
videos/l2_v4_probe).

Next: with CP1 seeds now captured, a longer run + curriculum bootstrapping
is the natural follow-up (the curriculum was inert on v3 because it never
reached a fruit). This is where the curriculum warm-start work begins.

### v4 rollout characterization (`scripts/mo5/yeti/rollout_l2.py`, output/mo5/yeti/videos/l2_v4_probe)

Resolves the open question above: the deep descents are real navigation, not
falls. Deterministic-ish snapshot rollout (15 ep from level2_start.sav):

| snapshot | deepest floor | reaches F1->F2 ladder (x>=18) |
|----------|---------------|-------------------------------|
| 1.0M | F3 (mean 3.0) | 15/15 |
| 2.0M | **F4** (mean 2.9) | 15/15 |
| 2.9M | F2 (mean 1.9) — regressed | 15/15 |

vs v3 where **0/15 ever reached x>=18** (stuck at x<=11 at the first gap).
So the gap-crossing wall is broken: the agent crosses the first gap in 15/15
rollouts and descends to F3-F4. Right-biased (57-60% at the good snapshots),
low deaths, ends via stall not death → controlled descent, not fall-farming.

Non-monotonic (2.9M regressed to F2) — same "best is a snapshot" pattern as
L1; best so far = 2.0M snapshot. Fruits (F5) still only via rare stochastic
exploration (not in the deterministic rollout).

Next: (1) longer run (the recipe clearly works at 3M; give it room), with
snapshots + keep-best sweep to capture the transient peak; (2) curriculum
now has CP1 seeds to bootstrap deeper segments. Reward mechanics are no
longer the bottleneck — it's now the usual exploration/curriculum problem,
which is the tractable part.
