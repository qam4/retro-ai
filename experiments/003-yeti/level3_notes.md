# Level 3 — notes (route, obstacles, geometry)

Captured from the user's play knowledge + a RAM tilemap extract of
`output/mo5/yeti/level3/level3_start.sav`. L3 is substantially harder than L2:
a long sequential route with three NEW mechanics (escalator, compressor,
snowballs) on top of goats.

## Layout facts
- **1 fruit** (`fruits_total=1`), `fruits_remaining` at 0x2B2F goes 1->0 on
  pickup. Fruit sprite = "box 3" in `level3_annotated.png`: agent (x=14,y=64),
  pixel (56,64) (tilemap ids 18-21 @ ~0x2D6E — RENDER tile, not confirmed as
  the logic presence byte; for 1 fruit we can derive presence from
  fruits_remaining).
- **Princess: top-left.** (RAM entity read gave (16,30) — treat as
  approximate; it was read mid-intro.)
- **Player start: bottom-left**, x_ram=0, y=166 (`level3_start.sav`). The
  start is captured mid-intro: `princess_flag` (0x2B2A) = 1 and bonus frozen
  at 1000 for ~15-20 frames, then the flag clears to 0 and bonus ticks. NB:
  baseline `prev_princess` from the loaded value so the intro doesn't fire a
  false princess-touch; consider recapturing a clean post-intro start.

## Route (start -> princess)
1. Start bottom-left; go RIGHT, jump UP onto a higher platform.
2. Onto a platform with TWO ladders (left + right) -> up to a platform with a
   **goat** -> reach the top platform, avoiding the goat.
3. Go right, jump RIGHT onto the **ESCALATOR** (NEW): one big wall ringed by
   moving platforms circulating COUNTER-CLOCKWISE (down the left side, up the
   right side). RIDE a descending (left-side) platform for a while, then TIME
   a jump RIGHT onto a platform to the right of the wall. (Wall may be
   jump-through.)
4. Go DOWN a ladder to the right platforms, then UP the ladders **avoiding
   snowballs** (NEW for L3 — L2 had none).
5. From the top platform, go LEFT and jump up **5 ascending platforms** (each
   a bit higher than the last):
   - platform #3: a **COMPRESSOR** (NEW) — time it or get squished.
   - platform #4: the **FRUIT**.
   - platform #5: a ladder UP to the **princess** platform (top-left).

## New obstacles and how they interact with our setup
- **Goat** — lethal contact, as in L2. Death via 0x2AFC (cause-agnostic).
  Perceivable in the frame. Handled by curriculum + death-gate.
- **Snowballs** — lethal rolling hazard (jump/avoid). Same handling as goat;
  L1 had these, so nothing new mechanically.
- **Compressor** — periodic crush on platform #3. A timing gate; death via
  0x2AFC. Curriculum-seed just before it so the agent practices the timing.
- **Escalator (moving platforms) — THE novel hard part, and it breaks a
  reward assumption:**
  - The platforms MOVE and are sprites, NOT tilemap floors, so
    `extract_level_map` / the static nav graph do NOT model them. The
    `fruit_bonus_path_progress` potential ("graph distance to target") is
    therefore BLIND/misleading across the escalator section.
  - The agent RIDES a platform: it is grounded but MOVING WITHOUT ACTING.
    Our PBRS shaping (holds prev_phi across airborne, credits movement on
    grounded frames) would spuriously credit/charge platform-induced motion.
  - Implication: treat the escalator as a CURRICULUM-CARRIED, SPARSE-reward
    timing section — neutralize path-progress shaping there and rely on
    waypoint seeds + the sparse fruit/princess/survival signal. Do NOT trust
    the static graph through it.

## Extracted geometry (from RAM tilemap, base 0x2C27; settle 30 past intro)
Platforms (floor tile rows) — fragmented, not full-width:
  row5 y40 cols21-28 | row6 y48 cols0-8 | row9 y72 cols0-5 | row10 y80 cols7-16
  row13 y104 cols25-38 | row16 y128 cols27-39 | row19 y152 cols26-38
  row21 y168 cols9-13 | row22 y176 cols6-39 | row23 y184 cols0-4
Ladders (x_px): 24, 72, 96, 168, 232, 240, 280 (two share x=280 at
different heights). agent_x = x_px/4.
Sprite blocks (candidates; box 3 = FRUIT): 1=(28,16) 2=(44,16)
3=(14,64)=FRUIT 4=(36,112).  (1/2/4 are goats/princess/compressor/decor.)
NOTE: escalator platforms + compressor + snowballs are moving sprites and are
NOT in this static extract — they must be located by play/observation.

## Waypoints (CONFIRMED) — one per ladder, at the arrival end
Placement rule (agreed): a waypoint seeds the START of the next hard segment,
so put it where the agent ARRIVES: the down-ladder -> BOTTOM, every climb
ladder -> TOP. Ladder climbs themselves are trivial; the hard bits are the
inter-ladder gaps (jumps / goat / snowballs / escalator / compressor). With
H-AK the waypoint count is nearly free, but arrival-end-only is cleaner and
avoids wasting steps re-climbing. Implemented via `LevelMap.waypoint_ends`
(opt-in; L1/L2 keep both ends). `yeti.waypoints(3)` returns:

  Lgoat_a_top (16,104), Lgoat_b_top (22,104)  -- the two ladders to the goat
      platform (Lgoat_a left / Lgoat_b right)
  Ldown_bot   (40,184)                         -- POST-ESCALATOR down ladder
  Lsc1_top (58,176) -> Lsc2_top (68,152) -> Lsc3_top (56,128) -> Lsc4_top
      (68,104)                                 -- snowball climb, bottom->top
  Lprincess_top (4,48)                         -- final ladder up to princess

Escalator has NO ladder/waypoint but is BRACKETED: Lgoat tops seed just before
it, Ldown_bot just after — so the agent seeded at a Lgoat top practices
crossing the escalator to reach Ldown_bot (captured on success).

## Build plan
1. `LEVEL3` in `yeti_map.py`: floors + ladders from the extract, fruit at
   (56,64)px, princess top-left. Accept that the escalator span is a
   graph "gap" (no static nodes there).
2. `FRUIT_PRESENCE_BY_LEVEL[3]` (1 fruit) / `fruits_total=1`; profile
   `mo5_yeti_fruit_level3.yaml`; curriculum config with the v10 recipe
   (WP curriculum + phase-2 anneal + H-AK + H-AB + `defer_fruit_credit`).
3. Reward/escalator: neutralize path-progress shaping across the escalator
   region (or gate it to graph-covered floors only); lean on waypoint
   seeding through the escalator/compressor gates.
4. Expect MANY more waypoints (one per gate) and more compute than L2.

## v1 result (yeti_curriculum_l3_v1_15m — from-scratch phase-1, 15M)
Completed 6h44m, exit 0. Outcome: reaches segment 1 only, WALLED by the
escalator.
- Only 2 waypoints ever captured: Lgoat_a_top / Lgoat_b_top (100 each, but
  goal_score 0.00 — even seeded at the goat platform it never gets further).
  NO Ldown_bot / Lsc* / Lprincess ever captured -> the escalator was crossed
  0 times in 15M. fruit collected 0 (success 0->1: 0%, reset_reach princess 0).
- 234k episodes, ALL end in death, reached_level 0. Death clusters:
  ~half near START (x~8-12,y~168-176) — segment 1 not solid from scratch;
  ~1/3 at/above the GOAT platform (x~16-24,y~88-112) — dying on the jump
  toward the escalator.
- Takeaway: from-scratch phase-1 + curriculum DOES bootstrap segment 1, but
  the escalator (moving-platform timing, no shaping signal) is a hard wall,
  as predicted. To make progress on the LATER segments (segment-first plan)
  we need post-escalator seeds, which we cannot reach by play. Candidate v2
  levers: (a) warm-start L2-v10 skills to make segment 1 solid + maximise
  escalator attempts; (b) obtain a post-escalator seed (human play-through
  capture — clean/on-distribution; RAM-poke rejected) and train the L2-like
  post-escalator segment (snowball climb + fruit + princess) in isolation;
  (c) attack the escalator directly from the Lgoat_top seeds (hard — no
  shaping guidance for the moving-platform timing).

## v5 result (yeti_curriculum_l3_v5_phase1_15m — from-scratch phase-1, 15M)
Completed, exit 0. **SOLVED THE ESCALATOR** (the v1 hard wall) after pose-13
seeding + the escalator crossing discovery (run right + jump into wall at
x32 -> pose 13 ride down y94->158 -> jump right at ride_y~146-158 -> land
ELAND; verified in `debug/escalator_crossing.mp4`). From the goat platform
the agent crossed the escalator ~81%. BUT the policy OSCILLATED (n_steps=16,
no target_kl = the L1/L2 destructive-update pattern): a from-reset snapshot
sweep showed BR/Lsc1 reachable ~62% at the 10.5M snapshot but ~1% at the
final step (a trough). Kept the **10.5M snapshot** as the champion
(`output/mo5/yeti/champions/l3_v5_10p5M/`, 10.5M weights + v5 seed pools) to
warm-start the anneal. Best captured chain from reset: escalator crossed,
descends to BR/Lsc1.

## v6 result (yeti_curriculum_l3_v6_anneal_15m — phase-2 anneal, 15M)
Completed 5h48m, exit 0. Warm-start (weights-only) from the v5 10.5M
champion; n_steps 16->512, target_kl=0.05, `ladder_segment_shaping: true`.
Trained on the OLD map (no jump-edges), so no upward gradient past the
snowball climb (fruit/princess unreachable by design this run).

**The anneal worked — the oscillation collapsed and the climb chained from
reset all the way up the snowball staircase to its second-from-top rung.**
Final training `wp_reach` (P(reset-origin episode reaches WP), decaying EMA):
Lgoat_a 0.97 / Lgoat_b 0.96 / Ldown_bot 0.89 / Lsc1 0.89 / Lsc2 0.89 /
Lsc3 **0.74** / Lsc4 **0.00** / Lprincess 0.00. (Lesc_top reads 0.00 — it is
a board point ridden THROUGH via pose 13, not a discrete reached WP.) Vs
v5-final's ~1% BR, this is real chaining from a cold reset.

**Keep-best snapshot sweep** (`climb_snapshot_sweep.py`, 30 snapshots stride
500k, 40 eps each, max 500 steps, from reset — floor labels BOTTOM/BR/SN1/
SN2/SN3 map onto the Ldown/Lsc1..Lsc4 staircase):

```
  step     BOTTOM  BR   SN1  SN2  SN3
   500000   100   65    2    0    0    early: bottom only, no climb
  5500000   100   95    2    0    0    BR peak, SN1 still noise
  9500000    32   30    0    0    0    mid-run trough
 12000000    68   42   32    0    0    SN1 emerges
 12500000    58   45   38   15    0    SN2 first appears
 13000000    60   55   48   12    0
 13500000    38   38   35   20    0
 15000000    55   50   45   38    0    FINAL = deepest chain
```

Findings:
- **The final (15M) snapshot is the best this time** — unlike v5 (final
  trough), the anneal held to the end. Clean monotone funnel from reset:
  BOTTOM 55 -> BR 50 -> SN1 45 -> SN2 38. SN1/SN2 chaining simply did not
  exist before ~12M. n_steps=512 + target_kl=0.05 fixed the oscillation
  (same recipe that solved L1 H-V / L2 H-AL). Champion = **v6 15M**.
- **New wall = SN2->SN3** (top of the snowball climb, Lsc4 top, y86): SN3 is
  0% across EVERY snapshot, matching the final `wp_reach` Lsc4=0.00. Nothing
  gets past the second-from-top rung.
- fruit/princess 0% everywhere as expected (old map, no jump-edges -> no
  gradient past the climb).

Takeaway: escalator SOLVED and the snowball climb learned up to Lsc3 from a
cold reset — a large jump over v1 (walled at the escalator) and v5
(oscillated, ~1% BR at final). Next wall is the single Lsc3->Lsc4 rung.

## SN2->SN3 stall — ROOT CAUSE (diagnosed on the v6 15M champion)
The 0% SN3 is a **1-pixel ladder-mount window on Lsc4** (same game mechanic
as L2's L34, notes section 22.1). Verified end-to-end:

- Geometry: `Lsc3` deposits the agent at the LEFT end of SN2 (ram x58, y110);
  the `Lsc4` ladder up to SN3 is at ram **x70** (centre_px 288); SN2 extent is
  ram 54-80. So the agent must traverse 12px right along SN2 and mount at x70.
- Reward gradient is CORRECT (not a bug): graph path-distance to princess
  decreases monotonically SN1 448 -> SN2-left 400 -> SN2-at-ladder(x70) 352 ->
  SN3 328 -> A1 216. Shaping pulls right-then-up; going back down `Lsc3`
  is against the gradient. (`debug/l3_sn2_gradient.py`.)
- Snowball is DODGEABLE, not the wall (user confirmed from the video: the
  agent jumps the snowball, survives, then walks PAST the ladder and even
  climbs back down `Lsc3`).
- **The wall is the mount mechanic.** Driving the model to the ladder alive
  then forcing UP (`debug/l3_lsc4_mount.py`, exact-x match, 150 eps each):
  x68 0/19, x69 0/4, **x70 4/4 (3 straight to SN3 y86)**, x71 0/3, x72 0/6.
  So the ladder mounts ONLY at exactly ram x=70; one pixel off, UP is a
  no-op. While dodging the snowball the agent almost never stops on that
  exact column, so from the policy's view the ladder "usually doesn't work."

Method caveats found & fixed during this dig (both invalidated only the
scripted probes, never the model rollouts): (1) probe used 5 settle-noops
(the L2 H-AB doom bug) — training uses `restore_frame_stack` + 0 settle;
(2) joystick action axis order is `[vertical(1=up,2=down), horizontal
(1=right,2=left), fire]`, not `[horiz,vert,fire]`.

Implication: this is a known-HARD-but-LEARNABLE motor skill (L2 cracked the
L34 1px window via waypoint seeding + compute), NOT a reward or
unavoidable-hazard problem. More compute alone is weak unless the agent gets
many reps of "stop at x70, press up." The committed jump-edges (SN3->A1..A5)
do NOT help the mount itself — they only add downstream pull ONCE SN3 is
reached, which would reinforce the rare successful mounts.

## v7 plan (revised after the SN2->SN3 diagnosis)
The bottleneck is the 1px `Lsc4` mount at x70 under snowball pressure. Levers,
in priority order:
1. **Reverse-curriculum the mount**: make sure `Lsc4_top` (SN3) and A1+ seeds
   accumulate (they were only 7/100 at v6 start) so the agent practices the
   downstream ascent from just ABOVE the wall and the fruit/princess reward
   compounds back — the same bootstrap that broke L2's L34. Warm-start v6 15M
   + committed jump-edges (so SN3->A1..A5 has gradient) + real compute.
2. Consider whether frame_skip (4) makes hitting the exact x70 column
   effectively impossible from a walking approach (the agent steps ~N px per
   action) — if so, the mount may need a finer approach, not just more reps.
3. Do NOT widen tolerances to "fix" this — the 1px window is the GAME's, not
   our resolver's; the agent must learn the precise positioning.
Use `training.output` for the kiro-monitor dir (not `debug/`).

## v7 result (yeti_curriculum_l3_v7_jumpedges_15m — warm-start v6 + jump-edges, 15M)
Completed 6h28m, exit 0. Warm-start weights from the v6 15M champion + v6 seed
pools; LEVEL3 jump-edges map active (SN3->A1..A5 connected).

**SN2->SN3 wall SOLVED.** The 1-pixel `Lsc4` mount got learned: `Lsc4_top`
(SN3) reset-origin wp_reach 0.00 (all of v6) -> **0.77**, and its seed pool
filled 7 -> **100**. The whole climb tightened (Lsc3 0.96, Lsc2 0.97, Lsc1
0.99). Confirms the user's call: reward map correct + more training composes
the mount, same as L1. Progressed steadily (Lsc4 reach 0.69 @7.5M -> 0.79
@11M -> 0.77 final).

**But NO fruit grab in 15M** (`cp=[0,0]`, success 0->1 = 0%, reset_reach[1]
= 0.00). The new wall is the **A1->A5 ascent** (SN3 -> compressor on A3 ->
fruit on A4 -> A5). Evidence:
- `Lsc4_top:d0x116r38596` — SN3 reached constantly but a ~330:1 reject ratio:
  almost every SN3 arrival is a doomed state (dies within the survival window
  before progressing). So the SN3 perch is hazardous AND the A1 jump off it
  mostly kills.
- `Lprincess_top:d48x0r0` — closest the agent EVER came to the final ladder is
  48px; it crept 64->48px over the run then plateaued. So it gets partway up
  A1..A5 but never near the top, and never grabs the fruit.

Incentive is NOT the problem (verified `debug/l3_ascent_gradient.py`): graph
path-distance to princess decreases monotonically every rung SN3 304 -> A1
212 -> A2 180 -> A3 144 -> A4/fruit 92 -> A5 44. Each successful step up pays
positive PBRS. The blocker is that A1-A5 are tiny platforms (A1 = 4px wide)
requiring precise jumps + compressor timing, the shaped reward only lands on
COMPLETED jumps (airborne freeze; a near-miss falls -> death gate -> 0), and
**A1-A5 have NO waypoints** so the segment is unseedable — the agent can only
practice it by surviving the whole chain to SN3 first, which mostly ends in
death (the 38k rejects). Pure on-distribution, hence slow/stalled.

## v8 plan (next) — jump-edge waypoints for A1..A5
Add jump-WP support so the A1..A5 platform landings become capture/seed/reach-
tracked waypoints (the graph already has the J-node endpoints J10_11..J14_15).
Benefits: (1) observability — per-platform wp_reach up the ascent; (2)
seedability — reverse-curriculum reps of the compressor/jump timing instead of
requiring survival of the whole chain first (the bootstrap that broke L2's F3
goat + L34). Default-off so L1/L2 stay byte-identical. Then warm-start from the
v7 best-SN3 snapshot + these WPs. This is the clearest lever; more compute
alone on v7's recipe is weak (A1-A5 stays unseeded).

## v8 result (yeti_curriculum_l3_v8_jumpwp_15m — A1..A5 jump-WPs, 15M)
Completed 6h20m, exit 0. Warm-start v7 bestSN3 (12M) + jump-WP code live
(A1..A5 now in `yeti.waypoints(3)`, seed/reach-tracked, NOT reward targets).

**NEGATIVE — the jump-WP reverse-curriculum could not bootstrap.** A1..A5 were
NEVER captured (0 seeds each) because the agent never LANDS on A1 even once in
15M. Final: `cp=[0,0]` (no fruit), SN3 (`Lsc4_top`) reset-reach 0.67, A1..A5
wp_reach all 0.00. `wp_near` closest grounded approaches: A1:d8 A2:d16 A3:d24
A4:d30 A5:d38 — i.e. it gets within 8px of A1 but never on it (tol 2), so the
capture-on-reach seed pool stays empty -> no reverse-curriculum. Chicken-and-egg
exactly as flagged for `Lesc_bot`: can't seed A1 until the agent reaches A1,
and it can't.

Root-cause dig (v8 champion, seed from `Lsc4_top`/SN3, faithful restore):
- 0/30 reach A1; best GROUNDED y stuck at 86 (SN3; A1 is y78) — never ascends.
- Deaths cluster at **(x68, y84, walking)** — killed mid-traverse on SN3 while
  heading LEFT toward the A1 jump-off (SN3 has its own snowball; `Lsc4_top`
  reject count r46943 corroborates: most SN3 arrivals die fast). A few fall off.
- Scripted brute-force (walk to SN3 left edge ram48-56 then jump-left, several
  delays; `debug/l3_sn3_a1_feasible.py`): 0/25 land A1 under EVERY plan
  (best_y stays 86). Crude scripts failed on the SN2 snowball too (where the
  model COULD dodge), so this isn't proof of impossibility — but neither 15M of
  training nor brute-force has landed a single A1.
- Geometry: SN3 floor10 ram50-77 y86; A1 floor11 ram42-45 (4px) y78; jump edge
  is a small up-LEFT hop from SN3's left edge (~ram48) to A1's right edge
  (~ram44), Δ~16px left + 8px up, onto a 4px platform, past an SN3 snowball.
- Video for eyeballing: `debug/l3_sn3_a1_best.mp4` (longest SN3 survivor).

Takeaway: SN3->A1 is the wall — a precise up-left jump onto a 4px platform
while dodging an SN3 snowball, AND unseedable (can't capture A1 without first
landing it). v9 options (needs a decision, see below).

## v9 (DECIDED + LAUNCHED) — launch-pad WPs + gentle re-heat
Two coordinated changes, both aimed at the unseeded SN3->A1 wall:
1. **Launch-pad waypoints** (`jump_waypoints` now emits `<name>_launch` on each
   jump's DEPARTURE platform). The key one, `A1_launch` = SN3's SAFE LEFT edge
   (ram48, floor10), is capturable (on SN3, which the agent reaches), so once
   captured it SEEDS the safe jump-off and the agent practises SN3->A1 directly
   (snowball-free) instead of dying on SN3's right / retreating down the ladder.
   Landing A1 then captures `A2_launch`, etc., up to the fruit on A4. Launch pads
   are SEED-ONLY (not reward targets); L1/L2 emit none (byte-identical).
2. **Gentle re-heat** (ent_coef 0.01->0.03, target_kl 0.05->0.10). Rationale
   (user's concern): v6->v7->v8 were ALL low-temperature anneal for ~45M steps,
   which converges but under-explores. That's fine for SEEDED skills (v7 learned
   the 1px SN3 mount from the Lsc4_top seed) but the A1_launch bootstrap needs a
   rare EXPLORATORY dodge to SN3-left to capture it the first time — a cold
   policy may never produce that. Re-heat restores exploration while staying
   self-cooling via target_kl (the intended anneal SCHEDULE: re-heat to escape a
   new plateau, then cool). NOT a dead local min — warm-start keeps the SN3
   skill; this just re-enables escaping to the next one. Risk: hotter policy can
   transiently degrade SN3 -> mitigated by 100k snapshots + keep-best sweep.
Warm-start from v8 final. Config:
`experiments/003-yeti/configs/yeti_curriculum_l3_v9_launchwp_15m.yaml`.

### v9 interim (BREAKTHROUGH on the ascent; lower chain collapsed)
By 50% (7.5M): the reverse-curriculum bootstrapped the ENTIRE upper level.
- `A1_launch` captured -> seeded the SAFE SN3-left jump-off -> the whole ascent
  chained. All 19 WP pools populated; `cp=[0,100]` (fruit grabbed reliably);
  `Lprincess_top` goal_score ~0.96 => the PRINCESS is touched ~96% from its seed.
  First fruit grabs + princess touches ever on L3.
- COST (expected, from the re-heat): the from-COLD-RESET lower chain collapsed
  (Lsc3 0.82@10% -> 0.04@50%, Ldown 0.9 -> 0.05). reset_reach still [1,0,0].
  NOT under-practice: `pick_start` weights CP0 by 1-goal_score and the fruit-CP
  is reach-gated out, so reset (CP0) was already ~60% of episodes; the collapse
  is the HOT updates (target_kl 0.10, ent 0.03), not starvation.
So v9 = "acquire the ascent" (done). Next = COOL to compose (v10).

IMPORTANT (pools persistence): `checkpoints.pkl` was historically saved ONLY on
normal completion, so v9 must FINISH to persist its (invaluable) ascent pools —
killing it mid-run would lose them. FIXED for future runs: added
`PoolSaveCallback` (saves pools every 1M steps, atomic temp+replace) +
`save_to_disk` now writes atomically. (Edit applied AFTER v9 launched, so it
does NOT affect the running v9 — v9 still only saves at the end.)

### v10 (COMPOSE phase) — cool the re-heat
One lever: cool ent_coef 0.03->0.01, target_kl 0.10->0.05 (back to the anneal),
so the hot updates stop breaking the lower chain and the now-seeded full route
composes from reset. Warm-start WEIGHTS from the best combined-reach v9 snapshot
(from-reset sweep picks it; lower chain intact + upper skill — likely early-ish
v9, else v8-final) + POOLS from v9 FINAL. Build champion dir
`output/mo5/yeti/champions/l3_v10_base/` = {chosen weights final_model.zip, v9
final checkpoints.pkl}. Config:
`experiments/003-yeti/configs/yeti_curriculum_l3_v10_compose_15m.yaml`.
Watch reset_reach[1] off 0, from-reset chain (Lsc4 -> A1..A5 -> Lprincess_top),
success 0->1, and a princess-from-reset touch.

## v9 options (SN3->A1 is the wall; jump-WP seeding can't bootstrap it)
1. VERIFY the route with the user (video) — confirm SN3->A1 is the intended
   path and the jump is as modelled (the user knows L3; the agent can't see it).
2. Human-demo seed for A1 (record a played state ON the A1 platform -> seed the
   pool -> reverse-curriculum from there). This is the honest unblock for the
   chicken-and-egg; needs the "record a session" tooling and revisiting the
   no-offline-injection stance (a real PLAYED state is on-distribution, unlike a
   RAM-poke).
3. Boost exploration specifically at SN3 (entropy schedule / longer at SN3
   seeds / strengthen the near-A1 shaping) to get a FIRST A1 landing by luck,
   then let capture bootstrap. Weak given 15M got 0.
4. Re-examine the SN3 snowball dodge as its own sub-skill (like SN2) — if the
   agent can't survive SN3 long enough to set up the jump, fix survival first.

## v10 result (compose phase — cooled; 15M, exit 0, 6h06m)
Warm-start DECOUPLED: v8-final weights + v9-final pools (champion dir
`l3_v10_base`), cooled to ent 0.01 / target_kl 0.05.

**Cooling restored the lower chain (best ever from reset) but did NOT compose
the ascent.**
- From RESET: Lgoat 1.00 / Ldown 0.97 / Lsc1 0.97 / Lsc2 0.96 / Lsc3 0.96 /
  **Lsc4 (SN3) 0.67** — the best lower chain of any run (v9's re-heat damage
  fully repaired, confirming the cool-to-compose half of the anneal schedule).
- But EVERY A-waypoint reads 0.00 from reset; `reset_reach[1]=0.00`,
  `success 0->1: 0%`. No fruit and no princess from a cold start.
- The SKILL is intact and MAINTAINED (from seeds): pools all full and GROWING
  (A3 43->92, A1_launch 77->99, Lprincess_top 13->25; A1 captured 2142x during
  v10), `Lprincess_top` goal_score 0.95, A4/A5 0.50 => fruit + princess still
  reliably done FROM SEEDS.

**ROOT CAUSE of the composition gap: surviving SN3 on arrival from reset.**
`Lsc4_top: x2916 admitted vs r46867 rejected` (~16:1). Reset episodes DO reach
SN3 (0.67) but almost all die within the survival window — before they can
traverse SN3 to `A1_launch` and start the ascent. Seeded episodes start ON SN3
already stable and proceed fine, which is exactly why the seeded skill looks
solved while the cold-start chain stops dead at SN3. Late-run SN3-from-reset
also oscillates 0.29-0.67 (ended 0.67).

So the remaining wall is a SURVIVAL/hand-off problem at exactly one spot (arrive
at SN3 -> stay alive -> cross to the safe left edge), not a missing skill.
Candidate v11 levers (undecided):
- (a) Make the SN3 hand-off practicable: seed from `Lsc4_top` states that are
  ARRIVAL-like (as a reset episode arrives, mid-climb/hot) rather than the
  stable grounded ones, so the agent practises surviving the hot arrival.
- (b) Attack the SN3 snowball dodge as its own sub-skill (the SN2 pattern) —
  the 16:1 reject ratio says arrival is near-doomed; find whether a dodge exists
  from the arrival state and whether the timing is learnable.
- (c) More compute at the cooled recipe (SN3 0.29->0.67 was still trending up;
  the chain may extend if SN3 survival keeps improving).
- (d) Revisit whether the WP survival gate (`min_survival_steps=30`) is
  filtering out precisely the hot-arrival states we need to learn from (the 46867
  rejects) — the H-R leniency question, now with a concrete case.

## CORRECTION + real diagnosis (post-v10 link measurements)
Two earlier claims in this file were WRONG; the per-link measurements
(`debug/l3_link_matrix.py`, `debug/l3_link_over_snapshots.py`) correct them:

1. **"v9 acquired the ENTIRE upper level" — WRONG.** That read pool sizes +
   goal_scores. In truth the goal_scores (A4 0.45 / A5 0.50 / Lprincess 0.95)
   come from seeds ABOVE the hard jumps and were INHERITED (stale) by v10 via
   checkpoints.pkl — they decay only when that pool is sampled. Only the TOP
   segment (A3/A4 -> A5 -> princess) is solid.
2. **"The upper skill is carried by the SEEDS" — WRONG.** Seeds supply the START
   STATE; the policy still must know how to execute. v10 (v8 weights + v9 pools,
   cooled) never re-learned the ascent: best snapshot only 5% on A1_launch->A1.

**Measured per-link reach (v9-final / v10-final, 20 eps, faithful restore):**
```
  Lsc4_top  -> A1_launch    0% / 0%    (100% death)  <- BROKEN: SN3 traverse
  A1_launch -> A1           5% / 0%                  <- weak JUMP
  A1        -> A2_launch   60% / 20%                 (walking: ok)
  A2_launch -> A2           0% / 0%                  <- weak JUMP
  A2        -> A3_launch   80% / 95%                 (walking: ok)
  A3_launch -> A3           5% / 0%                  <- weak JUMP
  A4_launch -> A4          65% / 25%
  A5_launch -> A5           0% / 15%                 <- weak JUMP
```
Pattern: WALKING links are fine (60-95%); the JUMP links (`X_launch -> X`) are
0-20%; and `Lsc4_top -> A1_launch` (walk across SN3 under the snowball) is 0%.

**KEY MEASUREMENT — the jumps are trivially EXECUTABLE, so this is a
credit/exploration problem, not a mechanics problem.**
`debug/l3_a1_jump_bruteforce.py`: from A1_launch seeds, an immediate jump-left
lands A1 **10/10** for EVERY hold length (2/4/6/8), and 9/10 even after a 5-step
wait. So no precise timing is required — yet the policy only manages 5-20%.

**ROOT CAUSE FOUND: the ascent had NO mandatory reward milestones.** L3's
`reward_waypoints` covered the whole lower chain (Lgoat, Ldown_bot, Lsc1..Lsc4)
— which IS mastered — and then jumped straight to `Lprincess_top`. A1..A5 had
none, so the ascent's only signal was the diffuse fruit/princess path-distance.
The mastered part of the route is exactly the part with milestones.

## v11 — add ASCENT milestones (A1, A2, A3, A5) to the reward
FIX (code, L3-only): added the jump-edge LANDING nodes to
`LEVEL3.reward_waypoints`: `J10_11_b` (A1), `J11_12_b` (A2), `J12_13_b` (A3),
`J14_15_b` (A5). Verified each resolves to EXACTLY its A-waypoint position
(A1 (44,78), A2 (38,70), A3 (32,62), A5 (10,54)). A4 deliberately omitted (the
FRUIT F1 is already a mandatory target on that platform — no double-count).
L1/L2 keep `reward_waypoints=None` => byte-identical. 328 tests pass (the one
failure, test_no_shared_reward_fn, is pre-existing: it looks for
`scripts/train_segment.py`, which lives at `scripts/mo5/yeti/`).
Now each completed ascent jump banks an explicit milestone, the same mechanism
that made the lower chain work.
Recipe: warm-start v10-final weights (BEST lower chain: SN3 0.67 from reset) +
v10 pools (all 19 full), MILD re-heat (ent 0.02, target_kl 0.07) — enough
exploration to discover jump-left at each launch pad, less destructive than v9's
0.03/0.10 which wrecked the lower chain. NOTE the reward changed, so the
warm-started critic is partly stale (pitfall #4) — expect an early dip.
Config: `experiments/003-yeti/configs/yeti_curriculum_l3_v11_ascentwp_15m.yaml`.
WATCH: the JUMP links (A1_launch->A1 etc.) rising, then `reset_reach[1]` and
from-reset A* wp_reach. Still-open risk: `Lsc4_top -> A1_launch` (0%, the SN3
snowball traverse) may need its own treatment even with milestones.

## v13 result (yeti_curriculum_l3_v13_seedfix_15m — seed-milestone fix, 15M)
Completed 6h33m, exit 0. v12's recipe restarted on the reworked logs.

**Best from-reset chain of any run.** Lgoat 1.00 / Ldown 0.99 / Lsc1 0.99 /
Lsc2 0.96 / Lsc3 0.95 / **Lsc4 (SN3) 0.81** (previous best: v10's 0.67). Note the
mid-run dip recovered on its own — at 8M the chain had eroded to Lsc3 0.63 /
Lsc4 0.27 and I recommended cutting the run; it came back to its best numbers in
the last third. These traces oscillate; do not act on a mid-run trend.

**Seeded practice improved everywhere** (the seed-milestone fix paying off):
`Lsc3_top` prog 0.02 -> 0.72, `A1_launch` 0.74, `A1` 0.79, `A3` 0.77,
`Lprincess_top` 0.86 (pool 100). Pairwise jump links, same definition as the
pre-fix probe: A1_launch->A1 5% -> 24%, A2_launch->A2 0% -> 32%,
A3_launch->A3 5% -> 27.5%, A5_launch->A5 0% -> 28%.

**But nothing above SN3 from reset** (`cp=0`, `reset_reach[1]=0.00`), and the gap
is at ONE spot, quantified from episodes.csv (last 20%):
```
SN3 arrivals from a SEED  -> onward 5.81%   (30/516)
SN3 arrivals from RESET   -> onward 0.03%   (2/7683)
```
A ~200x gap at the same location.

### SN3 debugged properly (several of my hypotheses refuted)
- **Save/restore is faithful.** Capturing a WP state live, reloading it and
  replaying the IDENTICAL action list gives identical survival: 0/27 mismatches
  (`debug/l3_capture_replay.py`). Seeding works as advertised.
- **The seeds are survivable.** Random action search from `Lsc4_top` seeds
  survives the full 30-step gate in 14-22 of 300 tries (~5-7%). So the admission
  gate is HONEST and there is no seeding bug — the earlier "these seeds are
  doomed" and "policy regression" stories were both wrong.
- **Ruled out** along the way: the death byte 32 is the normal ALIVE value; SN3
  seeds load at exactly (70,86) pose 8 (the reach moment, not later); no constant
  action survives (RIGHT is best at median 19 steps).
- **At SN3 the policy is no better than random** (~5-7% either way). It has
  learned nothing there, despite thousands of arrivals.

### ROOT CAUSE: the reward paid for ARRIVING, not for surviving
Measured with the real reward from an SN2 seed:
```
climb -> touch SN3 -> DIE        +5.04
wait at SN2 for a safe phase      0.00
climb -> SN3 -> traverse onward  +10.32
```
The death gate only suppressed shaping ON the fatal step; progress banked EARLIER
was kept. With an SN3 arrival lethal ~99.8% of the time, arriving immediately
strictly dominated waiting, so the policy correctly learned to arrive and die —
and never learned the dodge. `defer_fruit_credit` had protected FRUITS from
exactly this since L2 (H-AL); milestones and path-progress never got it. Fixed by
`credit_requires_survival` (one rule for every target type) -> the reckless line
becomes 0.00 while the surviving line is unchanged.

## v14 (RUNNING) — credit_requires_survival
One lever vs v13; warm-start v13-final + v13 pools. Config:
`configs/yeti_curriculum_l3_v14_survivalcredit_15m.yaml`. Also active (landed
with the target/ladder work, not a variable): the progress ladder gives L3 13
rungs instead of 1, so `success=[N->N+1]` finally carries information — it read
88-95% for rung 0->1 within the first 200k steps, where the old single-rung
version sat at 0% all run.
WATCH: does the agent start LOITERING at SN2 rather than climbing immediately
(the behavioural prediction), then `Lsc4_top` prog off 0.11, then
`reset_reach[1]` off 0.
EMULATOR: this run and v13 share the current core
(`retro_ai_native...so` sha256 fdd7e006..., mtime 2026-07-02), so v13-vs-v14 is a
valid comparison. v14's env.json predates the manifest `native` block (1bccba8);
later runs record this automatically. Every L3 run post-dates the state-restore
fix, so L3 hazard timing is LIVE — see core_provenance_2b0a45d.md, and note this
is why L3 feels harsher than L1/L2 ever did.

## v14 result (credit_requires_survival — NEGATIVE, and why)
`yeti_curriculum_l3_v14_survivalcredit_15m`, 15M, exit 0, 6h08m. One lever vs
v13: `credit_requires_survival`. Warm-started from v13 final (the best from-reset
chain of any run) + v13 pools.

**The from-reset chain collapsed.**
```
                v13 final   v14 final
Lgoat_a_top        1.00        0.99
Ldown_bot          0.99        0.96
Lsc1_top           0.99        0.00   <- lost the whole snowball climb
Lsc2_top           0.96        0.00
Lsc3_top           0.95        0.00
Lsc4_top (SN3)     0.81        0.00
```
Mean episode reward 38.4 -> 21.2; princess touches (last 20% of episodes)
89 -> 9. So v13 REMAINS the champion; v14's weights are worse (its pools are
fine, and richer).

**Mechanism.** With gamma=1 the episode's shaping TELESCOPES to
Phi(end) - Phi(start), so refunding it on death makes the whole shaping term
exactly zero for any episode that dies — and on L3 that was 17,024 of 17,033
episodes. Only the sparse terms survived. The agent therefore lost the
directional signal that taught the route and forgot the climb, while keeping what
sparse reward it could reach. The intervention removed the reckless-arrival bonus
by removing progress credit altogether, on a level where dying is the NORMAL
ending.

**Why the implementation was wrong (the substitution).** The stated goal was to
give milestones what `defer_fruit_credit` gives fruits. But those defer a SPARSE
payment: a fruit has a discrete reward at pickup, so "hold it until a
grounded-alive frame, else never pay" is well defined. A milestone has NO sparse
payment — its credit is implicit in the shaping (reaching it drops it from the
potential's sum, and the payment was the distance-reduction on the way). So the
mechanism we meant to reuse did not structurally exist for milestones, and
instead of building it, an episode-wide shaping refund was substituted and
described as the same idea generalised. Deferring one payment is narrow;
refunding all shaping is global.

**Why tests passed.** They asserted the intended property (arrival-then-death
pays ~0) but never the CONSERVATION property — that everything the agent
SURVIVED still pays what it used to. The golden sequences only lock the
flag-OFF path, so nothing compared flag-on shaping against the baseline. Add
that test before retrying: for a surviving trajectory, flag-on total must equal
flag-off total.

**The correct generalisation (for v15).** Give each target an ARRIVAL payment
that is deferred exactly like the fruit's: hold that target's credit until the
agent has survived a short window past it, and leave all other shaping alone.
That kills the +5.04 "touch SN3 then die" incentive without touching the other
99% of the signal. Implement per-target, not per-episode.

**Still standing from the same work** (do not throw these out with v14): the
Target model, the progress ladder (L3 1 -> 13 rungs), the alias resolution, and
the seed-milestone RESTORE that fixed the measured retreat bug.

## Open questions / risks
- Escalator: can the agent's ride be made observable enough (4-frame stack)
  for reliable jump timing? This is the main research risk.
- Locate escalator platforms / compressor / snowball spawns in RAM (entity
  table ~0x2B00-0x2B74) if we need signals beyond the pixels.
- Confirm the fruit's logic presence byte (vs deriving from fruits_remaining).
