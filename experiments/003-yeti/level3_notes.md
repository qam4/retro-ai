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

## Open questions / risks
- Escalator: can the agent's ride be made observable enough (4-frame stack)
  for reliable jump timing? This is the main research risk.
- Locate escalator platforms / compressor / snowball spawns in RAM (entity
  table ~0x2B00-0x2B74) if we need signals beyond the pixels.
- Confirm the fruit's logic presence byte (vs deriving from fruits_remaining).
