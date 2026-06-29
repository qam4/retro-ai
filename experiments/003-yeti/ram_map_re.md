# Yeti (MO5) — RAM reverse engineering: the level map

Goal: read the level-2 layout (floors / ladders / gaps / 2 fruits / princess)
directly from RAM, instead of driving the agent around to discover it.
Go-Explore is confirmed unable to map level 2 (it gets stuck at floor 2's
gap jumps), so a RAM-based map reader is the way forward. Level 1 is already
mapped (known fruit/ladder/floor/princess coordinates), so we use it as the
calibration reference.

## Emulator facts (from `emulators/crayon/src`)

- RAM exposed by `iface.read_ram()` is the **0xC000 (49152-byte) CPU space**:
  - `0x0000-0x1FFF`  video RAM, bank-switched (`memory_system.cpp`):
    - page 0 (`get_color_ram`)  = fond / colour attributes
    - page 1 (`get_pixel_ram`)  = forme / shape bitmaps
  - `0x2000-0x9FFF`  **user RAM (32 KB) = game state** (`user_ram[addr-0x2000]`)
  - `0xA000-0xBFFF`  monitor/BASIC area
- Video format (`gate_array.cpp`): 320x200, `offset = y*40 + col`, 40 bytes
  per pixel-row, 200 rows. Each pixel byte = 8 horizontal px (1 bit each);
  each colour byte = fg (high nibble) + bg (low nibble).
- Tiles are **8x8**; fruits are **16x16** (2x2 tiles) — per the user.

## Coordinate units (from `go_explore.py` + observations)

- Player X (RAM `11090` / 0x2B52): range ~0..79 → screen is 320 px, so
  **X unit = 4 px** (x_px = x_ram * 4).
- Player Y (RAM `11089` / 0x2B51): pixel-ish, 0..200. Level 1 spawn y=182
  (bottom, climbs up); level 2 spawn y=30 (top, descends down).

## Known level-1 map (calibration reference, from `annotate_ladders.py`)

Fruit upper-left pixels and presence bytes (byte→0 when collected):

| fruit | floor | UL pixel (x,y) | centre (x,y) | presence addr |
|-------|-------|----------------|--------------|---------------|
| 1     | 1     | (176,176)      | (184,184)    | 0x2FAD (12205)|
| 2     | 2     | (72,142)       | (80,150)     | 0x2F00 (12032)|
| 3     | 3     | (136,112)      | (144,120)    | 0x2E68 (11880)|
| 4     | 4     | (264,80)       | (272,88)     | 0x2DD8 (11736)|

Ladders (x = UL pixel, 16 px wide, between floor pairs): L12 @112, L12 @272,
L23 @232, L34 @168, L45 @200. Princess @ x=304, floor 5. Floor-top pixel y:
{1:200, 2:168, 3:136, 4:104, 5:72}.

## Finding the map region (diff method, `scripts/find_map_in_ram.py`)

Diffed RAM across: L1 start, L2 start, L2-after-moving.
A static map table is level-specific (differs L1 vs L2) AND unchanged as the
player moves (same L2a vs L2b).

- Only **442 bytes differ** between level 1 and level 2 starts.
- They are concentrated in **0x2AE0–0x2FC0** (user RAM = game state), the
  same band that holds the known fruit-presence bytes.
- => the level map / object table lives around **0x2B00–0x2FC0** in user RAM.

(snapshots saved: `output/mo5/yeti/level2/ram/ram_l1.npy`, `ram_l2a.npy`, `ram_l2b.npy`)

## The tilemap (CRACKED)

The screen layout is a **40x25 grid of 1-byte tile-ids in user RAM**,
row-major, base **0x2C27**:

    tile(col, row) = RAM[0x2C27 + row*40 + col]        col 0..39, row 0..24

Calibration: solving `addr = base + row*W + col` on level 1's four known
fruit tiles gives an exact fit with **W=40, base=0x2C27** (e.g. fruit1
@0x2FAD = 0x2C27 + 22*40 + 22).

Coordinate conversions (8x8 tiles):

    pixel_x = col*8      pixel_y = row*8
    agent_X (RAM 0x2B52 / 11090) = pixel_x/4 = col*2   (player X is 4px units)
    agent_Y (RAM 0x2B51 / 11089) = pixel_y    = row*8

### Tile-id legend (confirmed by user)

| ids        | meaning                                            |
|------------|----------------------------------------------------|
| 1,2,3,4    | ladder (a,b,c,d; c/d = ladder-passing-through-floor)|
| 5,6,7,8    | floor (e body / alt body / f left-end / g right-end)|
| >=10       | sprites: each **fruit** = a 2x2 block of 4 distinct ids; **princess** (L1) = id 30 over ~9 tiles |

### Validation on level 1 (`scripts/extract_level_map.py`)

Recovered level 1 **exactly**:
- 5 ladders at x = 112, 168, 200, 232, 272  (matches the 5 hand-mapped ladders)
- 4 fruits at (264,80),(136,112),(72,144),(176,176)  (matches known UL pixels)
- princess sprite (id 30) at top-right ~ (288,16)

## LEVEL 2 MAP (read from RAM, `output/mo5/yeti/level2/level2_map.json`)

6 full floors (top to bottom), 10 ladders, 2 fruits. Annotated grid
(`=` floor, `H` ladder, `1`/`2` fruit; from `scripts/render_tilemap.py`):

```
    col 0         1         2         3
        0123456789012345678901234567890123456789
 6  F1  ====..==========..=====..===============   top floor (player start)
 7          HH                          HH
 8          HH                          HH
 9  F2  ============..=============..===========
10       HH                    HH
11       HH                    HH
12  F3  ====..======..======..==========..======
13                   HH
14                   HH
15  F4  ===================..======..===========
16          H 11                       22  H       fruits hover at rows 16-17
17          H 11                       22  H
18  F5  ===========..=====..==========..========   <- fruits stand on this floor
19                 HH
20                 HH
21  F6  =================..=====================   bottom floor
22         HH                    HH
23         HH                    HH                 ladders descend below F6
24         HH                    HH                 (toward princess?)
```

### Floors (pixel_y / agent_y)
F1 row6 y48 | F2 row9 y72 | F3 row12 y96 | F4 row15 y120 | F5 row18 y144 | F6 row21 y168

### Ladders (which floors they connect; x in px and agent units)
| floors  | ladder x (px) | agent_x |
|---------|---------------|---------|
| F1<->F2 | 72, 296       | 18, 74  |
| F2<->F3 | 8, 184        | 2, 46   |
| F3<->F4 | 128 (only one)| 32      |
| F4<->F5 | 32, 288       | 8, 72   |
| F5<->F6 | 112 (only one)| 28      |
| F6<->below | 48, 224    | 12, 56  |

The F3<->F4 and F5<->F6 single ladders are descent bottlenecks. The
rows21-24 ladders (x=48, x=224) descend below the bottom floor F6 — likely
toward the princess.

### Fruits (both on floor F5, row 18)
| fruit | tile ids | pixel (x,y) | agent (x,y) | presence addr |
|-------|----------|-------------|-------------|---------------|
| left  | 31-34    | (56,128)    | (14,128)    | 0x2EAE (11950)|
| right | 35-38    | (256,128)   | (64,128)    | 0x2EC7 (11975)|

**Fruit presence addresses are positional, not fixed.** The presence byte IS
the fruit's UL cell in the tilemap (`0x2C27 + row*40 + col`), so it moves with
the fruit. That's why level 1's `{0x2FAD,0x2F00,0x2E68,0x2DD8}` cannot be
reused for level 2 — reading them against the L2 state gives garbage
(0x2FAD->0 would even look like a *collected* fruit, 0x2F00->5 is a floor
tile, 0x2E68->2 a ladder tile). Verified: L2 addrs 0x2EAE=31 and 0x2EC7=35,
stable as the player moves; fruits_remaining(11055)=2. Design upshot: derive
per-fruit presence addrs from the tilemap (`extract_level_map.py` now emits
`presence_addr`) rather than hard-coding per level — generalizes to all levels.
NOT yet collection-tested (zeroing-on-pickup needs level-2 gameplay; same
engine mechanism as L1, so inferred).

### Princess (FOUND — separate entity bytes, not in the tilemap)

The princess is a moving-animation sprite (2 animation frames, per the user),
not part of the background tilemap, so she lives in the globals/entity area:

| value     | RAM addr        | L1   | L2   |
|-----------|-----------------|------|------|
| princess Y| 0x2B00 (11008)  | 54   | 182  |
| princess X| 0x2B01 (11009)  | 72   | 72   |

- X is in 4px units → x_px = X*4 = **288** (right side) in BOTH levels.
- Y flips: L1 = 54 (top-right), L2 = 182 (**bottom-right**). Matches the game.
- Static within a level (same L2a vs L2b) — rules out the player/goats.
- Bytes 0x2B04-0x2B06 differ by level too (L1: 24,66,192; L2: 0,0,0) — likely
  princess sprite/animation attributes.
- Confirmed visually: the bottom-right of the level-2 frame shows the
  princess sprite at ~(288,182).

So level-2 princess: pixel ~(288,182), **agent (x=72, y=182)**, bottom-right.

### Enemies
- Level 2 has **goats** that chase the player (Pac-Man style; jump gaps, use
  ladders) and a **yeti** that paces left/right along the bottom. These are
  dynamic sprites (they show up in the "moves-in-L2" bytes of
  `scripts/compare_entity_region.py`, e.g. around 0x2B27-0x2B2E, 0x2B40,
  0x2B44-0x2B46). Exact per-enemy records not yet pinned (not needed for the
  static map; the CNN will learn evasion from pixels).

### Note on floor numbering
The two fruits read out on floor F5 (row 18) = 5th line from the top / 2nd
from the bottom, vs the user's recollection of "3rd floor". The pixel/agent
coords are validated ground truth (the level-1 extractor matched the known
map exactly), so this is just a floor-counting-convention difference.

## Tooling
- `scripts/find_map_in_ram.py`   — diff RAM to locate level-specific static bytes
- `scripts/render_tilemap.py`    — dump the 40x25 tilemap (occupancy + tile-id views)
- `scripts/extract_level_map.py` — structured floors/ladders/fruits (+ JSON)


---

# Level-2 training plan (approach A — chosen)

Decision (user): **approach A** — get a first level-2 run going with the
*simplest* setup, no nav-graph path shaping. Add the full nav graph (B) only
if A stalls. Rationale: H-W showed phase-1 exploration + checkpoint curriculum
is powerful, the gap-jump geometry is fiddly to model, and level 2 is *shorter*
than level 1 (2 fruits vs 4), so a baseline may crack it without shaping.

## Reward (A)
`fruit_princess_bonus` (already registered in `rewards.py`): C++ fruit-bonus
per pickup + a one-shot princess-touch bonus. **No `yeti_map` nav graph**, so
none of the level-1 floor/ladder/gap geometry is needed for the reward. Keep
the survival bonus + PBRS as on level 1. (Path-progress rewards
`fruit_bonus_path_progress[_universal]` are the level-1 nav-graph ones we are
deliberately NOT using for L2 yet.)

## Recipe (from H-V / H-W)
Two-phase anneal: phase-1 = big exploratory steps (`n_steps`=16, no
`target_kl`) to escape plateaus; phase-2 = anneal (`n_steps`=512,
`target_kl`=0.05) to converge. Checkpoint curriculum bootstraps CP seeds as
fruits are collected. keep-best snapshot sweep to capture the transient peak.

## Verified RAM hooks (level 2)
- CP0 start state: `output/mo5/yeti/level2/level2_start.sav` (load on reset
  instead of game-reset, which boots level 1).
- Fruits (2): presence `0x2EAE` (11950) and `0x2EC7` (11975) — auto-derived
  from the tilemap; verified to hold the fruit tile-ids and be position-stable.
  NOT yet collection-tested (zeroing on pickup) — needs L2 gameplay.
- `fruits_remaining` (11055) = 2 at start. Globals (lives 11095, bonus
  11010/11, score 11093/94, x 11090, y 11089, princess-flag 11050) carry over.
- Goals = 2 fruits + princess = **3** (level 1 had 5).

## Code seams to parameterize (in `scripts/train_checkpoint_curriculum.py`)
Everything below hard-codes level 1; all need a level param (default = L1, so
level-1 runs stay byte-identical):
1. `FRUIT_PRESENCE_ADDRS = {1:0x2FAD,...}` (module const) -> per-level dict.
2. `CheckpointManager.FRUITS_TOTAL = 4` (class attr) -> instance, from config.
3. `4 - fruits` CP math in `reset()`/`step()`/`_log_episode()` -> `fruits_total - fruits`.
4. Princess = CP level `5` -> `fruits_total + 1`.
5. `_fruit_vector()` returns a 4-d vector (for MultiInputPolicy) -> `fruits_total`-d.
6. CP0 reset: `gym_env.reset()` -> load `start_state` save when configured.
7. `CurriculumCallback._write_diag` / `_write_admission_diag` hard-code 5 goal
   columns (reach1..reach_princess, gscore0..4) -> size to `fruits_total`.
   (Highest-risk spot: fixed-width CSV header + f-string rows.)
8. New config fields (likely a `level:` block or `curriculum` additions):
   `start_state`, `fruits_total`, `fruit_presence_addrs`.

Same constants are duplicated in `scripts/train_segment.py`,
`scripts/eval_from_reset.py`, `scripts/go_explore.py` — for the first L2 run we
only need the curriculum-training + eval paths. Cleanest long-term: a shared
per-level table (dataclass/registry) the profile selects; the inline dicts in
~10 scripts then collapse to one source.

## Open questions / hypotheses
- **H (A-works):** the 2-fruit level is short enough that fruit+princess
  reward + curriculum + phase-1 exploration reaches the princess without nav
  shaping. If reach-fruit plateaus like level-1's pre-PBRS 2-fruit wall, add B.
- **Gaps:** level 2 has floor gaps requiring jumps (walking off = death). No
  reward models this; the CNN must learn it from pixels (it learned jumps on
  L1). If the agent farms deaths at gaps, may need a gap-aware penalty/shaping.
- **Enemies:** goats chase (Pac-Man style) + a bottom yeti paces L/R. Death
  penalty already exists; evasion learned from pixels. Watch for the agent
  getting stuck avoiding rather than progressing.
- **Princess detect:** assume the level-cleared flag (11050) rising edge works
  on L2 as on L1 (same engine). Verify on the first run that a princess touch
  registers (it gates CP_last success + the one-shot bonus).
- **Start-state determinism:** every CP0 episode loads the same
  `level2_start.sav`; rely on env stochasticity (action sampling) for diversity
  as on level 1. May want random no-op frames at reset if it's too deterministic.

## Status
Map + hooks done & verified. Next: implement the seam parameterization
(level-1-preserving defaults), add a `yeti_curriculum_l2_v1` config + L2
profile, smoke-test 50-100 steps from `level2_start.sav`, then launch phase-1.

---

# Level-2 wiring — implementation notes & a key gotcha (approach A)

Wired level-2 support into the checkpoint curriculum with **level-1-preserving
defaults** (level-1 runs are byte-identical; new behavior only when configured):

- `run_config.py` `CurriculumConfig` gained `fruits_total` (default 4),
  `fruit_presence_addrs` (default None -> level-1 dict), `start_state`
  (default None -> game reset).
- `train_checkpoint_curriculum.py`: `CheckpointManager.FRUITS_TOTAL` is now an
  instance value; the env reads the per-level fruit addrs / fruits_total /
  start-state; all `4 - fruits` -> `fruits_total - fruits`, princess level =
  `fruits_total + 1`; the fruit-presence vector and the diag-CSV column widths
  size to `fruits_total`.
- New `game_profiles/mo5_yeti_fruit_level2.yaml` (profile `yeti_fruit_level2`).
- New config `experiments/003-yeti/configs/yeti_curriculum_l2_v1.yaml`.

## GOTCHA found by the smoke test: the bonus-stall termination
Loading `level2_start.sav` made every episode end at length 1 (`env_done`).
Root cause: the save loads with the **bonus frozen at 1000** (level 2 holds
it ~6 gym steps / ~24 frames before ticking) and **flag 11050 = 1** (carryover
from clearing level 1). The level-1 profile's C++ `bonus_stall_frames: 10`
treats a frozen bonus as a stuck agent and terminates within ~2 gym steps
(frame_skip=4). Level 1 never hits this because its bonus ticks continuously
from frame 0.
- **Fix:** `mo5_yeti_fruit_level2.yaml` sets `bonus_stall_frames: 120` —
  survives the initial freeze and the slow inter-tick gaps (~8 frames). The
  Python env-level `stall_threshold` (15 gym steps) remains the binding
  anti-stuck guard during play.
- **flag 11050 behavior (good news):** it is 1 at the level-2 start, then
  drops to 0 once level 2 is underway (~step 6), and would rise 0->1 on a
  level-2 princess touch — so the existing rising-edge princess detector
  works on level 2 (the env reads `_prev_princess_flag` after the reset
  noops, sees 1, and no spurious touch fires).
- **Open:** the bonus can also freeze mid-level in odd/stuck states (seen with
  a dumb fixed action). 120 frames is a generous guard; watch real runs for
  spurious stalls and whether bonus-stall is even the right signal for L2.

## Validation
- L2 smoke (1500 steps): episodes now normal length (37-89, mean 66), start
  at y=30 (level-2 top), 3 CP pools (CP0/CP1/CP2 for 2 fruits). No crashes.
- L1 smoke (game reset, defaults): still collects fruits (0->1 success,
  CP1 pool filled) — **no regression**.
- Diag-CSV writer unit-tested for fruits_total=2: columns
  `reach1,reach2,reach_princess,succ_ema1..2,start_frac0..2,gscore0..2` — no
  index errors.

## Status / next
Implementation complete & validated by smoke tests. Not yet committed; the
real 20M phase-1 run (`yeti_curriculum_l2_v1`) not yet launched.

---

# Level-2 run 1 (yeti_curriculum_l2_v1, approach A) — RESULT: NEGATIVE

20M steps, 9.5h, phase-1 exploration, fruit_princess_bonus (no shaping).
**Zero fruits collected in the entire run** — cp=[0,0,0], saves=[0,0,0],
reset_reach=[1.00, 0,0,0]. The curriculum never bootstrapped a single CP seed.

Diagnosis from episodes.csv (275,864 episodes):
- **All** episodes ended in `stall` (none death/max_steps/princess), mean
  length **72 steps** (max_steps=1000 never reached).
- `reached_level` always 0; `final_y` mostly 16-48 (start y=30; the agent
  rarely leaves the **top floor F1 @ y48**), deepest-ever y=102, vs fruits at
  y~128-144 (floor F5). `final_x` max 27.

Root causes (compounding):
1. **No gradient to descend.** Level 2 spawns at the TOP; the nearest fruit is
   ~5 floors DOWN. fruit_princess_bonus only pays at a fruit/princess, so PPO
   has nothing to move toward and never discovers the multi-floor descent.
   (Level 1 worked sparse because spawn is right next to fruit 1.)
2. **Stall guard truncates exploration.** Bonus freezes whenever the agent
   isn't actively progressing, so the env stall (15 gym steps) kills episodes
   at ~72 steps — far too short to stumble down 5 floors under sparse reward.

Conclusion: **approach A is insufficient for level 2.** Need (a) descent
shaping and (b) longer episodes.

## Next: approach B-lite (proposed)
- **Shaping:** reward progress toward the nearest uncollected target using the
  RAM-derived coordinates (fruit A agent(14,128), fruit B (64,128), then
  princess (72,182)). A simple pixel/Manhattan distance potential gives a
  descent gradient WITHOUT a full floor/ladder nav graph. (If the agent farms
  the gradient by oscillating, use the per-target best-distance ratchet like
  fruit_bonus_path_progress.)
- **Episodes:** relax/disable the bonus-stall termination (it fires constantly
  and isn't a useful signal on level 2) and let max_steps bound the episode,
  so the agent has room to explore the descent.
- Keep phase-1 exploration + curriculum (which will start bootstrapping CP
  seeds once the agent reaches a fruit even occasionally).
