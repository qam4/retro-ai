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

