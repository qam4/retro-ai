---
inclusion: manual
---

# Reward Discovery Guide

How to discover score, timer, and game-over conditions for a new game.
This is the playbook we follow every time we add a game to retro-ai.

## Overview

Retro games store score/timer/lives in RAM as BCD or binary values.
To train an RL agent, we need to:
1. Find which RAM addresses hold the score (reward signal)
2. Find which addresses hold the timer or lives (episode termination)
3. Determine the encoding (BCD vs binary, byte order, byte count)
4. Wire it all up in a game profile YAML

There are multiple discovery methods depending on what's available:

| Method | When to use | Tools |
|--------|------------|-------|
| RAM scanning | Emulator exposes `read_ram()` | `scripts/scan_ram.py`, `scripts/ram_watcher.py` |
| OCR / vision | No RAM access, or score is only on screen | `scripts/dump_frames.py` + OCR pipeline |
| Manual annotation | Complex scoring, no clear RAM pattern | `scripts/ram_watcher.py` interactive mode |
| Disassembly | Need to understand game logic deeply | Emulator debugger, disassembly docs |

## Method 1: RAM Scanning (preferred)

### Prerequisites
- Emulator adapter implements `read_ram()` in C++ and Python bindings
- Game profile YAML exists with ROM paths and startup sequence
- You can get the game into active gameplay via actions

### Step 1: Automated scan

```bash
python scripts/scan_ram.py --profile game_profiles/<game>.yaml --action 1 --seconds 15
```

- `--action`: the action that causes score to change (usually Up=1 for racing, Fire=5 for shooters)
- `--seconds`: how long to play (should be long enough to see score change multiple times)
- `--interval`: sampling rate (default 1.0s, use 0.5 for fast-changing scores)

Look for:
- INCREASING addresses → score candidates
- DECREASING addresses → timer or lives candidates

### Step 2: Detailed inspection

Re-run with `--detail` on candidate addresses and their neighbors:

```bash
python scripts/scan_ram.py --profile ... --detail 53,54,55,56,65,66
```

This shows every sample value for those addresses, making it easy to spot:
- BCD encoding (values like 0x59, 0x58, 0x57... = decimal 59, 58, 57)
- Multi-byte scores (one byte wraps 0x99→0x00 while adjacent byte increments)
- Little-endian vs big-endian byte order

### Step 3: Interactive confirmation (optional)

```bash
python scripts/ram_watcher.py --profile game_profiles/<game>.yaml
```

Use `hold <action>` to play, `m` to mark snapshots, `f` to filter for monotonic increases.

### Step 4: Update game profile

Add `reward_params` to the game profile YAML:

```yaml
reward_mode: memory
reward_params:
  score_address_count: "1"
  score_address_0_addr: "54"       # RAM address of first score byte
  score_address_0_bytes: "2"       # Number of bytes (1, 2, or 4)
  score_address_0_bcd: "1"         # 1 = BCD encoded, 0 = binary
  score_address_0_le: "1"          # 1 = little-endian, 0 = big-endian
  timer_minutes_addr: "65"         # Timer address (minutes/high byte)
  timer_seconds_addr: "66"         # Timer address (seconds/low byte)
  done_when_timer_zero: "true"     # Episode ends when timer hits 0
```

### Step 5: Verify

Run a full episode and check that:
- Reward total matches the on-screen score at game over
- `done=True` fires at the right moment
- No negative reward spikes from BCD wrapping

## Method 2: OCR / Vision-based

For emulators without RAM access, or when the score display is the only
reliable source.

### When to use
- Emulator doesn't expose internal RAM
- Score is computed from multiple sources (combo systems, multipliers)
- Score display uses custom graphics that don't map to simple RAM values

### Approach
1. Use `scripts/dump_frames.py` to capture frames during gameplay
2. Identify the screen region where the score is displayed
3. Configure the `vision` reward mode with the score region coordinates
4. Optionally add OCR post-processing for exact score extraction

### Game profile config

```yaml
reward_mode: vision
reward_params:
  screen_region_x: 112    # X offset of score display
  screen_region_y: 80     # Y offset
  screen_region_w: 40     # Width in pixels
  screen_region_h: 14     # Height in pixels
```

## Method 3: Manual / Hybrid

Some games have complex scoring that doesn't map cleanly to either method:
- Lives system (3 lives, game over on 0)
- Level-based progression (score resets between levels)
- Combo multipliers stored in separate RAM locations

Use `ram_watcher.py` interactively to understand the game's memory layout,
then configure multiple score addresses or custom reward logic.

## Common Patterns by Platform

### Videopac (8048 CPU)
- 64 bytes internal RAM + 128 bytes external RAM = 192 bytes total
- Scores typically in IntRAM[32-63] (general purpose area)
- Timers often in ExtRAM[0x00-0x0F]
- Almost always BCD encoded
- Often little-endian (low byte at lower address)

### MO5 (6809 CPU)
- Larger RAM space, scores can be anywhere
- Mix of BCD and binary encoding
- Check zero-page first (0x00-0xFF) for frequently accessed variables

### General tips
- Registers (R0-R7) and stack are rarely score — focus on general RAM
- If an address oscillates rapidly, it's probably a display counter or animation state
- Timer addresses decrease by exactly 1 BCD unit per second (~60 frames)
- Multi-byte scores: the high byte only changes when the low byte wraps

## Adding a New Emulator

When adding `read_ram()` to a new emulator adapter:

1. Add `std::vector<uint8_t> read_ram() const override` to the adapter class
2. Return a flat byte vector covering all game-relevant RAM
3. Document the memory layout in the game profile comments
4. Add a label function to `scan_ram.py` (`LABEL_FUNCS` dict) for human-readable output
5. Add the emulator type to `create_emulator()` in both scan scripts
