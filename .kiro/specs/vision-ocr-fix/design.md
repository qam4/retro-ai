# Vision OCR Fix — Bugfix Design

## Overview

The `VisionRewardSystem` OCR pipeline fails to detect Videopac digit characters because three hardcoded constants are wrong: `kDigitWidth=7` (should be 8), `kDigitHeight=10` (should be 14), and the sliding window step equals `kDigitWidth` (should be 16, the quad character spacing). Additionally, the default `ScreenRegion` width of 40px doesn't evenly accommodate digits at 16px spacing. The fix updates these constants, adjusts the legacy `digit_templates_` array dimensions to match, and widens the default screen region.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — `extract_score()` is called with a framebuffer containing valid Videopac digits rendered at the correct Intel 8245 VDC dimensions (8×14 at 16px spacing), but the undersized/misaligned extraction windows fail to match them.
- **Property (P)**: The desired behavior — `extract_score()` correctly detects and returns the numeric score from VDC-rendered digits.
- **Preservation**: Existing behaviors that must remain unchanged — empty/blank observation handling, Otsu thresholding, 0.70 acceptance threshold, `compute_reward()` delta logic, luminance formula, ROM data, and ROM row/column mapping.
- **`extract_score()`**: Method in `VisionRewardSystem` (`src/reward_systems/vision.cpp`) that slides a window across the score region, crops digit patches, and matches them against ROM templates to produce a numeric score.
- **`match_videopac_digit()`**: Free function in `vision.cpp` that binarizes a grayscale patch via Otsu thresholding and compares it against the `videopac_digit_rom` patterns using normalized cross-correlation.
- **`kDigitWidth` / `kDigitHeight`**: Static constants in `VisionRewardSystem` defining the pixel dimensions of a single digit patch.
- **`kQuadCharSpacing`**: The horizontal distance (16px) between successive character origins in the Intel 8245 quad-character display mode (8px character + 8px gap).
- **`videopac_digit_rom[10][8]`**: The authoritative Intel 8245 character ROM patterns — 10 digits, each 8 bytes (8 rows), MSB-first, row 7 blank.

## Bug Details

### Bug Condition

The bug manifests when `extract_score()` processes a framebuffer containing valid Videopac digit characters rendered by the Intel 8245 VDC. The extraction pipeline uses incorrect dimensions (7×10 instead of 8×14) and incorrect step size (7 instead of 16), causing digit patches to be undersized, misaligned with the character grid, and unable to match the ROM templates at the 0.70 acceptance threshold.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type { observation: uint8_t[], score_region: ScreenRegion }
  OUTPUT: boolean

  digits := renderVDCDigits(input.observation, input.score_region)

  RETURN digits.count > 0
         AND digits.charWidth == 8
         AND digits.charHeight == 14
         AND digits.spacing == 16
         AND (kDigitWidth != 8 OR kDigitHeight != 14 OR slidingStep != 16)
END FUNCTION
```

### Examples

- **Single digit "5" at position (112, 80)**: The VDC renders an 8×14 character. The system crops a 7×10 patch starting at (112, 80), missing the rightmost column and bottom 4 scanlines. The truncated patch scores below 0.70 against the ROM template → returns -1 instead of 5.
- **Score "123" at positions (112, 128, 144) with 16px spacing**: The system steps by 7px, so it reads patches at offsets 0, 7, 14, 21, 28 within the region. None of these align with the actual digit positions at offsets 0, 16, 32 → returns -1 instead of 123.
- **Score "00" at positions (112, 128)**: Even the first digit at offset 0 is cropped to 7×10 instead of 8×14, so the partial "0" pattern doesn't match → returns -1 instead of 0.
- **Edge case — default region width**: With `ScreenRegion{112, 80, 40, 14}` and 16px step, `max_digits = 40/16 = 2`. A 3-digit score at offsets 0, 16, 32 requires width ≥ 48. The third digit at offset 32 needs columns 32–39 (8 pixels), which fits in width 40, but `max_digits` truncates to 2 → third digit is never read.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `extract_score()` called with an empty observation must continue to return -1
- `extract_score()` called with a framebuffer containing no digit characters must continue to return -1 without false positives
- `match_videopac_digit()` must continue to use Otsu thresholding for binarization and the 0.70 normalized cross-correlation acceptance threshold
- `compute_reward()` must continue to return the delta between current and previous detected scores, returning 0.0 when no score is visible or on first detection
- `crop_to_grayscale()` must continue to use the luminance formula (0.299R + 0.587G + 0.114B) and respect observation space bounds
- The `videopac_digit_rom[10][8]` ROM data must remain unmodified
- `match_videopac_digit()` must continue to map patch rows to ROM rows 0–6 (skipping blank row 7) and patch columns to ROM columns 0–7 (MSB-first)

**Scope:**
All inputs that do NOT involve VDC-rendered digit characters at the correct 8×14 / 16px-spacing dimensions should be completely unaffected by this fix. This includes:
- Empty observations
- Framebuffers with no digit content (blank regions, non-digit graphics)
- The Otsu binarization and cross-correlation matching logic
- The `compute_reward()` delta calculation
- The grayscale conversion pipeline

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

1. **Incorrect `kDigitWidth` constant (7 instead of 8)**: The Intel 8245 VDC renders characters 8 pixels wide (7 active ROM columns + 1 blank column). `kDigitWidth = 7` crops patches 1 pixel too narrow, losing the rightmost column. This directly affects `extract_score()` patch extraction and the dimensions passed to `match_videopac_digit()`.

2. **Incorrect `kDigitHeight` constant (10 instead of 14)**: The VDC doubles each of the 7 ROM rows into 2 scanlines, producing 14 scanlines per character. `kDigitHeight = 10` crops patches 4 scanlines too short, losing the bottom portion of the character. The ROM row mapping `(py * 7) / patch_h` in `match_videopac_digit()` is designed to handle any patch height, but the cropped patch itself is missing pixel data.

3. **Incorrect sliding window step (7 instead of 16)**: In `extract_score()`, `step = kDigitWidth = 7`. Videopac quad-mode characters are spaced 16 pixels apart (8px char + 8px gap). A step of 7 means successive windows land at offsets 0, 7, 14, 21, ... which never align with the actual character positions at 0, 16, 32, 48, ...

4. **Default ScreenRegion width too narrow**: `ScreenRegion{112, 80, 40, 14}` with a 16px step gives `max_digits = 40/16 = 2`, which cannot read a 3-digit score. The width needs to be at least 48 (3 × 16) to accommodate 3 digits, or wider for 4-digit scores.

5. **Legacy `digit_templates_` dimension mismatch**: The `digit_templates_` array is declared as `kDigitWidth * kDigitHeight` elements per digit. Changing the constants from 7×10 to 8×14 changes the array size from 70 to 112 elements per digit. The template data must be updated to match, or the templates can be removed if they are truly unused (currently `match_digit()` delegates to `match_videopac_digit()` which uses the ROM data, not `digit_templates_`).

## Correctness Properties

Property 1: Bug Condition — Correct Score Detection for VDC-Rendered Digits

_For any_ framebuffer containing N valid Videopac digit characters (0–9) rendered at 8px width, 14 scanlines height, and 16px quad spacing within the score region, the fixed `extract_score()` function SHALL return the correct numeric score (digits concatenated left-to-right as a base-10 integer).

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation — Non-Digit and Empty Input Behavior

_For any_ input where the bug condition does NOT hold (empty observation, blank framebuffer, or framebuffer with no recognizable digit patterns), the fixed `extract_score()` function SHALL produce the same result as the original function (returning -1), preserving all existing rejection behavior, thresholding logic, luminance conversion, and reward delta computation.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**


## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `include/retro_ai/reward_systems/vision.hpp`

**Constants and Array Declaration**:
1. **Update `kDigitWidth`**: Change from 7 to 8 to match the full VDC character width.
2. **Update `kDigitHeight`**: Change from 10 to 14 to match the doubled-scanline VDC rendering.
3. **Add `kQuadCharSpacing`**: Introduce a new named constant `static constexpr int kQuadCharSpacing = 16;` to make the sliding window step explicit and self-documenting.
4. **`digit_templates_` array**: The array dimensions are `kDigitWidth * kDigitHeight` (changes from 70 to 112 elements per digit). Update the template data in `vision.cpp` to 8×14, OR remove `digit_templates_` entirely since `match_digit()` already delegates to `match_videopac_digit()` which uses the authoritative ROM data. Removal is preferred to eliminate the redundant data source.

**File**: `src/reward_systems/vision.cpp`

**Function**: `extract_score()`

**Specific Changes**:
1. **Sliding window step**: Replace `int step = kDigitWidth;` with `int step = kQuadCharSpacing;` (16px) so successive windows align with the VDC character grid.
2. **`digit_templates_` data**: Either update all 10 digit template arrays from 7×10 (70 elements) to 8×14 (112 elements) to match the new constants, or remove the `digit_templates_` static member entirely. Removal is recommended since `match_digit()` delegates to `match_videopac_digit()` which uses `videopac_digit_rom` — the templates are dead data.
3. **No changes to `match_videopac_digit()`**: The ROM row/column mapping logic (`(py * 7) / patch_h` and `(px * 8) / patch_w`) is already parameterized by `patch_w` and `patch_h`, so it will automatically handle the corrected 8×14 dimensions without modification.
4. **No changes to `crop_to_grayscale()`**: The crop function uses `score_region_` dimensions, not `kDigitWidth`/`kDigitHeight`, so it is unaffected.

**File**: `src/reward_systems.cpp`

**Default ScreenRegion**:
1. **Update default width**: Change `ScreenRegion{112, 80, 40, 14}` to `ScreenRegion{112, 80, 48, 14}` so that `max_digits = 48 / 16 = 3`, accommodating a 3-digit score. The last digit at offset 32 needs columns 32–39 (8 pixels), which fits within width 48.
2. **Update parameterized default**: Also update the fallback value in `param_int(params, "screen_region_w", 40)` from 40 to 48.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Construct synthetic framebuffers containing known Videopac digit characters rendered at the correct VDC dimensions (8×14 at 16px spacing). Call `extract_score()` on the unfixed code and observe that it returns -1 or incorrect values.

**Test Cases**:
1. **Single Digit Test**: Render digit "5" as an 8×14 character at offset 0 in the score region. Call `extract_score()` on unfixed code (will return -1 because the 7×10 patch misses data).
2. **Multi-Digit Alignment Test**: Render "123" at offsets 0, 16, 32 with 16px spacing. Call `extract_score()` on unfixed code (will return -1 or wrong value because step=7 misaligns windows).
3. **Dimension Mismatch Test**: Render digit "0" at correct dimensions. Extract a 7×10 patch manually and a 8×14 patch, compare match scores against ROM template to confirm the 7×10 patch scores below 0.70.
4. **Default Region Width Test**: Render a 3-digit score "456" at offsets 0, 16, 32 within a width-40 region. Verify that `max_digits = 40/7 = 5` on unfixed code reads garbage at wrong offsets (will fail on unfixed code).

**Expected Counterexamples**:
- `extract_score()` returns -1 for framebuffers containing valid VDC digits
- Match scores for 7×10 patches fall below the 0.70 threshold due to missing pixel data
- Multi-digit scores are garbled because the step=7 window lands between characters

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := extract_score_fixed(input.observation)
  ASSERT result == expectedScore(input)
END FOR
```

Where `expectedScore(input)` is the numeric value of the VDC-rendered digit sequence.

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT extract_score_original(input) == extract_score_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain (random noise framebuffers, blank regions, partial content)
- It catches edge cases that manual unit tests might miss (boundary pixels, near-threshold matches)
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for empty observations, blank framebuffers, and non-digit content, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Empty Observation Preservation**: Verify `extract_score({})` returns -1 on both unfixed and fixed code
2. **Blank Region Preservation**: Generate random framebuffers with uniform or noise content (no digit patterns) and verify both versions return -1
3. **Threshold Preservation**: Verify the 0.70 acceptance threshold is unchanged by testing patches that score just above and just below the threshold
4. **Reward Delta Preservation**: Verify `compute_reward()` returns correct deltas for sequences of detected scores on both versions

### Unit Tests

- Test `extract_score()` with a synthetic framebuffer containing a single VDC digit at correct 8×14 dimensions → verify correct digit returned
- Test `extract_score()` with multi-digit scores at 16px spacing → verify correct concatenated score
- Test `extract_score()` with empty observation → verify returns -1
- Test `extract_score()` with blank/noise framebuffer → verify returns -1
- Test that `kDigitWidth == 8`, `kDigitHeight == 14`, `kQuadCharSpacing == 16`
- Test default `ScreenRegion` width accommodates 3 digits at 16px spacing

### Property-Based Tests

- Generate random digit sequences (1–3 digits, values 0–9), render them into synthetic framebuffers at 8×14 / 16px spacing, and verify `extract_score()` returns the correct numeric score
- Generate random non-digit framebuffers (uniform color, random noise, gradient patterns) and verify `extract_score()` returns -1 on both original and fixed code
- Generate random `ScreenRegion` configurations and verify `crop_to_grayscale()` produces identical output on both versions (luminance formula preservation)

### Integration Tests

- Test full `compute_reward()` flow: construct two `StepResult`s with different rendered scores, verify the reward equals the score delta
- Test `reset()` followed by `compute_reward()` returns 0.0 on first call
- Test score detection across multiple consecutive frames with increasing scores to verify the reward accumulation pipeline end-to-end

## Follow-Up: Vision System Per-Emulator Abstraction

**This section captures a known architectural issue for a future feature spec. It is NOT in scope for this bugfix.**

The current `VisionRewardSystem` is hardcoded with Videopac-specific assumptions:

1. `videopac_digit_rom[10][8]` patterns are baked into `vision.cpp` — no equivalent for MO5 or other emulators
2. `match_videopac_digit()` is the only digit matcher — `match_digit()` unconditionally delegates to it
3. Default `ScreenRegion{112, 80, 48, 14}` and `ObservationSpace{160, 240, 3, 8}` are Videopac dimensions (MO5 is 320×200)
4. The `RewardSystemFactory` doesn't know which emulator is requesting the reward system, so it can't select platform-appropriate templates

If someone sets `reward_mode: vision` on an MO5 game, it silently uses Videopac digit templates against MO5 framebuffers — guaranteed garbage.

**Proposed future work:**
- Pass emulator type through to the factory so it can select the right templates/dimensions
- Or make `VisionRewardSystem` configurable with a "digit matcher" strategy that varies per platform
- The existing `load_digit_templates()` hook was designed for this but is currently a no-op
- MO5 would need its own character ROM patterns, screen dimensions, and character spacing
