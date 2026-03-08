# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** — VDC Digit Detection Failure
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For this deterministic bug, scope the property to concrete failing cases: single digits (0–9) rendered at 8×14 with 16px spacing
  - Create `tests/python/test_vision_ocr_bug.py` using pytest and hypothesis
  - Build a helper that renders a synthetic VDC digit into a flat RGB888 framebuffer at the correct 8×14 dimensions with 16px quad spacing, using `videopac_digit_rom` bit patterns (7 ROM rows doubled to 14 scanlines, 8 columns MSB-first, row 7 blank)
  - Import `retro_ai_native` and call `VisionRewardSystem.extract_score()` via the Python bindings (or test through `compute_reward()` with constructed `StepResult`s)
  - Property: for any digit d in 0–9 rendered at offset 0 in a ScreenRegion(0, 0, 48, 14) framebuffer, `extract_score()` returns d
  - Also test multi-digit: render "123" at offsets 0, 16, 32 → expect 123
  - Run test on UNFIXED code: `PYTHONPATH=build/ci-linux:python pytest tests/python/test_vision_ocr_bug.py -x`
  - **EXPECTED OUTCOME**: Test FAILS (this is correct — it proves the bug exists)
  - Document counterexamples found (e.g., `extract_score()` returns -1 for valid VDC digit framebuffers)
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** — Non-Digit and Empty Input Behavior
  - **IMPORTANT**: Follow observation-first methodology
  - Create `tests/python/test_vision_ocr_preservation.py` using pytest and hypothesis
  - Observe on UNFIXED code: `extract_score({})` returns -1 (empty observation)
  - Observe on UNFIXED code: `extract_score(blank_framebuffer)` returns -1 (all-zero pixels)
  - Observe on UNFIXED code: `extract_score(random_noise_framebuffer)` returns -1 (no digit patterns)
  - Write property-based test: for all framebuffers where isBugCondition is false (empty, blank, random noise with no recognizable digit patterns), `extract_score()` returns -1
  - Use hypothesis to generate random uint8 framebuffers of the correct observation size (160×240×3) and verify `extract_score()` returns -1 (random noise is overwhelmingly unlikely to match ROM templates at 0.70 threshold)
  - Verify `compute_reward()` returns 0.0 delta when no score is detected on both calls
  - Verify `crop_to_grayscale()` luminance formula is unchanged (0.299R + 0.587G + 0.114B)
  - Run tests on UNFIXED code: `PYTHONPATH=build/ci-linux:python pytest tests/python/test_vision_ocr_preservation.py -x`
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 3. Fix VDC digit detection constants and defaults

  - [x] 3.1 Update constants in `include/retro_ai/reward_systems/vision.hpp`
    - Change `kDigitWidth` from 7 to 8 (full VDC character width: 7 active ROM columns + 1 blank)
    - Change `kDigitHeight` from 10 to 14 (VDC doubled scanlines: 7 ROM rows × 2)
    - Add `static constexpr int kQuadCharSpacing = 16;` (8px char + 8px gap)
    - Remove `digit_templates_` static array declaration (dead data — `match_digit()` delegates to `match_videopac_digit()` which uses ROM data)
    - _Bug_Condition: isBugCondition(input) where kDigitWidth != 8 OR kDigitHeight != 14 OR slidingStep != 16_
    - _Expected_Behavior: kDigitWidth == 8, kDigitHeight == 14, kQuadCharSpacing == 16_
    - _Preservation: videopac_digit_rom[10][8] unchanged, match_videopac_digit() logic unchanged_
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Update `src/reward_systems/vision.cpp`
    - In `extract_score()`: change `int step = kDigitWidth;` to `int step = kQuadCharSpacing;` (16px quad spacing)
    - Remove the `digit_templates_` static data definition (all 10 digit template arrays, ~70 elements each)
    - Do NOT modify `match_videopac_digit()` — its row/column mapping is already parameterized by patch dimensions
    - Do NOT modify `crop_to_grayscale()` — it uses `score_region_` dimensions, not kDigitWidth/kDigitHeight
    - _Bug_Condition: slidingStep == kDigitWidth (7) instead of kQuadCharSpacing (16)_
    - _Expected_Behavior: slidingStep == kQuadCharSpacing == 16, digit_templates_ removed_
    - _Preservation: match_videopac_digit() Otsu threshold + 0.70 NCC unchanged, crop_to_grayscale() luminance formula unchanged_
    - _Requirements: 2.3, 2.4, 3.3, 3.5, 3.6, 3.7_

  - [x] 3.3 Update default ScreenRegion in `src/reward_systems.cpp`
    - Change `ScreenRegion{112, 80, 40, 14}` to `ScreenRegion{112, 80, 48, 14}` (48 / 16 = 3 digits)
    - Change `param_int(params, "screen_region_w", 40)` to `param_int(params, "screen_region_w", 48)`
    - _Bug_Condition: default width 40 with 16px step gives max_digits=2, cannot read 3-digit scores_
    - _Expected_Behavior: default width 48 gives max_digits=3_
    - _Preservation: all other ScreenRegion defaults (x=112, y=80, h=14) unchanged_
    - _Requirements: 2.5_

  - [x] 3.4 Build and verify compilation
    - Run: `cmake --build build/ci-linux --target retro_ai_native -j4`
    - Verify no compile errors or warnings from the changed files
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** — VDC Digit Detection
    - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
    - The test from task 1 encodes the expected behavior (correct digit detection for 8×14 / 16px-spaced VDC characters)
    - Run: `PYTHONPATH=build/ci-linux:python pytest tests/python/test_vision_ocr_bug.py -x`
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** — Non-Digit and Empty Input Behavior
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run: `PYTHONPATH=build/ci-linux:python pytest tests/python/test_vision_ocr_preservation.py -x`
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all preservation tests still pass after fix (no regressions in empty/blank/noise handling, thresholds, luminance, reward deltas)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 4. Checkpoint — Ensure all tests pass
  - Run full test suite: `PYTHONPATH=build/ci-linux:python pytest tests/python/ -x`
  - Verify both `test_vision_ocr_bug.py` and `test_vision_ocr_preservation.py` pass
  - Verify existing tests (`test_preprocessing.py`, `test_exceptions.py`, `test_logging.py`) still pass
  - Ensure all tests pass, ask the user if questions arise
