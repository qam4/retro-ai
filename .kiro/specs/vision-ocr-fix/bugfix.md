# Bugfix Requirements Document

## Introduction

The `VisionRewardSystem` OCR fails to detect Videopac digit characters because the template dimensions (`kDigitWidth=7`, `kDigitHeight=10`) and sliding window step (`7`) do not match the actual Intel 8245 VDC character rendering dimensions (8px wide, 14 scanlines tall, 16px quad spacing). This causes `extract_score()` to return -1 or incorrect scores, breaking vision-based reward computation for all Videopac games that use score display.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the VisionRewardSystem crops a digit patch from the framebuffer THEN the system uses `kDigitWidth=7` producing a 7px-wide patch that is 1 pixel too narrow, cutting off the rightmost column of the 8px-wide VDC character

1.2 WHEN the VisionRewardSystem crops a digit patch from the framebuffer THEN the system uses `kDigitHeight=10` producing a 10px-tall patch that is 4 scanlines too short, missing the bottom 4 scanlines of the 14-scanline VDC character

1.3 WHEN the VisionRewardSystem slides across the score region to find successive digits THEN the system advances by `step = kDigitWidth = 7` pixels instead of 16 pixels (the quad character spacing), causing digit windows to misalign with the actual character grid positions

1.4 WHEN `extract_score()` is called with a framebuffer containing valid Videopac digit characters THEN the system returns -1 (no digits detected) or an incorrect score because the undersized, misaligned patches do not match the ROM templates at the 0.70 acceptance threshold

1.5 WHEN the default ScreenRegion `{112, 80, 40, 14}` is used with the corrected digit dimensions THEN the region height (14) is exactly one digit tall but the region width (40) does not evenly accommodate digits at 16px spacing (40 / 16 = 2.5), potentially misaligning the third digit position

### Expected Behavior (Correct)

2.1 WHEN the VisionRewardSystem crops a digit patch from the framebuffer THEN the system SHALL use a digit width of 8 pixels to match the full VDC character width (7 active ROM columns + 1 blank column)

2.2 WHEN the VisionRewardSystem crops a digit patch from the framebuffer THEN the system SHALL use a digit height of 14 scanlines to match the VDC doubled-scanline rendering (7 ROM rows × 2 scanlines per row)

2.3 WHEN the VisionRewardSystem slides across the score region to find successive digits THEN the system SHALL advance by 16 pixels per step to match the quad character spacing (8px character + 8px gap)

2.4 WHEN `extract_score()` is called with a framebuffer containing valid Videopac digit characters THEN the system SHALL correctly detect and return the numeric score by matching correctly-sized 8×14 patches against the ROM templates

2.5 WHEN the default ScreenRegion is used THEN the region dimensions SHALL accommodate the corrected digit dimensions and quad spacing so that all expected digit positions fall within the region bounds

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `extract_score()` is called with an empty observation THEN the system SHALL CONTINUE TO return -1

3.2 WHEN `extract_score()` is called with a framebuffer containing no digit characters (blank or non-digit content) THEN the system SHALL CONTINUE TO return -1 without false positives

3.3 WHEN `match_videopac_digit()` compares a patch against the ROM patterns THEN the system SHALL CONTINUE TO use Otsu thresholding for binarization and the 0.70 normalized cross-correlation acceptance threshold

3.4 WHEN `compute_reward()` is called THEN the system SHALL CONTINUE TO return the delta between the current and previous detected scores, returning 0.0 when no score is visible or on the first detection

3.5 WHEN `crop_to_grayscale()` converts the observation region to grayscale THEN the system SHALL CONTINUE TO use the luminance formula (0.299R + 0.587G + 0.114B) and respect observation space bounds

3.6 WHEN the `videopac_digit_rom[10][8]` ROM data is used for template matching THEN the system SHALL CONTINUE TO use the existing correct Intel 8245 ROM patterns without modification

3.7 WHEN `match_videopac_digit()` scales ROM patterns to patch dimensions THEN the system SHALL CONTINUE TO map patch rows to ROM rows 0-6 (skipping blank row 7) and patch columns to ROM columns 0-7 (MSB-first)
