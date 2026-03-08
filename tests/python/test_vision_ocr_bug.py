"""
Bug condition exploration test for VisionRewardSystem OCR.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4**

Property 1: Bug Condition — VDC Digit Detection Failure

These tests verify that extract_score() correctly detects Videopac digit
characters rendered at the correct Intel 8245 VDC dimensions (8px wide,
14 scanlines tall, 16px quad spacing).

On UNFIXED code these tests MUST FAIL — failure confirms the bug exists.
The buggy constants (kDigitWidth=7, kDigitHeight=10, step=7) cause the
extraction pipeline to produce undersized, misaligned patches that cannot
match the ROM templates at the 0.70 acceptance threshold.

Since VisionRewardSystem and extract_score() are not exposed to Python
bindings, we test through the compiled C++ GoogleTest binary which has
direct access to the class internals.
"""

import os
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Path to the C++ test binary (built by cmake)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build", "ci-linux")
_TEST_BINARY = os.path.join(_BUILD_DIR, "tests", "retro_ai_tests")


def _ensure_test_binary():
    """Build the C++ test binary if it doesn't exist."""
    if os.path.isfile(_TEST_BINARY):
        return
    result = subprocess.run(
        ["cmake", "--build", _BUILD_DIR, "--target", "retro_ai_tests", "-j4"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to build C++ test binary:\n{result.stderr}")


def _run_gtest(filter_pattern: str) -> subprocess.CompletedProcess:
    """Run the C++ test binary with a gtest filter and return the result."""
    _ensure_test_binary()
    return subprocess.run(
        [_TEST_BINARY, f"--gtest_filter={filter_pattern}"],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Property 1: Single digit detection (digits 0–9)
#
# For any digit d in 0–9 rendered at 8×14 with 16px quad spacing at offset 0
# in a ScreenRegion(0, 0, 48, 14) framebuffer, extract_score() should return d.
#
# On UNFIXED code: kDigitWidth=7, kDigitHeight=10, step=7 → the 7×10 patch
# misses the rightmost column and bottom 4 scanlines → match score falls
# below 0.70 → returns -1 instead of the digit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("digit", range(10))
def test_single_digit_detection(digit: int):
    """
    **Validates: Requirements 1.1, 1.2, 2.1, 2.2**

    Render digit `digit` as an 8×14 VDC character at offset 0 in the score
    region. extract_score() should detect it correctly.

    On unfixed code, the 7×10 extraction window misses pixel data, causing
    the match to fall below the 0.70 threshold.
    """
    result = _run_gtest(f"AllDigits/VisionOCRBugTest.SingleDigitDetection/Digit{digit}")
    assert result.returncode == 0, (
        f"Single digit detection FAILED for digit {digit} "
        f"(expected on unfixed code — confirms bug exists).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Property 1: Multi-digit score detection
#
# Render "123" at offsets 0, 16, 32 with 16px quad spacing.
# extract_score() should return 123.
#
# On UNFIXED code: step=7 means windows land at offsets 0, 7, 14, 21, 28, 35
# which never align with the actual digit positions at 0, 16, 32.
# ---------------------------------------------------------------------------


def test_three_digit_score_123():
    """
    **Validates: Requirements 1.3, 2.3, 2.4**

    Render score "123" at 16px quad spacing. extract_score() should return 123.

    On unfixed code, the sliding window step of 7 (instead of 16) causes
    windows to misalign with the actual character grid positions.
    """
    result = _run_gtest("VisionOCRBugMultiDigit.ThreeDigitScore123")
    assert result.returncode == 0, (
        "Three-digit score detection FAILED for '123' "
        "(expected on unfixed code — confirms bug exists).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Edge case: Score "00" detection
#
# Render "00" at offsets 0, 16. extract_score() should return 0 (not -1).
# On UNFIXED code: even the first digit at offset 0 is cropped to 7×10
# instead of 8×14, so the partial "0" pattern doesn't match.
# ---------------------------------------------------------------------------


def test_two_digit_score_00():
    """
    **Validates: Requirements 1.4, 2.4**

    Render score "00" and verify it's detected as 0 (not -1).

    On unfixed code, the undersized 7×10 patch cannot match the 8×14 VDC
    digit "0" at the 0.70 acceptance threshold.
    """
    result = _run_gtest("VisionOCRBugMultiDigit.TwoDigitScore00")
    assert result.returncode == 0, (
        "Two-digit score '00' detection FAILED "
        "(expected on unfixed code — confirms bug exists).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Aggregate: run ALL C++ bug condition tests at once
# ---------------------------------------------------------------------------


def test_all_bug_condition_tests():
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4**

    Run all C++ bug condition exploration tests. On unfixed code, most or all
    should fail, confirming the bug exists.
    """
    result = _run_gtest("*")
    assert result.returncode == 0, (
        "Bug condition exploration tests FAILED "
        "(expected on unfixed code — confirms bug exists).\n"
        f"Test output summary:\n{result.stdout[-2000:]}"
    )
