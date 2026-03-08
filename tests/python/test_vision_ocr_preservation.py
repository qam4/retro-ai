"""
Preservation property tests for VisionRewardSystem OCR.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

Property 2: Preservation — Non-Digit and Empty Input Behavior

These tests verify that extract_score() and compute_reward() behave
correctly for inputs where the bug condition does NOT hold: empty
observations, blank framebuffers, all-white framebuffers, random noise,
and reward delta / reset semantics.

These tests MUST PASS on the current UNFIXED code — they capture
baseline behavior that must be preserved after the fix.

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
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
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
# Test 1: Empty observation → compute_reward() returns 0.0
#
# Validates: Requirement 3.1
# ---------------------------------------------------------------------------


def test_empty_observation_returns_zero_reward():
    """
    **Validates: Requirements 3.1**

    Empty observation → extract_score({}) returns -1 → compute_reward() returns 0.0.
    """
    result = _run_gtest("VisionOCRPreservation.EmptyObservationReturnsZeroReward")
    assert result.returncode == 0, (
        f"Empty observation preservation test FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 2: Blank (all-zero) framebuffer → compute_reward() returns 0.0
#
# Validates: Requirement 3.2
# ---------------------------------------------------------------------------


def test_blank_framebuffer_returns_zero_reward():
    """
    **Validates: Requirements 3.2**

    All-zero (black) framebuffer → extract_score() returns -1 → reward = 0.0.
    """
    result = _run_gtest("VisionOCRPreservation.BlankFramebufferReturnsZeroReward")
    assert result.returncode == 0, (
        f"Blank framebuffer preservation test FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 3: All-white framebuffer → compute_reward() returns 0.0
#
# Validates: Requirement 3.2
# ---------------------------------------------------------------------------


def test_all_white_framebuffer_returns_zero_reward():
    """
    **Validates: Requirements 3.2**

    All-255 (white) framebuffer → no contrast for Otsu → extract_score()
    returns -1 → reward = 0.0.
    """
    result = _run_gtest("VisionOCRPreservation.AllWhiteFramebufferReturnsZeroReward")
    assert result.returncode == 0, (
        f"All-white framebuffer preservation test FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 4: Random noise framebuffers → compute_reward() returns 0.0
#
# Validates: Requirements 3.2, 3.3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed",
    [42, 123, 999, 2024, 31415, 65535, 100000, 271828, 314159, 999999],
)
def test_random_noise_returns_zero_reward(seed: int):
    """
    **Validates: Requirements 3.2, 3.3**

    Random uint8 noise framebuffer → won't match ROM templates at 0.70
    threshold → extract_score() returns -1 → reward = 0.0.
    """
    result = _run_gtest(
        f"MultipleSeeds/VisionOCRPreservationNoise.RandomNoiseReturnsZeroReward/Seed{seed}"
    )
    assert result.returncode == 0, (
        f"Random noise preservation test FAILED (seed={seed}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 5: Reward delta preservation — no score on consecutive calls → 0.0
#
# Validates: Requirement 3.4
# ---------------------------------------------------------------------------


def test_reward_delta_preservation_no_score():
    """
    **Validates: Requirements 3.4**

    When no score is detected on consecutive calls, compute_reward()
    returns 0.0 each time.
    """
    result = _run_gtest("VisionOCRPreservation.RewardDeltaPreservationNoScore")
    assert result.returncode == 0, (
        f"Reward delta preservation test FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 6: Reset preservation — after reset(), first compute_reward() returns 0.0
#
# Validates: Requirement 3.4
# ---------------------------------------------------------------------------


def test_reset_preservation():
    """
    **Validates: Requirements 3.4**

    After reset(), has_previous_ is false, so first compute_reward()
    returns 0.0 regardless of observation content.
    """
    result = _run_gtest("VisionOCRPreservation.ResetPreservation")
    assert result.returncode == 0, (
        f"Reset preservation test FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Aggregate: run ALL preservation tests at once
# ---------------------------------------------------------------------------


def test_all_preservation_tests():
    """
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

    Run all C++ preservation tests. These MUST PASS on unfixed code.
    """
    result = _run_gtest("*Preservation*")
    assert result.returncode == 0, (
        f"Preservation tests FAILED — this is unexpected on unfixed code.\n"
        f"Test output summary:\n{result.stdout[-2000:]}"
    )
