"""Tests for PreprocessedEnv frame-stack save/restore (H-AB).

A save-state start should restore the REAL motion history captured with the
seed (on-distribution), not reseed the stack with copies of one frozen frame.
These tests use a tiny fake env (no emulator) so they run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline


class _FakeEnv:
    """Minimal env returning a solid-colour RGB frame whose value is
    ``next_val`` (so a stack encodes a known 'motion history')."""

    def __init__(self):
        self.next_val = 0

    def _frame(self):
        return np.full((2, 2, 3), self.next_val, dtype=np.uint8)

    def reset(self, seed=None):
        return self._frame(), {}

    def step(self, action):
        return self._frame(), 0.0, False, False, {}


def _penv(frame_stack=3):
    # grayscale of a solid value v -> a (2,2,1) frame of value v, so the
    # stacked observation directly encodes the sequence of frame values.
    pipe = PreprocessingPipeline(
        grayscale=True, resize=None, frame_stack=frame_stack, frame_skip=1
    )
    return PreprocessedEnv(_FakeEnv(), pipe, frame_maxpool=False), pipe


def test_export_then_restore_reproduces_observation():
    penv, _ = _penv(frame_stack=3)
    env = penv.env
    env.next_val = 10
    penv.reset()  # stack seeded with [10,10,10]
    for v in (20, 30, 40):
        env.next_val = v
        penv.step([0])
    # Stack now holds the real motion history [20,30,40].
    blob = penv.export_frame_stack()
    assert blob is not None and len(blob["frames"]) == 3
    obs_at_export = penv.current_observation().copy()

    # Advance past it (simulating other episodes reusing this env).
    env.next_val = 99
    penv.step([0])
    assert not np.array_equal(penv.current_observation(), obs_at_export)

    # Restore -> the observation is byte-identical to the captured history.
    assert penv.restore_frame_stack(blob) is True
    np.testing.assert_array_equal(penv.current_observation(), obs_at_export)


def test_restore_rejects_signature_mismatch():
    penv, _ = _penv(frame_stack=3)
    penv.env.next_val = 5
    penv.reset()
    blob = penv.export_frame_stack()
    blob["sig"] = ("totally", "different")
    assert penv.restore_frame_stack(blob) is False


def test_restore_rejects_missing_or_wrong_length():
    penv, _ = _penv(frame_stack=3)
    penv.env.next_val = 5
    penv.reset()
    assert penv.restore_frame_stack(None) is False
    good = penv.export_frame_stack()
    bad = {"sig": good["sig"], "frames": good["frames"][:2]}  # wrong length
    assert penv.restore_frame_stack(bad) is False


def test_export_none_without_stack():
    # frame_stack == 1 -> no stack buffer -> nothing to export.
    penv, _ = _penv(frame_stack=1)
    penv.env.next_val = 1
    penv.reset()
    assert penv.export_frame_stack() is None


def test_restored_stack_matches_a_live_continuation():
    """The whole point: restoring == being mid-episode. Two envs fed the
    SAME frame sequence must yield the SAME observation, one via a
    continuous run and one via export->restore."""
    seq = [11, 22, 33, 44, 55]
    # Continuous run.
    live, _ = _penv(frame_stack=3)
    live.env.next_val = seq[0]
    live.reset()
    for v in seq[1:]:
        live.env.next_val = v
        live.step([0])
    live_obs = live.current_observation().copy()

    # Run to the capture point, export, then restore into a disturbed env.
    cap, _ = _penv(frame_stack=3)
    cap.env.next_val = seq[0]
    cap.reset()
    for v in seq[1:]:
        cap.env.next_val = v
        cap.step([0])
    blob = cap.export_frame_stack()
    cap.env.next_val = 123
    cap.step([0])  # disturb
    assert cap.restore_frame_stack(blob) is True
    np.testing.assert_array_equal(cap.current_observation(), live_obs)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
