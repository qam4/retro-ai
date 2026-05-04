"""Tests for the save-state viability check.

We don't spin up the real emulator here — the validator takes the
env interactions as callables, so a tiny fake counter is enough to
cover the branches.
"""

from __future__ import annotations

from retro_ai.training.state_validator import ValidationResult, validate_state


class _FakeCounter:
    """Counter that starts at ``start`` and drops by ``drop_per_frame``
    on each ``step`` call, never going below 0.
    """

    def __init__(self, start: int, drop_per_frame: int) -> None:
        self.value = start
        self._drop = drop_per_frame
        self.load_calls = 0
        self.step_calls = 0

    def load(self, _state: bytes) -> None:
        self.load_calls += 1

    def step(self) -> None:
        self.step_calls += 1
        self.value = max(0, self.value - self._drop)

    def read(self) -> int:
        return self.value


# ---------------------------------------------------------------------------
# Basic plumbing
# ---------------------------------------------------------------------------


def test_viable_state_with_healthy_counter_drop() -> None:
    """Counter starts well above zero and drops steadily → viable."""
    c = _FakeCounter(start=1000, drop_per_frame=1)
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=c.step,
        read_counter=c.read,
        settle_frames=5,
        probe_frames=30,
        min_drop=2,
    )
    assert isinstance(result, ValidationResult)
    assert result.viable is True
    assert result.reason == "ok"
    # load happened once; step happened settle + probe times
    assert c.load_calls == 1
    assert c.step_calls == 35


def test_validator_returns_counter_values() -> None:
    """ValidationResult reports the observed counter values."""
    c = _FakeCounter(start=500, drop_per_frame=1)
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=c.step,
        read_counter=c.read,
        settle_frames=5,
        probe_frames=30,
        min_drop=2,
    )
    # After 5 settle frames counter is 495, after 30 more it's 465
    assert result.bonus_at_load == 495
    assert result.bonus_at_end == 465
    assert result.frames_probed == 30


# ---------------------------------------------------------------------------
# Rejection: bonus_zero
# ---------------------------------------------------------------------------


def test_bonus_zero_at_load_is_rejected() -> None:
    """A state that loads with a zero counter is rejected immediately."""
    c = _FakeCounter(start=0, drop_per_frame=0)
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=c.step,
        read_counter=c.read,
    )
    assert result.viable is False
    assert result.reason == "bonus_zero"
    assert result.bonus_at_load == 0
    # we don't bother probing an already-zero state
    assert result.frames_probed == 0


def test_bonus_reaches_zero_during_settle_is_rejected() -> None:
    """If the counter runs out during the settle frames, reject."""
    # start=3, drop=1 per frame, settle=5 → counter hits 0 by frame 3
    c = _FakeCounter(start=3, drop_per_frame=1)
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=c.step,
        read_counter=c.read,
        settle_frames=5,
    )
    assert result.viable is False
    assert result.reason == "bonus_zero"


# ---------------------------------------------------------------------------
# Rejection: bonus_frozen
# ---------------------------------------------------------------------------


def test_frozen_counter_is_rejected() -> None:
    """Counter that doesn't drop during the probe is rejected."""
    c = _FakeCounter(start=500, drop_per_frame=0)
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=c.step,
        read_counter=c.read,
    )
    assert result.viable is False
    assert result.reason == "bonus_frozen"
    assert result.bonus_at_load == 500
    assert result.bonus_at_end == 500


def test_single_tick_drop_is_rejected_with_default_min_drop() -> None:
    """Default ``min_drop=2`` rejects a state that only ticks once then freezes.

    Matches the behavior we saw in B's CP4: bonus went 767->766 once then
    stuck. The old validator accepted any change; the new one requires
    a more substantial drop.
    """
    class OneTickThenFreeze:
        def __init__(self) -> None:
            self.value = 500
            self.ticked = False
        def load(self, _b: bytes) -> None: pass
        def step(self) -> None:
            if not self.ticked:
                self.value -= 1
                self.ticked = True
        def read(self) -> int:
            return self.value

    c = OneTickThenFreeze()
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=c.step,
        read_counter=c.read,
        settle_frames=0,
        probe_frames=30,
        min_drop=2,
    )
    assert result.viable is False
    assert result.reason == "bonus_frozen"


# ---------------------------------------------------------------------------
# Threshold knob
# ---------------------------------------------------------------------------


def test_min_drop_one_accepts_slow_counter() -> None:
    """With ``min_drop=1`` a counter dropping once every ~30 frames passes."""
    c = _FakeCounter(start=500, drop_per_frame=0)
    # Simulate a single drop at the tail of the probe.
    original_step = c.step
    calls = {"n": 0}
    def step_once_then_drop() -> None:
        calls["n"] += 1
        if calls["n"] == 30:
            c.value -= 1
        original_step()
    result = validate_state(
        state_bytes=b"unused",
        load_state=c.load,
        step_noop=step_once_then_drop,
        read_counter=c.read,
        settle_frames=0,
        probe_frames=30,
        min_drop=1,
    )
    assert result.viable is True
    assert result.reason == "ok"
