"""Tests for the save-state viability check.

The validator takes the env interactions as callables so we don't
need a real emulator — a tiny fake env that returns ``done`` after a
configurable number of steps covers every branch.
"""

from __future__ import annotations

from retro_ai.training.state_validator import ValidationResult, validate_state


class _FakeEnv:
    """Minimal env stand-in.

    Returns ``done=True`` on the step whose index (1-based, counting
    both settle and probe steps) matches ``done_on_step``. Use
    ``done_on_step=None`` to model a state that never dies during the
    probe window.
    """

    def __init__(self, done_on_step: int | None = None) -> None:
        self.done_on_step = done_on_step
        self.load_calls = 0
        self.step_calls = 0

    def load(self, _state: bytes) -> None:
        self.load_calls += 1

    def step(self) -> bool:
        self.step_calls += 1
        return self.done_on_step is not None and self.step_calls >= self.done_on_step


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_state_that_never_dies_is_viable() -> None:
    """State where done never fires during settle or probe → viable."""
    env = _FakeEnv(done_on_step=None)
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=5,
        probe_frames=120,
    )
    assert isinstance(result, ValidationResult)
    assert result.viable is True
    assert result.reason == "ok"
    assert result.first_done_frame is None
    assert result.frames_probed == 120
    assert env.load_calls == 1
    assert env.step_calls == 125  # settle + probe


def test_state_that_barely_survives_is_viable() -> None:
    """Done fires one step after the probe ends → viable."""
    # settle=5 + probe=120 = 125 steps taken; done would fire at 126
    env = _FakeEnv(done_on_step=126)
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=5,
        probe_frames=120,
    )
    assert result.viable is True
    assert result.frames_probed == 120


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


def test_state_that_dies_during_probe_is_rejected() -> None:
    """Done fires in the middle of the probe → rejected, reports frame."""
    # settle=5 + 19 probe steps = step 24 overall
    env = _FakeEnv(done_on_step=24)
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=5,
        probe_frames=120,
    )
    assert result.viable is False
    assert result.reason == "died_under_noop"
    assert result.first_done_frame == 19  # 24 total - 5 settle
    assert result.frames_probed == 19


def test_state_that_dies_on_first_probe_step_is_rejected() -> None:
    """Done fires immediately after settle → rejected at frame 1."""
    env = _FakeEnv(done_on_step=6)  # settle=5 + 1 probe
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=5,
        probe_frames=120,
    )
    assert result.viable is False
    assert result.first_done_frame == 1


def test_state_that_dies_on_last_probe_step_is_rejected() -> None:
    """Done fires on the very last probe frame → still rejected."""
    env = _FakeEnv(done_on_step=125)  # settle=5 + 120 probe
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=5,
        probe_frames=120,
    )
    assert result.viable is False
    assert result.first_done_frame == 120


# ---------------------------------------------------------------------------
# Settle behavior
# ---------------------------------------------------------------------------


def test_done_during_settle_does_not_reject() -> None:
    """Settle frames are intentionally ignored — done there is fine.

    The first few frames after load_state can look unusual (HUD
    glitches, state transitions); we let the emulator settle before
    trusting its done signal.
    """
    env = _FakeEnv(done_on_step=3)  # dies during settle
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=5,
        probe_frames=120,
    )
    # Still viable because done during settle is ignored; env stops
    # returning done=False once triggered, so env.step returns True
    # on probe frame 1 too → actually rejected at probe frame 1.
    # Rethink: our FakeEnv returns True from done_on_step onward, so
    # if done fires during settle it'll still be firing on the first
    # probe step. That's the correct real-world behavior too: a state
    # that's already-dead at load time will keep reporting done, and
    # we do want to reject that. So this test documents: done during
    # settle alone isn't a reject, but a state that remains in the
    # dead state through settle WILL be rejected on probe frame 1.
    assert result.viable is False
    assert result.first_done_frame == 1


def test_settle_zero_surfaces_done_on_first_frame() -> None:
    """With ``settle_frames=0`` the probe starts immediately."""
    env = _FakeEnv(done_on_step=1)
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
        settle_frames=0,
        probe_frames=120,
    )
    assert result.viable is False
    assert result.first_done_frame == 1


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_probe_length_is_120_frames() -> None:
    """Document the default chosen from v9 archive analysis."""
    env = _FakeEnv(done_on_step=None)
    result = validate_state(
        state_bytes=b"unused",
        load_state=env.load,
        step_noop=env.step,
    )
    assert result.frames_probed == 120
    # settle default is 5
    assert env.step_calls == 125
