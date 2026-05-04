"""Save-state viability check.

A curriculum bag of save-states is only as useful as the states in it.
Some states are unusable — either the game clock is already at zero,
or it's frozen in an animation the agent can't interrupt. Training
episodes starting from these states contribute nothing: they die fast
(or stall forever) regardless of the policy.

The validator runs a short no-op probe from a loaded state and checks
two things on the in-game countdown counter ("bonus" in Yeti):

- At load time the counter must be non-zero. A zero counter means
  the state has no time left, and the game will end almost
  immediately regardless of input.
- Across a small number of no-op frames, the counter must drop by at
  least ``min_drop``. If it doesn't, the game clock is frozen — the
  state is mid-animation or otherwise non-interactive — and the agent
  has no effective control.

The first check is necessary because a zero counter looks superficially
fine (bonus==0 is technically "unchanged" too) but is semantically
different from "the clock is frozen above zero".

Design notes
------------

- The validator is game-agnostic: it takes callables for loading the
  state and reading the counter, so it also works for any other game
  that has a countdown we can read from RAM.
- Noop-only. Random actions produce noisier survival data in Yeti
  (some directional inputs immediately push the agent off a platform)
  which makes them a bad probe for state viability. Noops just let
  the simulation run; if the clock is advancing, the state is alive.
- Doesn't touch the caller's env state. The caller is expected to
  save the env's current state before validating and restore it
  after, since we load the candidate state into the same env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one save-state.

    ``viable`` is the summary bit the rest of the system uses. Other
    fields are preserved so callers can explain WHY a state failed.
    """

    viable: bool
    reason: str                 # "ok", "bonus_zero", or "bonus_frozen"
    bonus_at_load: int          # the counter after loading + settle
    bonus_at_end: int           # the counter after the noop probe
    frames_probed: int          # how many noop frames we ran


def validate_state(
    state_bytes: bytes,
    load_state: Callable[[bytes], None],
    step_noop: Callable[[], None],
    read_counter: Callable[[], int],
    settle_frames: int = 5,
    probe_frames: int = 30,
    min_drop: int = 2,
) -> ValidationResult:
    """Run a no-op probe from ``state_bytes`` and report viability.

    Parameters
    ----------
    state_bytes
        The save-state to check.
    load_state
        Loads ``state_bytes`` into whatever env the caller owns.
    step_noop
        Advances the env one frame with a no-op action.
    read_counter
        Returns the current value of the in-game countdown
        (``bonus`` for Yeti).
    settle_frames
        Number of no-op frames to run after load before reading the
        starting counter value. Absorbs any mid-transition RAM state
        right after load.
    probe_frames
        Number of no-op frames used to measure the counter's drop.
    min_drop
        Minimum required drop over ``probe_frames`` for the state to
        be viable. Default 2 tolerates a 1-tick transient and still
        requires genuine clock progress.

    Returns
    -------
    ValidationResult
    """
    load_state(state_bytes)
    for _ in range(settle_frames):
        step_noop()
    start = read_counter()
    if start == 0:
        return ValidationResult(
            viable=False,
            reason="bonus_zero",
            bonus_at_load=0,
            bonus_at_end=0,
            frames_probed=0,
        )
    for _ in range(probe_frames):
        step_noop()
    end = read_counter()
    drop = start - end
    if drop < min_drop:
        return ValidationResult(
            viable=False,
            reason="bonus_frozen",
            bonus_at_load=start,
            bonus_at_end=end,
            frames_probed=probe_frames,
        )
    return ValidationResult(
        viable=True,
        reason="ok",
        bonus_at_load=start,
        bonus_at_end=end,
        frames_probed=probe_frames,
    )


__all__ = ["ValidationResult", "validate_state"]
