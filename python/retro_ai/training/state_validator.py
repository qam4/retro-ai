"""Save-state viability check.

A curriculum bag of save-states is only as useful as the states in it.
Some states are unusable — either the game clock is already at zero,
the agent is mid-jump into a snowball, or the state is frozen in a
dying animation. Training episodes starting from these states
contribute nothing: they die within the first few frames regardless
of policy.

The validator delegates to the same death rule the training environment
uses. It loads the candidate state, runs a short no-op probe, and
checks whether the env reports ``done=True`` at any point. If it does,
the state is unplayable and the validator rejects it.

Why "done from the env" and not a hand-rolled counter rule
-----------------------------------------------------------

Earlier versions of this module re-implemented the "player is dead"
detection in Python (bonus counter must drop by ``min_drop`` over
``probe_frames``). That drifted from the C++ detection the training
env actually uses (bonus unchanged for ``bonus_stall_frames`` in a row
→ ``done=True``). States could pass the Python rule and still be
terminated on the very first step in training — validator said fine,
training said dead within 20 frames.

By reading ``done`` from the env, the validator now uses the exact
same rule as training by construction.

Probe length
------------

Empirically (v9 archive, 432 cells, noop probe), ``probe_frames=120``
cleanly separates playable and unplayable states:

- 284 cells die within 120 noop frames; human review confirmed every
  sample is unplayable (already-dead, landing on a snowball, or
  falling to a lower floor with no time to react).
- 148 cells survive 120 noops; human review confirmed the sampled
  ones are playable.

Longer probes reject additional states (343/432 at 200 frames, 351/432
at 500 frames), but those extra rejected states were also confirmed
playable on video — they just have a snowball arriving further away.
A policy gets its chance to act; the validator shouldn't pre-emptively
reject states where an agent could reasonably respond.

Design notes
------------

- The validator is game-agnostic: it takes callables for loading and
  stepping, so it works for any game whose env provides a ``done``
  signal.
- Noop-only. Some directional inputs push the agent off a platform
  immediately, which would make the probe noisier. Noop lets the
  simulation run; if ``done`` doesn't fire, the state gave the agent
  time to think.
- Settle frames run before the probe and are intentionally not counted
  as "died" — the frame right after load_state can briefly look
  unusual (HUD glitches, state transitions), so we give the emulator
  a few frames to settle before trusting its ``done`` output.
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
    reason: str  # "ok" or "died_under_noop"
    first_done_frame: int | None  # 1-indexed frame where done fired, or None
    frames_probed: int  # how many probe frames we ran


def validate_state(
    state_bytes: bytes,
    load_state: Callable[[bytes], None],
    step_noop: Callable[[], bool],
    settle_frames: int = 5,
    probe_frames: int = 120,
) -> ValidationResult:
    """Run a no-op probe from ``state_bytes`` and report viability.

    Parameters
    ----------
    state_bytes
        The save-state to check.
    load_state
        Loads ``state_bytes`` into whatever env the caller owns.
    step_noop
        Advances the env one frame with a no-op action and returns the
        env's ``done`` flag. During settle frames the return value is
        ignored.
    settle_frames
        Number of no-op frames to run after load before the probe
        starts. The frame right after ``load_state`` can briefly look
        unusual (HUD glitches, state transitions) so we give the
        emulator a few frames to settle before trusting its ``done``
        signal. Default 5.
    probe_frames
        Number of no-op frames to probe. If ``done`` fires at any
        point during the probe, the state is rejected. Default 120
        based on empirical analysis of the v9 archive — see module
        docstring for the data.

    Returns
    -------
    ValidationResult
    """
    load_state(state_bytes)
    for _ in range(settle_frames):
        step_noop()
    for i in range(1, probe_frames + 1):
        done = step_noop()
        if done:
            return ValidationResult(
                viable=False,
                reason="died_under_noop",
                first_done_frame=i,
                frames_probed=i,
            )
    return ValidationResult(
        viable=True,
        reason="ok",
        first_done_frame=None,
        frames_probed=probe_frames,
    )


__all__ = ["ValidationResult", "validate_state"]
