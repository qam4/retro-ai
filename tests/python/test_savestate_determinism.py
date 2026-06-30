"""Determinism of save_state/load_state.

Round-trip: run a trajectory under a FIXED action sequence; save at step K;
then load that save and replay actions[K:]. We check two properties:

1. RAM determinism (test_savestate_ram_roundtrip_deterministic) — the full
   CPU-space RAM must match bit-for-bit at every replayed step. This is the
   property the checkpoint/curriculum machinery and all our RAM-based rewards
   actually rely on. It PASSES, proving save_state captures the CPU/RAM state
   needed for a deterministic resume.

2. Rendered-observation determinism (test_savestate_observation_roundtrip) —
   the rendered frame must also match. This previously failed: the MO5
   character font lives in the monitor ROM, and ``MemorySystem::set_state``
   used to copy the whole memory struct on load — overwriting the live ROMs
   with the deserialized (zero-filled, non-serialized) ROM arrays. With the
   font wiped, HUD/text glyphs rendered as blanks after load_state. Fixed by
   preserving the immutable ROMs across set_state; this test now guards it.

Skips if the native build / ROM env isn't present (Python-only dev machines).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_ACTIONS = [
    [0, 0, 0],
    [0, 2, 0],
    [0, 2, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 2, 0],
    [0, 2, 0],
    [0, 0, 0],
]


@pytest.fixture
def base():
    try:
        from retro_ai.training.env_builder import build_training_env
        from retro_ai.training.run_config import EnvConfig
    except Exception as e:  # pragma: no cover
        pytest.skip(f"training env not importable: {e}")

    if not os.environ.get("RETRO_AI_ROM_DIR"):
        pytest.skip("RETRO_AI_ROM_DIR not set (integration test)")

    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=10**9,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    return build_training_env("yeti_fruit", cfg).base


def _ram(b) -> bytes:
    return bytes(b._interface.read_ram())


def _obs(b) -> np.ndarray:
    assert b._last_raw_obs is not None, "no observation cached after step"
    return np.asarray(b._last_raw_obs).copy()


def _actions(n: int):
    return [_ACTIONS[i % len(_ACTIONS)] for i in range(n)]


def _replay(b, *, record):
    """Run A (record per-step) then run B (load mid-save, replay).

    ``record(b)`` returns the value compared per step. Returns
    (values_run_a[k:], values_run_b) as two equal-length lists.
    """
    n, k = 80, 30
    acts = _actions(n)
    b.reset(seed=0)
    saved = None
    a_vals = []
    for i in range(n):
        if i == k:
            saved = b.save_state()
        b.step(acts[i])
        if i >= k:
            a_vals.append(record(b))
    b.reset(seed=0)
    b.load_state(saved)
    b_vals = []
    for i in range(k, n):
        b.step(acts[i])
        b_vals.append(record(b))
    return a_vals, b_vals


def test_savestate_ram_roundtrip_deterministic(base):
    """Full RAM matches bit-for-bit after a reloaded save + replay."""
    a_vals, b_vals = _replay(base, record=_ram)
    for j, (ra, rb) in enumerate(zip(a_vals, b_vals)):
        assert ra == rb, (
            f"RAM diverged {j} steps after a reloaded save — save_state is "
            f"missing CPU/RAM state needed for a deterministic resume."
        )


def test_savestate_observation_roundtrip(base):
    """Rendered frame matches after a reloaded save + replay.

    Regression guard for the monitor-ROM wipe on load (the character font
    lives in the monitor ROM; wiping it blanked HUD/text glyphs after load).
    """
    a_vals, b_vals = _replay(base, record=_obs)
    for j, (oa, ob) in enumerate(zip(a_vals, b_vals)):
        assert np.array_equal(oa, ob), f"observation diverged at replay step {j}"
