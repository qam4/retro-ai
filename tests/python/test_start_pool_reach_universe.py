"""Every pool the start gate judges must have a row in the reach table.

THE BUG THIS PINS. `_wp_eligible` decides whether a pool may seed an episode by looking
its from-reset reach up in `wp_reach_ema`. That table was built over the POSITIONAL
DETECTOR universe (`self._waypoints`), while the gate judges the START-POOL universe
(`_manager.waypoints`). A pool in the second but not the first gets no row, reads 0.0,
fails the reach gate, falls through to the `route_order` predecessor rule, is absent
from that too, and is refused forever.

Measured on L4, where the sets differed by exactly `F1` -- the fruit pool, created when
fruit seeds moved out of the rung-keyed checkpoint slots:

    v24  fruit seeds in checkpoint slot '3'  ->  925 starts of 24121 episodes (3.8%)
    v26  same seeds in the F1 waypoint pool  ->    0 starts of 28137 episodes
         F1 held 100 seeds and logged 10439 captures, and was never sampled once

Downstream, per-waypoint from-reset reach fell a flat 0.08-0.12 across the whole route
and the deep half's share of starts dropped from 24.3% to 14.6%.

Nothing here is about fruit. The invariant is that the reach table is keyed by what is
gated, so a pool named anything other than a positional waypoint is tracked instead of
silently frozen out.

No emulator needed.
"""

import importlib.util
import pathlib

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)


def _mod():
    spec = importlib.util.spec_from_file_location("_yeti_reachuniverse", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


class _Pool:
    """Minimal stand-in for StartPool: non-empty is all the gate looks at."""

    def __init__(self, n=1):
        self.n = n

    def __len__(self):
        return self.n

    def weight(self):
        return 1.0


def _manager_with(mod, pools, reach):
    """A CheckpointManager with its gate inputs set, bypassing __init__."""
    m = object.__new__(mod.CheckpointManager)
    m.waypoints = {k: _Pool() for k in pools}
    m.wp_reach_ema = dict(reach)
    m.gate_waypoints = True
    m.gate_waypoints_by_predecessor = True
    m.reach_threshold = 0.15
    from retro_ai.training.yeti_map import get_level_map

    m.route_order = list(get_level_map(4).route_order or [])
    return m


def test_pool_absent_from_route_order_is_refused_without_a_reach_row():
    """The failure mode itself: no reach row and not on the route => never eligible.

    This is what happened to `F1` for a whole 6M run. Kept as a test so the shape of the
    bug is executable: it is the reason the reach table must cover the pools.
    """
    mod = _mod()
    m = _manager_with(mod, ["Fr2", "F1"], {"Fr2": 0.9})
    assert m._wp_eligible("Fr2") is True
    assert m._wp_eligible("F1") is False, (
        "F1 with no reach row should be refused -- if this passes, the gate changed "
        "and the test below is the one that matters"
    )


def test_a_reach_row_makes_the_pool_eligible_on_its_own_evidence():
    """THE FIX. Give the pool a row from reset-origin evidence and it gates itself.

    98% is what L4's fruit actually measures from reset, so the primary reach test
    passes and the `route_order` predecessor fallback is never consulted -- which is why
    this fix leaves `route_order` (and therefore the evaluator's frontier scoring)
    untouched.
    """
    mod = _mod()
    m = _manager_with(mod, ["Fr2", "F1"], {"Fr2": 0.9, "F1": 0.98})
    assert m._wp_eligible("F1") is True


def test_reach_universe_passed_to_record_episode_covers_the_pools():
    """The call site must hand over the union, not just the positional detectors.

    Guards the actual regression: `all_wps` is what creates rows in `wp_reach_ema`, so
    if it omits a pool that pool can never become eligible however often the agent
    reaches it.
    """
    src = _SRC.read_text()
    i = src.index("_manager.record_episode(")
    call = src[i : src.index(")", src.index("start_rung=", i))]
    assert "_reach_universe" in call, "all_wps must be the widened universe"
    assert "_reached_targets(" in call, (
        "reached_wps must fold in collected fruits, else a pool's row decays to 0 "
        "even once it exists"
    )
    assert (
        "set(self._waypoints.keys()) | _pool_ids" in src
    ), "the universe must be detectors UNION start pools"
