"""The `wall:` field must name WHERE a run is stuck, not just how far it got.

Motivation. On L4 v1 the aggregate scalar said "route[33]: 4/33" — true, and
useless: it does not say whether the agent is failing at the first ladder or the
last jump. The pair (last point reliably reached, first point not) is the
actionable fact, and on that run it would have read
``Lfruit_top 0.00 (from the start)`` on every progress line, which is exactly the
diagnosis that took an hour of offline analysis to reach.

Also pins the jump-landing tolerance, whose absence made Rope1 read 1.8% while the
crossing was happening ~87% of the time.
"""

import importlib.util
import pathlib
import sys

import pytest
from retro_ai.training.yeti_map import get_level_map, jump_waypoints

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location("tcc_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tcc_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _mgr(tcc, reach):
    m = tcc.CheckpointManager.__new__(tcc.CheckpointManager)
    m.waypoints = {}
    m.wp_reach_ema = dict(reach)
    return m


ORDER = get_level_map(4).route_order

# The real L4 v1 numbers at 3.0M steps.
L4_V1 = {
    "Lfruit_top": 0.00,
    "Fr1": 0.00,
    "Lascent_top": 1.00,
    "Lclimb1_top": 0.97,
    "Rope1_launch": 0.89,
    "Rope1": 0.00,
    "Lclimb2_top": 0.85,
}


def test_names_the_first_unreached_point(tcc):
    """L4 v1 never attempted the fruit trip; the wall must say so."""
    w = _mgr(tcc, L4_V1).wall(ORDER)
    assert w.startswith("Lfruit_top 0.00")
    assert "from the start" in w


def test_reports_the_last_reached_point_as_context(tcc):
    """The pair is what makes it actionable: got to X, then failed at Y."""
    reach = {w: 0.9 for w in ORDER}
    reach["Rope1"] = 0.02
    for w in ORDER[ORDER.index("Rope1") + 1 :]:
        reach[w] = 0.0
    out = _mgr(tcc, reach).wall(ORDER)
    assert out.startswith("Rope1 0.02")
    assert "after Rope1_launch 0.90" in out


def test_clear_when_everything_is_reached(tcc):
    out = _mgr(tcc, {w: 0.9 for w in ORDER}).wall(ORDER)
    assert out == f"clear to {ORDER[-1]}"


def test_handles_no_route_order(tcc):
    """L1/L2 declare no order, so there is no 'next' point to name."""
    out = _mgr(tcc, {"A": 0.3, "B": 0.7}).wall([])
    assert "no route order" in out and "B" in out


def test_empty_is_not_an_error(tcc):
    assert _mgr(tcc, {}).wall(ORDER) == "n/a"


def test_threshold_is_configurable(tcc):
    reach = {w: 0.6 for w in ORDER}
    assert _mgr(tcc, reach).wall(ORDER).startswith("clear to")
    assert _mgr(tcc, reach).wall(ORDER, threshold=0.7).startswith(ORDER[0])


def test_jump_waypoints_get_a_wider_tolerance_than_ladders():
    """A ladder is a single x_ram; a jump landing is wherever the arc drops you,
    and its waypoint sits on the platform edge. One tolerance for both made the
    L4 rope crossing invisible."""
    jw = set(jump_waypoints(get_level_map(4)))
    assert "Rope1" in jw and "Rope1_launch" in jw
    # Ladder waypoints must NOT be in the widened set.
    assert "Lfruit_top" not in jw
    assert "Lprincess_top" not in jw
