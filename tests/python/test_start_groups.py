"""Start groups: one run trains from several level starts without mixing them.

A group is a CheckpointManager plus its level start. These pin the four things the
design rests on: a single-group run is unchanged (no extra random draw), groups are
drawn by 1 - goal score with no fixed share, a group the run cannot serve is refused,
and each group keeps its own pools and reach evidence.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.util
import random
from pathlib import Path

import pytest
from retro_ai.training.run_config import RunConfig

REPO = Path(__file__).resolve().parents[2]
_SRC = REPO / "scripts/mo5/yeti/train_checkpoint_curriculum.py"
_L1_CFG = (
    REPO
    / "experiments/003-yeti/configs/yeti_curriculum_l1_v20_l4settings_warm_v17_6m.yaml"
)


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location("tcc_groups", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - ML deps absent (CI)
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _cfg(groups):
    cfg = RunConfig.from_yaml(str(_L1_CFG))
    return dataclasses.replace(
        cfg, curriculum=dataclasses.replace(cfg.curriculum, start_groups=list(groups))
    )


def _mgr(tcc, cfg):
    ladder, rungs = tcc._progress_ladder(cfg)
    return tcc._make_manager(cfg, ladder, rungs)


@pytest.fixture
def clean_groups(tcc):
    yield
    tcc._groups.clear()
    tcc._group_starts.clear()


def test_a_single_group_run_draws_nothing_from_random(tcc, clean_groups):
    """Every pick_start draw after the group choice must see the same random stream
    it saw before groups existed, or a single-group run is not the same run."""
    cfg = _cfg([])
    tcc._groups[:] = [tcc.StartGroup("", _mgr(tcc, cfg), None, "checkpoints.pkl")]
    random.seed(3)
    assert tcc._pick_group() == 0
    after = random.random()
    random.seed(3)
    assert after == random.random()


def test_groups_are_drawn_by_one_minus_the_level_start_goal_score(tcc, clean_groups):
    """At today's measured goal scores (L1 0.851, L5 0.312 from the 300-episode
    evaluations) L1 gets ~18% of starts with no constant; when L1 slips it gets more."""
    cfg = _cfg([1, 5])
    l1, l5 = _mgr(tcc, cfg), _mgr(tcc, cfg)
    tcc._groups[:] = [
        tcc.StartGroup("L1", l1, None, "checkpoints.pkl"),
        tcc.StartGroup("L5", l5, b"save", "checkpoints_L5.pkl"),
    ]

    def l1_share(l1_goal, n=20000):
        l1.checkpoints[0].goal_score = l1_goal
        l5.checkpoints[0].goal_score = 0.312
        random.seed(11)
        c = collections.Counter(tcc._pick_group() for _ in range(n))
        return c[0] / n

    expected = 0.149 / (0.149 + 0.688)
    assert abs(l1_share(0.851) - expected) < 0.01
    assert l1_share(0.6) > l1_share(0.851) + 0.1, "a slipping level must get more"


def test_specs_without_groups_are_the_runs_own_start(tcc):
    assert tcc._start_group_specs(_cfg([])) == [("", None)]


def test_specs_resolve_levels_through_the_registry(tcc):
    assert tcc._start_group_specs(_cfg([1, 5])) == [
        ("L1", None),
        ("L5", "output/mo5/yeti/level5/level5_start.sav"),
    ]


@pytest.mark.parametrize(
    "groups, message",
    [
        ([1, 2], "layout 2"),  # L2 needs another reward geometry and profile
        ([5, 1], "first start group must be the run's own start"),
        ([1, 1], "twice"),
        ([1, 99], "game level 99"),
    ],
)
def test_specs_refuse_what_one_run_cannot_serve(tcc, groups, message):
    with pytest.raises(SystemExit, match=message):
        tcc._start_group_specs(_cfg(groups))


def test_resume_keeps_fruit_pools_and_drops_removed_waypoints(tcc, capsys):
    """Fruit pools are named F1.. and are not in yeti.waypoints; the stale check used
    to drop all of them on every resume (v17's file holds 100 states in each)."""
    cfg = _cfg([])
    m = _mgr(tcc, cfg)
    for name in ("F1", "F4", "L12a_top", "Gone_launch"):
        m.save_waypoint(name, b"s", 100, True, source_cp=0, bonus=0)
    tcc._drop_stale_pools(m, cfg)
    assert sorted(m.waypoints) == ["F1", "F4", "L12a_top"]
    assert "['Gone_launch']" in capsys.readouterr().out


def test_each_group_keeps_its_own_pools_and_reach(tcc):
    """L1 and L5 share waypoint names; their states and evidence must not mix."""
    cfg = _cfg([1, 5])
    l1, l5 = _mgr(tcc, cfg), _mgr(tcc, cfg)
    assert l1.waypoint_anchors == l5.waypoint_anchors  # same layout, same anchors
    l5.save_waypoint("L12a_top", b"l5-state", 100, True, source_cp=0, bonus=0)
    assert "L12a_top" in l5.waypoints and "L12a_top" not in l1.waypoints
    # A root episode in L5 that reached the princess moves only L5's evidence.
    l5.record_episode(
        0, l5.N_RUNGS + 1, reached_wps={"L12a_top"}, all_wps={"L12a_top"}, start_rung=0
    )
    assert l5.checkpoints[0].goal_score > 0 and l1.checkpoints[0].goal_score == 0
    assert l5.wp_reach_ema.get("L12a_top", 0) > 0
    assert "L12a_top" not in l1.wp_reach_ema
    assert l1.reset_reach_ema[-1] == 0 and l5.reset_reach_ema[-1] > 0
