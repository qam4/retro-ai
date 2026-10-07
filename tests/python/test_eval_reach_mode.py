"""The evaluator must decide "reached" the way the trainer does.

`yeti_rollout` claimed to mirror the trainer's detection exactly while using the box
test and a grounded-only pose gate. The trainer moved to sprite overlap with a fail-open
pose gate in de21939 and the rollout never followed. It went unnoticed until an agent
finally crossed L4's rope 2: v30's champion touched the princess in 208 of 300 episodes
and every one recorded `Low2` as unreached and `max_rung` 11 instead of 12, because the
crossing FLIES over `Low2`'s anchor (px 128), lands at px 88 and walks left -- it is
never grounded inside the box.

The frames below are from the scripted rope-2 crossing traced on 2026-09-29.

No emulator needed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import itertools
import pathlib

import pytest
from retro_ai.games import yeti
from retro_ai.games.yeti_rollout import rollout_episode, waypoint_frame_reaches
from retro_ai.training.run_config import CurriculumConfig
from retro_ai.training.targets import within_tol

LOW2 = yeti.waypoints(4)["Low2"][:2]  # (x_ram 30, y 70), px 128 on floor 13
JUMP_LEFT, FALL, WALK_LEFT = 10, 11, 4
SEED = frozenset(yeti.SURFACE_POSES | {13})


def _both(x, y, pose, tol=6):
    return (
        waypoint_frame_reaches(LOW2, x, y, pose, tol, "box", seed_poses=SEED),
        waypoint_frame_reaches(LOW2, x, y, pose, tol, "sprite", seed_poses=SEED),
    )


def test_the_rope2_flyover_counts_under_sprite_and_not_box():
    """Crossing frame 42: px 128, y 58, pose 10 -- over the anchor, airborne."""
    box, sprite = _both(30, 58, JUMP_LEFT)
    assert sprite, "the trainer counts this frame; the evaluator must too"
    assert not box, "box mode must stay what it was, for reproducing old evals"


def test_the_rope2_landing_counts_under_neither():
    """Touchdown at px 88 (x_ram 20), grounded: outside the box and the sprite."""
    assert _both(20, 70, WALK_LEFT) == (False, False)


def test_falling_past_the_anchor_never_counts():
    assert _both(30, 58, FALL) == (False, False)


def test_standing_on_the_anchor_counts_under_both():
    assert _both(30, 70, WALK_LEFT) == (True, True)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        waypoint_frame_reaches(LOW2, 30, 70, WALK_LEFT, 6, "bbox")


def test_box_mode_is_identical_to_the_old_rollout_rule():
    """'box' exists to reproduce evals made before the fix, so it must be exact.

    The old loop was ``pose in seed_poses and within_tol(pos, x, y, tol)``.
    """
    for x, y, pose, tol in itertools.product(
        range(20, 41, 2), range(50, 91, 4), range(0, 18), (2, 6)
    ):
        old = pose in SEED and within_tol(LOW2, x, y, tol)
        new = waypoint_frame_reaches(LOW2, x, y, pose, tol, "box", seed_poses=SEED)
        assert new == old, (x, y, pose, tol)


def test_eval_default_matches_the_trainer_default():
    """The drift guard. If the trainer's default changes, this fails until the
    evaluator follows -- which is exactly what was missing for four weeks."""
    trainer_default = {f.name: f.default for f in dataclasses.fields(CurriculumConfig)}[
        "waypoint_reach_mode"
    ]
    eval_default = inspect.signature(rollout_episode).parameters["reach_mode"].default
    assert eval_default == trainer_default


_SWEEP = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "keep_best_sweep.py"
)


def _captured_eval_cmd(monkeypatch, tmp_path, **kw):
    spec = importlib.util.spec_from_file_location("_yeti_keepbest_reach", _SWEEP)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"keep_best_sweep not importable here: {exc}")
    seen = {}

    def fake_run(cmd, **_):
        seen["cmd"] = list(cmd)
        out = cmd[cmd.index("--out") + 1]
        pathlib.Path(out).write_text(
            '{"episodes": 1, "princess_touches": 0, "max_cp_counts": {}, "rows": []}'
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._eval_snapshot(
        "m.zip",
        1,
        "cpu",
        str(tmp_path / "e.json"),
        profile="p",
        fruits_total=1,
        start_state=None,
        stall_threshold=40,
        max_steps=10,
        level=4,
        **kw,
    )
    return seen["cmd"]


def test_referee_forwards_box(monkeypatch, tmp_path):
    cmd = _captured_eval_cmd(monkeypatch, tmp_path, reach_mode="box")
    assert cmd[cmd.index("--reach-mode") + 1] == "box"


def test_referee_omits_the_default(monkeypatch, tmp_path):
    assert "--reach-mode" not in _captured_eval_cmd(monkeypatch, tmp_path)
