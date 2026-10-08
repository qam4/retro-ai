"""The referee's frontier on a level with no route order: the progress ladder.

The frontier is the deepest route point a policy still reaches. It walked the level
map's `route_order`, which L3/L4 define and L1/L2 do not, so on L1 every snapshot read
`frontier=-` and was ranked on princess and mean rung alone -- though L1's fruits ARE
its progress ladder. With no route order the frontier now walks the ladder: `rungK` =
the fraction of episodes with at least K rungs, which needs no assumption about the
order fruits are collected in.

What must not move: a level WITH a route order scores exactly as before.

No emulator needed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_SWEEP = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "keep_best_sweep.py"
)


def _mod():
    spec = importlib.util.spec_from_file_location("_yeti_keepbest_frontier", _SWEEP)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"keep_best_sweep not importable here: {exc}")
    return mod


def _rates(monkeypatch, tmp_path, eval_json):
    mod = _mod()

    def fake_run(cmd, **_):
        out = cmd[cmd.index("--out") + 1]
        pathlib.Path(out).write_text(json.dumps(eval_json))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    *_, rates = mod._eval_snapshot(
        "m.zip",
        1,
        "cpu",
        str(tmp_path / "e.json"),
        profile="yeti_fruit",
        fruits_total=4,
        start_state=None,
        stall_threshold=15,
        max_steps=1000,
        level=1,
    )
    return mod, rates


# 10 L1 episodes: 2 got no fruit, 3 got one, 5 got two. JSON keys are strings, as
# eval_from_reset writes them.
L1_EVAL = {
    "episodes": 10,
    "princess_touches": 0,
    "max_cp_counts": {},
    "rows": [],
    "n_rungs": 4,
    "rung_counts": {"0": 2, "1": 3, "2": 5},
}


def test_ladder_rates_are_at_least_k(monkeypatch, tmp_path):
    _, r = _rates(monkeypatch, tmp_path, L1_EVAL)
    assert (r["rung1"], r["rung2"], r["rung3"], r["rung4"]) == (0.8, 0.5, 0.0, 0.0)


def test_l1_gets_a_frontier_from_its_fruits(monkeypatch, tmp_path):
    mod, r = _rates(monkeypatch, tmp_path, L1_EVAL)
    route = mod._frontier_route(None, 4)
    assert route == ["rung1", "rung2", "rung3", "rung4"]
    assert mod._frontier(r, route) == ("rung2", 0.5)


def test_a_level_with_a_route_order_is_untouched(monkeypatch, tmp_path):
    """The rung keys exist for every level, but a route order takes precedence, so the
    frontier, and therefore the score, are exactly what they were."""
    mod, r = _rates(
        monkeypatch,
        tmp_path,
        {
            "episodes": 10,
            "princess_touches": 0,
            "max_cp_counts": {},
            "n_rungs": 12,
            "rung_counts": {"9": 10},
            "rows": [{"reached_points": ["Lfruit_top", "Rope1", "Low1"]}] * 7
            + [{"reached_points": ["Lfruit_top"]}] * 3,
        },
    )
    order = ["Lfruit_top", "Rope1", "Low1", "Low2"]
    assert mod._frontier_route(order, 12) == order
    assert mod._frontier(r, mod._frontier_route(order, 12)) == ("Low1", 0.7)
    assert mod._frontier(r, order) == ("Low1", 0.7)  # the pre-change call


def test_no_ladder_and_no_route_means_no_frontier():
    mod = _mod()
    assert mod._frontier_route(None, 0) == []
    assert mod._frontier({}, []) == (None, 0.0)
