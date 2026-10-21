"""The referee scores every level a run trains on, with settings from the registry.

`keep_best_sweep.py --config` replaces the level flags the runner used to build by hand,
and with start groups it refuses to crown a snapshot that falls below a level's
registered champion by more than the sample sizes explain.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from retro_ai.training.run_config import RunConfig
from retro_ai.training.yeti_registry import regressed_against

REPO = Path(__file__).resolve().parents[2]
_SRC = REPO / "scripts/mo5/yeti/keep_best_sweep.py"
_CFG = REPO / "experiments/003-yeti/configs"


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("kbs_levels", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_cfg(tmp_path, base, **curriculum):
    d = RunConfig.from_yaml(str(_CFG / base)).to_dict()
    d["curriculum"].update(curriculum)
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return str(p)


def test_regression_threshold_comes_from_the_sample_sizes():
    """Against l1_v17's 215/300, 30 episodes flag 13 princesses and not 14."""
    assert regressed_against(13, 30, 215, 300)[0] is True
    assert regressed_against(14, 30, 215, 300)[0] is False
    # a bigger sample resolves a smaller drop
    assert regressed_against(180, 300, 215, 300)[0] is True
    # no reference, no verdict
    assert regressed_against(0, 30, 0, 0) == (False, 0.0)


def test_a_plain_run_is_scored_on_its_own_levels_held_out_starts(ref):
    _reg, _cfg, plan = ref._registry_plan(
        str(_CFG / "yeti_curriculum_l4_v30_blockmax_cold_15m.yaml")
    )
    assert [(n, pool) for n, _s, _r, pool in plan] == [(4, L4_POOL)]
    s = plan[0][1]
    assert s.start_state == "output/mo5/yeti/level4/level4_start.sav"
    assert (s.env.resize_mode, s.reach_mode) == ("max", "sprite")
    assert len(plan[0][2]) > 0  # L4 has a route order


def test_a_grouped_run_is_scored_on_every_group(ref, tmp_path):
    cfg = _write_cfg(
        tmp_path,
        "yeti_curriculum_l1_v20_l4settings_warm_v17_6m.yaml",
        start_groups=[1, 5],
    )
    registry, _cfg, plan = ref._registry_plan(cfg)
    # L1 has a start pool: its held-out starts decide, its game reset is informational.
    # L5 has none yet, so its own start decides.
    # Each level is scored on its pool's held-out starts; L1's game reset is kept
    # beside them for information, L5's saved start is not scored.
    assert [(n, pool) for n, _s, _r, pool in plan] == [
        (1, None),
        (1, L1_POOL),
        (5, L5_POOL),
    ]
    assert ref._informational(plan) == {"1"}
    assert plan[2][1].start_state == "output/mo5/yeti/level5/level5_start.sav"
    assert registry.champions.get(1) == "l1_v22" and registry.champions.get(5) is None


L1_POOL = "experiments/003-yeti/data/respawns/l1_r1.json"
L4_POOL = "experiments/003-yeti/data/start_pools/l4_p1.json"
L5_POOL = "experiments/003-yeti/data/start_pools/l5_p1.json"
# An L2 run under L2's current stop rules (l2_v10's own config trained under 1000 / 40,
# which the referee now refuses).
L2_CFG = "yeti_curriculum_l2_v11_l4rules_warm_v10_6m.yaml"


def test_a_level_without_a_pool_is_scored_on_its_own_start(ref, monkeypatch):
    import retro_ai.training.yeti_registry as reg

    real = reg.Registry.load()
    levels = dict(real.levels)
    levels[4] = dataclasses.replace(levels[4], start_pool=None)
    fake = reg.Registry(levels, real.policies, real.champions)
    monkeypatch.setattr(reg.Registry, "load", classmethod(lambda cls, *a, **k: fake))
    _r, _c, plan = ref._registry_plan(
        str(_CFG / "yeti_curriculum_l4_v30_blockmax_cold_15m.yaml")
    )
    assert [(n, pool) for n, _s, _r, pool in plan] == [(4, None)]


def _pooled_registry(monkeypatch, tmp_path, level, game_level_in_manifest=None):
    """The real registry, with ``level`` given a tiny start pool of its own."""
    import retro_ai.training.yeti_registry as reg

    m = tmp_path / f"l{level}_pool.json"
    m.write_text(
        json.dumps(
            dict(game_level=game_level_in_manifest or level, layout=level, states=[])
        )
    )
    real = reg.Registry.load()
    levels = dict(real.levels)
    levels[level] = dataclasses.replace(levels[level], start_pool=str(m))
    fake = reg.Registry(levels, real.policies, real.champions)
    monkeypatch.setattr(reg.Registry, "load", classmethod(lambda cls, *a, **k: fake))
    return str(m)


def test_a_saved_level_start_is_not_scored_beside_its_pool(ref, monkeypatch, tmp_path):
    """L2's saved start is one draw among many (its hazards depend on when L1 ended),
    so with a start pool only the held-out starts are scored. Only a GAME RESET, which
    every real game plays first, is kept beside them."""
    m = _pooled_registry(monkeypatch, tmp_path, 2)
    _reg, _cfg, plan = ref._registry_plan(str(_CFG / L2_CFG))
    assert [(n, pool) for n, _s, _r, pool in plan] == [(2, m)]
    assert ref._informational(plan) == set()


def test_a_pool_for_another_level_is_refused(ref, monkeypatch, tmp_path):
    _pooled_registry(monkeypatch, tmp_path, 2, game_level_in_manifest=3)
    with pytest.raises(SystemExit, match="not for game level 2"):
        ref._registry_plan(str(_CFG / L2_CFG))


class _FakeRegistry:
    """Champion l1_x on L1: 215/300 from the reset, 0/300 on the held-out starts."""

    champions = {1: "l1_x"}

    def rate(self, pid, n, native, start_pool=None, pool_split=None):
        return (0, 300) if start_pool else (215, 300)


def _result(k, n=30):
    # (princess, reach_top, mean_rung, n_rungs, rates, touches, episodes, native)
    return (k / n, 0.0, 0.0, 4, {}, k, n, "N")


def test_the_reset_counts_toward_nothing_when_l1_has_a_pool(ref):
    """The snapshot is judged on the held-out starts only: a collapse on the memorised
    reset sequence (0/30 against the champion's 215/300) neither blocks it nor lowers
    its score, and the champion bar it is held to is the champion's HELD-OUT rate."""
    plan = [(1, None, [], None), (1, None, [], L1_POOL)]
    results = [(1, [], None, _result(0)), (1, [], L1_POOL, _result(3))]
    out, regressed, score = ref._score_levels(results, plan, _FakeRegistry(), {})
    assert regressed == []
    assert out["1"]["informational"] and "champion" not in out["1"]
    assert (out["1-heldout"]["champion_k"], out["1-heldout"]["champion_n"]) == (0, 300)
    assert score == out["1-heldout"]["score"]


def test_a_level_without_a_pool_is_still_held_to_its_champion(ref):
    plan = [(1, None, [], None), (5, None, [], None)]
    results = [(1, [], None, _result(5)), (5, [], None, _result(0))]
    out, regressed, score = ref._score_levels(results, plan, _FakeRegistry(), {})
    assert regressed == [1] and out["1"]["champion_regressed"]
    assert score == (out["1"]["score"] + out["5"]["score"]) / 2


def test_a_run_trained_off_its_levels_settings_is_refused(ref, tmp_path):
    cfg = RunConfig.from_yaml(
        str(_CFG / "yeti_curriculum_l1_v20_l4settings_warm_v17_6m.yaml")
    )
    d = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, stall_threshold=99))
    p = tmp_path / "off.yaml"
    p.write_text(yaml.safe_dump(d.to_dict()), encoding="utf-8")
    with pytest.raises(SystemExit, match="stall_threshold 40"):
        ref._registry_plan(str(p))


def test_hand_typed_level_flags_are_refused_with_config(ref, monkeypatch):
    argv = [
        "--snapshots-dir",
        "/nonexistent",
        "--config",
        str(_CFG / "yeti_curriculum_l1_v20_l4settings_warm_v17_6m.yaml"),
        "--level",
        "1",
        "--check",
    ]
    monkeypatch.setattr(ref.sys, "argv", ["keep_best_sweep.py", *argv])
    with pytest.raises(SystemExit, match="--level"):
        ref.main()
