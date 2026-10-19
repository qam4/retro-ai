"""The referee scores every level a run trains on, with settings from the registry.

`keep_best_sweep.py --config` replaces the level flags the runner used to build by hand,
and with start groups it refuses to crown a snapshot that falls below a level's
registered champion by more than the sample sizes explain.
"""

from __future__ import annotations

import dataclasses
import importlib.util
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


def test_a_plain_run_is_scored_on_its_own_level(ref):
    _reg, _cfg, plan = ref._registry_plan(
        str(_CFG / "yeti_curriculum_l4_v30_blockmax_cold_15m.yaml")
    )
    assert [(n, pool) for n, _s, _r, pool in plan] == [(4, None)]
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
    assert [n for n, _s, _r, _pool in plan] == [1, 5]
    assert plan[1][1].start_state == "output/mo5/yeti/level5/level5_start.sav"
    assert registry.champions.get(1) == "l1_v22" and registry.champions.get(5) is None


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
