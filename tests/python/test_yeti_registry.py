"""The Yeti registry: one source for how each policy is evaluated on each level.

Most of these need only tracked files (the registry, configs, archived run records), so
they run in CI. The model-hash check and the evaluator's own flags need local files and
skip without them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
from retro_ai.training import yeti_registry as reg
from retro_ai.training.yeti_map import LEVELS

REPO = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO / "scripts/mo5/yeti/eval_from_reset.py"


@pytest.fixture(scope="module")
def registry():
    return reg.Registry.load()


def _as_tuple(v):
    return tuple(v) if isinstance(v, list) else v


def test_every_entry_points_at_something_real(registry):
    for n, lvl in registry.levels.items():
        assert lvl.layout in LEVELS, f"level {n}: layout {lvl.layout} has no map"
    for pid, pol in registry.policies.items():
        assert (REPO / pol.config).is_file(), f"{pid}: config {pol.config} missing"
        assert (REPO / pol.run_record).is_file(), f"{pid}: {pol.run_record} missing"
        assert len(pol.model_sha256) == 64, f"{pid}: model_sha256 is not a sha256"
    for n, pid in registry.champions.items():
        assert n in registry.levels and pid in registry.policies


@pytest.mark.parametrize("pid", sorted(reg.Registry.load().policies))
def test_each_policy_trained_on_a_level_whose_settings_match(registry, pid):
    """The registry's level is the one the policy's config trained on.

    If a config and the registry disagree on profile, start, fruit count or layout,
    every evaluation through the registry would score the policy on a level it never
    trained on, without an error.

    STOP RULES ARE NOT CHECKED HERE, deliberately. They belong to the level, not to the
    policy: L1 moved from 1000 / 15 to L4's 1500 / 40, and l1_v17 -- trained under the
    old ones -- is scored under the new ones like any later L1 policy. A result
    measured under other rules is a different measurement and is not pooled
    (test_results_under_other_stop_rules_are_not_pooled). A NEW run has to train under
    the current rules: the referee's --check refuses it otherwise (Registry.mismatches).
    """
    n = registry.training_level(pid)
    lvl = registry.level(n)
    cfg = registry.config(pid)
    assert cfg.env.profile == lvl.profile
    assert cfg.curriculum.fruits_total == lvl.fruits_total
    assert cfg.curriculum.start_state == lvl.start_state
    assert int(cfg.reward.params.get("level", 1)) == lvl.layout


@pytest.mark.parametrize("pid", sorted(reg.Registry.load().policies))
def test_config_still_matches_the_run_that_used_it(registry, pid):
    """The policy's view is read from its config, so the config must not have been
    edited since the run. Checked against the run record archived at launch, for every
    field that record has (older records predate some fields)."""
    cfg = registry.config(pid)
    extras = registry.run_record(pid)["extras"]
    for key, value in (extras.get("env") or {}).items():
        assert _as_tuple(getattr(cfg.env, key)) == _as_tuple(value), f"env.{key}"
    rec_cur = extras.get("curriculum") or {}
    if "waypoint_reach_mode" in rec_cur:
        assert cfg.curriculum.waypoint_reach_mode == rec_cur["waypoint_reach_mode"]


def test_l1_policy_on_l5_takes_l5s_start_and_its_own_view(registry):
    s = registry.eval_settings("l1_v17", 5)
    assert s.start_state == "output/mo5/yeti/level5/level5_start.sav"
    assert s.layout == 1 and s.fruits_total == 4
    assert s.env.profile == "yeti_fruit"
    assert (s.env.max_steps, s.env.stall_threshold) == (1500, 40)
    # the policy's view, from its config
    assert s.env.resize_mode == "max" and s.env.resize == (84, 84)
    assert s.reach_mode == "sprite"


def test_runs_from_before_the_reach_flag_report_box(registry):
    """L2 v10 (2026-07-29) and L3 v15 (2026-08-13) predate `waypoint_reach_mode`
    (c666500, 2026-09-08). Today's RunConfig default is sprite, which they never used.
    """
    for pid in ("l2_v10", "l3_v15"):
        assert "waypoint_reach_mode" not in (
            registry.run_record(pid)["extras"].get("curriculum") or {}
        )
        assert registry.eval_settings(pid, registry.training_level(pid)).reach_mode == (
            "box"
        )
    assert registry.eval_settings("l4_v30", 4).reach_mode == "sprite"


def test_level_settings_win_over_the_policys_training_level(registry):
    """l2_v10 trained on L2's stop rules; on L3 it must get L3's."""
    s = registry.eval_settings("l2_v10", 3)
    assert (s.env.profile, s.env.max_steps, s.env.stall_threshold) == (
        "yeti_fruit_level3",
        1500,
        40,
    )
    assert s.env.resize_mode == "nearest"  # still its own view


def test_unknown_names_fail_loudly(registry):
    with pytest.raises(KeyError, match="game level 99"):
        registry.eval_settings("l1_v17", 99)
    with pytest.raises(KeyError, match="nope"):
        registry.eval_settings("nope", 1)


def _toy_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--resize-mode", default="nearest")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    return p


def test_explicit_flags_are_found_in_every_spelling():
    argv = ["--resize", "max", "--max-steps=5", "--stochastic", "--episodes", "200"]
    # abbreviation, --opt=value, a flag, and a value equal to its default all count
    assert reg.explicit_dests(_toy_parser(), argv) == {
        "resize_mode",
        "max_steps",
        "stochastic",
        "episodes",
    }
    assert reg.hand_settings_given(_toy_parser(), argv) == [
        "--max-steps",
        "--resize-mode",
    ]
    assert reg.hand_settings_given(_toy_parser(), ["--episodes", "3"]) == []


def _write(folder: Path, name: str, **d):
    folder.mkdir(parents=True, exist_ok=True)
    base = {
        "policy": "l1_v17",
        "game_level": 1,
        "model_sha256": reg.Registry.load().policy("l1_v17").model_sha256,
        "native_sha256": "N1",
        "deterministic": False,
        "seed": 1,
        "episodes": 100,
        "princess_touches": 70,
        # The rules it was measured under, as eval_from_reset records them.
        "profile": "yeti_fruit",
        "max_steps": 1500,
        "stall_threshold": 40,
    }
    base.update(d)
    (folder / name).write_text(json.dumps(base), encoding="utf-8")


def test_results_come_from_files_and_are_checked(tmp_path):
    folder = tmp_path / "l1_v17"
    _write(folder, "a.json")
    _write(folder, "b.json", seed=2, princess_touches=60)
    _write(folder, "same_seed.json", seed=1, princess_touches=99)  # same episodes as a
    _write(folder, "other_core.json", seed=3, native_sha256="N2")
    _write(folder, "other_model.json", seed=4, model_sha256="0" * 64)
    _write(folder, "deterministic.json", seed=5, deterministic=True)
    _write(folder, "level5.json", seed=6, game_level=5, princess_touches=0)
    (folder / "legacy.json").write_text(
        json.dumps({"episodes": 300, "princess_touches": 208}), encoding="utf-8"
    )
    loaded = reg.Registry.load()
    r = reg.Registry(
        loaded.levels, loaded.policies, loaded.champions, tmp_path, results_dir=tmp_path
    )
    names = sorted(Path(x.file).name for x in r.results("l1_v17", 1))
    assert names == ["a.json", "b.json", "other_core.json", "same_seed.json"]
    assert r.rate("l1_v17", 1, "N1") == (130, 200)  # a + b; seed 1 counted once
    assert r.rate("l1_v17", 1, "N2") == (70, 100)
    assert r.rate("l1_v17", 5, "N1") == (0, 100)
    assert r.rate("l2_v10", 2, "N1") == (0, 0)


def test_results_under_other_stop_rules_are_not_pooled(tmp_path):
    """L1 moved to 1500 / 40. A result measured under 1000 / 15 -- every L1 result
    before the move -- is a different measurement, and so is one that does not record
    its rules at all."""
    folder = tmp_path / "l1_v17"
    _write(folder, "now.json")
    _write(folder, "old_rules.json", seed=2, max_steps=1000, stall_threshold=15)
    _write(folder, "old_profile.json", seed=3, profile="yeti_fruit_level2")
    _write(folder, "unrecorded.json", seed=4)
    data = json.loads((folder / "unrecorded.json").read_text())
    for k in ("profile", "max_steps", "stall_threshold"):
        del data[k]
    (folder / "unrecorded.json").write_text(json.dumps(data))
    loaded = reg.Registry.load()
    r = reg.Registry(
        loaded.levels, loaded.policies, loaded.champions, tmp_path, results_dir=tmp_path
    )
    assert [Path(x.file).name for x in r.results("l1_v17", 1)] == ["now.json"]
    # L5 shares L1's rules, so an L5 result is filtered by the same three fields.
    _write(folder, "l5_old.json", seed=5, game_level=5, max_steps=1000)
    assert r.rate("l1_v17", 5, "N1") == (0, 0)


def test_registered_models_have_the_registered_hash(registry):
    checked = 0
    for pid, pol in registry.policies.items():
        path = REPO / pol.model
        if not path.is_file():
            continue
        assert reg.sha256_of(path) == pol.model_sha256, pid
        checked += 1
    if not checked:
        pytest.skip("no registered model is present on this machine")


def _eval_module():
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("eval_from_reset", EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_evaluator_has_every_flag_the_registry_owns():
    """Otherwise a hand-typed flag the registry should own would slip past the check."""
    dests = {a.dest for a in _eval_module().build_parser()._actions}
    missing = set(reg.HAND_SETTINGS) - dests
    assert not missing, missing


def test_the_evaluator_refuses_hand_settings_with_policy(monkeypatch):
    mod = _eval_module()
    argv = ["--policy", "l1_v17", "--game-level", "1", "--resize-mode", "nearest"]
    monkeypatch.setattr(mod.sys, "argv", ["eval_from_reset.py", *argv])
    with pytest.raises(SystemExit, match="--resize-mode"):
        mod.main()
