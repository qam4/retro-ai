"""Root pools: respawn states as extra level starts, drawn by snowball sequence.

Uses a fake manifest of tiny files, so no emulator or harvested save is needed.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest
import yaml
from retro_ai.training import root_pool
from retro_ai.training.run_config import RunConfig
from retro_ai.training.yeti_registry import Registry

REPO = Path(__file__).resolve().parents[2]
_CFG = (
    REPO
    / "experiments/003-yeti/configs/yeti_curriculum_l1_v20_l4settings_warm_v17_6m.yaml"
)


def _manifest(tmp_path, sequences, heldout=(), game_level=1, layout=1):
    """sequences: {seq_id: n_states}."""
    states = []
    for seq, n in sequences.items():
        for i in range(n):
            p = tmp_path / f"s{seq}_{i}.sav"
            p.write_bytes(f"state {seq} {i}".encode())
            states.append(
                dict(
                    path=str(p),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    sequence=seq,
                    heldout=seq in heldout,
                )
            )
    m = tmp_path / "pool.json"
    m.write_text(json.dumps(dict(game_level=game_level, layout=layout, states=states)))
    return str(m)


def test_splits_keep_heldout_sequences_out_of_training(tmp_path):
    m = _manifest(tmp_path, {0: 3, 1: 2, 2: 1}, heldout={1})
    assert sorted(root_pool.load(m, "train").sequences) == [0, 2]
    assert sorted(root_pool.load(m, "heldout").sequences) == [1]
    assert root_pool.load(m, "all").n_states == 6


def test_a_changed_state_file_is_refused(tmp_path):
    m = _manifest(tmp_path, {0: 1})
    (tmp_path / "s0_0.sav").write_bytes(b"something else")
    with pytest.raises(ValueError, match="sha256"):
        root_pool.load(m, "train")


def test_draws_are_uniform_over_sequences_not_states(tmp_path):
    """A sequence harvested 50 times must weigh the same as one harvested once."""
    pool = root_pool.load(_manifest(tmp_path, {0: 50, 1: 1}), "train")
    rng = random.Random(4)
    c = collections.Counter(pool.draw(rng)[0].split("/")[-1][1] for _ in range(4000))
    assert abs(c["0"] / 4000 - 0.5) < 0.03


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location(
        "tcc_rootpool", REPO / "scripts/mo5/yeti/train_checkpoint_curriculum.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - ML deps absent (CI)
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _cfg(**cur):
    c = RunConfig.from_yaml(str(_CFG))
    return dataclasses.replace(c, curriculum=dataclasses.replace(c.curriculum, **cur))


def _groups(tcc, cfg):
    specs = tcc._start_group_specs(cfg)
    ladder, rungs = tcc._progress_ladder(cfg)
    groups = [
        tcc.StartGroup(n, tcc._make_manager(cfg, ladder, rungs), None, "x")
        for n, _s in specs
    ]
    return groups, specs


def test_a_group_without_a_pool_draws_nothing_from_random(tcc):
    g, _ = _groups(tcc, _cfg())
    random.seed(9)
    assert g[0].pick_root() is None
    after = random.random()
    random.seed(9)
    assert after == random.random()


def test_the_pool_becomes_extra_level_start_sequences(tcc, tmp_path):
    m = _manifest(tmp_path, {0: 2, 1: 1, 2: 1}, heldout={2})
    cfg = _cfg(root_pool=m)
    g, specs = _groups(tcc, cfg)
    line = tcc._attach_root_pool(cfg, g, specs)
    assert "2 snowball sequences" in line
    # the plain level start (game reset, None) is one sequence; held-out 2 is absent
    assert [len(r) for r in g[0].roots] == [1, 2, 1]
    random.seed(1)
    drawn = collections.Counter(
        "reset" if (b := g[0].pick_root()) is None else b.decode()[:7]
        for _ in range(3000)
    )
    assert set(drawn) == {"reset", "state 0", "state 1"}
    assert all(abs(v / 3000 - 1 / 3) < 0.04 for v in drawn.values())


def test_a_pool_for_another_level_is_refused(tcc, tmp_path):
    m = _manifest(tmp_path, {0: 1}, game_level=5)
    cfg = _cfg(root_pool=m)
    g, specs = _groups(tcc, cfg)
    with pytest.raises(SystemExit, match="game level 5"):
        tcc._attach_root_pool(cfg, g, specs)


def test_pool_results_do_not_mix_with_level_start_results(tmp_path):
    loaded = Registry.load()
    sha = loaded.policy("l1_v17").model_sha256
    folder = tmp_path / "l1_v17"
    folder.mkdir()
    base = dict(
        policy="l1_v17",
        game_level=1,
        model_sha256=sha,
        native_sha256="N",
        deterministic=False,
        episodes=100,
        profile="yeti_fruit",
        max_steps=1500,
        stall_threshold=40,
    )
    (folder / "a.json").write_text(json.dumps(dict(base, seed=1, princess_touches=70)))
    (folder / "b.json").write_text(
        json.dumps(
            dict(
                base,
                seed=2,
                princess_touches=0,
                start_pool="p.json",
                pool_split="heldout",
            )
        )
    )
    r = Registry(loaded.levels, loaded.policies, loaded.champions, tmp_path, tmp_path)
    assert r.rate("l1_v17", 1, "N") == (70, 100)
    assert [x.princess for x in r.results("l1_v17", 1, "p.json", "heldout")] == [0]


L1_POOL = "experiments/003-yeti/data/respawns/l1_r1.json"


def _referee():
    spec = importlib.util.spec_from_file_location(
        "kbs_pool", REPO / "scripts/mo5/yeti/keep_best_sweep.py"
    )
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


def _run_cfg(tmp_path, root_pool_path):
    d = RunConfig.from_yaml(str(_CFG)).to_dict()
    d["curriculum"]["root_pool"] = root_pool_path
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


def test_the_referee_scores_the_levels_registered_pool(tmp_path):
    """A run training on L1's registered pool is scored on its held-out sequences; the
    game reset is kept beside them."""
    ref = _referee()
    _reg, _cfg2, plan = ref._registry_plan(_run_cfg(tmp_path, L1_POOL))
    assert [(n, pool) for n, _s, _r, pool in plan] == [(1, None), (1, L1_POOL)]
    assert ref._entry_key(1, L1_POOL) == "1-heldout"


def test_training_on_another_pool_than_the_levels_is_refused(tmp_path):
    """Two pools harvested apart do not share a held-out split: training on one can
    train on the other's held-out starts."""
    ref = _referee()
    m = _manifest(tmp_path, {0: 1, 1: 1}, heldout={1})
    with pytest.raises(SystemExit, match="start pool"):
        ref._registry_plan(_run_cfg(tmp_path, m))
