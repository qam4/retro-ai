"""An episode is credited to the start pool it was DRAWN from.

pick_start weights each pool by 1 - its goal score, so a pool's score must move with
the episodes drawn from it. It did not when the env re-derived the key at episode end
from the state's progress rung: a pool file is filed by each state's recorded route
points, which never include fruits, while the rung at episode start also counts the
fruits missing in the game. l3_v15's pool file holds 100 states on L3's fruit platform
with the fruit taken: filed under rung 8, started at rung 9. Every episode drawn from
rung 8 was credited to rung 9, rung 8's score stayed 0.00, and in l3_v16 (warm from
that file) it took 73-78% of all starts for the first 1.2M steps -- 10,457 of 13,694
episodes, median 21 steps -- before the from-reset chain collapsed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path

import pytest
from retro_ai.training.run_config import RunConfig

REPO = Path(__file__).resolve().parents[2]
_SRC = REPO / "scripts/mo5/yeti/train_checkpoint_curriculum.py"
_L3_CFG = (
    REPO
    / "experiments/003-yeti/configs/yeti_curriculum_l3_v16_l4rules_warm_v15_6m.yaml"
)
_V15_POOLS = (
    REPO / "output/mo5/yeti/training/yeti_curriculum_l3_v15_gatewp_15m/checkpoints.pkl"
)


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location("tcc_credit", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - ML deps absent (CI)
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _mgr(tcc, cfg):
    ladder, rungs = tcc._progress_ladder(cfg)
    return tcc._make_manager(cfg, ladder, rungs)


def test_a_rung_pool_is_judged_from_the_rung_its_episodes_start_on(tcc):
    """Drawn from rung 8, started on rung 9, ended on rung 9: the episode is rung 8's
    to score, and it advanced nothing -- it gained no step over where it began."""
    cfg = RunConfig.from_yaml(str(_L3_CFG))
    m = _mgr(tcc, cfg)
    for _ in range(200):
        m.record_episode(8, 9, start_rung=9)
    assert m.checkpoints[8].goal_score > 0.5, "the drawn pool's score must move"
    assert m.checkpoints[9].goal_score == 0.0, "nothing was drawn from rung 9"
    assert m.segment_attempts[8] == 200 and m.segment_successes[8] == 0
    assert m.seg_success_ema[8] == 0.0
    # Without a start rung the key is the start rung, as before.
    m.record_episode(4, 5)
    assert m.segment_successes[4] == 1


class _Rows:
    def __init__(self):
        self.rows = []

    def log(self, **row):
        self.rows.append(row)


def test_the_env_credits_the_pool_it_drew_from_l3_v15s_file(tcc):
    """The real case: a state from l3_v15's file, filed under rung 8 with the fruit
    already taken in the game. Needs the ROMs, the L3 start and the local pool file."""
    if not os.environ.get("RETRO_AI_ROM_DIR"):
        pytest.skip("RETRO_AI_ROM_DIR not set")
    cfg = RunConfig.from_yaml(str(_L3_CFG))
    if not _V15_POOLS.is_file() or not (REPO / cfg.curriculum.start_state).is_file():
        pytest.skip("needs l3_v15's local pool file and the L3 level start")
    from retro_ai.games import yeti
    from retro_ai.training.rewards import create as create_reward

    # A short episode is enough: the credit is decided when it ends.
    cfg = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, max_steps=15))
    mgr = _mgr(tcc, cfg)
    mgr.load_from_disk(str(_V15_POOLS))
    pool = mgr.checkpoints[8]
    assert len(pool) > 0, "l3_v15's file no longer fills rung 8"
    saved = tcc._manager, list(tcc._groups)
    tcc._manager, tcc._groups[:] = mgr, []
    try:
        params = dict(cfg.reward.params)
        params.setdefault("gamma", cfg.ppo.gamma)
        params["waypoint_reach_mode"] = cfg.curriculum.waypoint_reach_mode
        rows = _Rows()
        env = tcc.CheckpointCurriculumEnv(
            cfg=cfg,
            reward_fn=create_reward(cfg.reward.name, params),
            env_id=0,
            episode_logger=rows,
        )
        env.reset(seed=0)  # boots the emulator from the level start
        # A rung-8 state whose fruit is already gone in the game.
        pick = None
        for entry in pool.states:
            env.iface.load_state(entry[2])
            if env.iface.read_ram_byte(yeti.FRUITS_ADDR) == 0:
                pick = entry
                break
        assert pick is not None, "no rung-8 state with the fruit taken"
        _src, _bonus, state, stack, reached = pick

        def _draw():
            mgr.stats["starts"][8] += 1
            return 8, state, stack, reached

        mgr.pick_start = _draw
        before = {n: mgr.checkpoints[n].goal_score for n in (8, 9)}
        rows.rows.clear()
        env.reset(seed=0)
        for _ in range(cfg.env.max_steps + 1):
            _obs, _r, done, trunc, _info = env.step([0, 0, 0])
            if done or trunc:
                break
        assert len(rows.rows) == 1
        row = rows.rows[0]
        assert row["start_level"] == 9, "the state should start one rung above its pool"
        assert row["start_key"] == 8, "episodes.csv must name the pool drawn from"
        assert mgr.checkpoints[8].goal_score > before[8], "rung 8 was never credited"
        assert mgr.checkpoints[9].goal_score == before[9], "rung 9 was credited instead"
    finally:
        tcc._manager, tcc._groups[:] = saved[0], saved[1]
