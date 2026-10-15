"""Every Yeti level detects death the same way and stops under the same rules.

Level 1 used to be the exception. Its profile stopped on a lives decrease and a 10-frame
bonus stall, its trainer path skipped the 0x2AFC death flag "to keep L1's termination
timing", and the registry gave it 1000 steps / stall 15 where L4 has 1500 / 40. The
lives test cannot fire -- the lives byte does not move at the death frame on any level
(games/yeti.py, LIVES_ADDR) -- so on L1 the reward never saw a death and every death was
logged "env_done": 14970 of v19's first 15326 episodes, 0 "death".

These pin the one rule set so a new level, or a new script, cannot quietly go back.
The profile and source checks need no emulator; the last test drives the trainer's env.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "game_profiles"
FRUIT_PROFILES = {
    1: "mo5_yeti_fruit.yaml",
    2: "mo5_yeti_fruit_level2.yaml",
    3: "mo5_yeti_fruit_level3.yaml",
    4: "mo5_yeti_fruit_level4.yaml",
}


def _profile(name):
    return yaml.safe_load((PROFILES / name).read_text())


def _yeti_profiles_with_rewards():
    for p in sorted(PROFILES.glob("mo5_yeti*.yaml")):
        d = yaml.safe_load(p.read_text())
        if (d or {}).get("reward_params"):
            yield p.name, d


@pytest.mark.parametrize("name, d", list(_yeti_profiles_with_rewards()))
def test_no_yeti_profile_detects_death_from_lives(name, d):
    rp = d["reward_params"]
    assert "lives_addr" not in rp, f"{name} still stops on a lives decrease"
    assert (rp.get("death_flag_addr"), rp.get("death_flag_value")) == ("11004", "65")


def test_every_level_runs_the_level_4_profile():
    """Same game-global reward params everywhere: only the name may differ."""
    ref = _profile(FRUIT_PROFILES[4])
    for level, name in FRUIT_PROFILES.items():
        d = _profile(name)
        diff = sorted(
            k
            for k in set(d) | set(ref)
            if k not in ("name", "display_name") and d.get(k) != ref.get(k)
        )
        assert not diff, f"L{level} profile differs from L4's in {diff}"


def test_level_1_and_5_stop_like_level_4():
    from retro_ai.training.yeti_registry import Registry

    reg = Registry.load()
    l4 = reg.level(4)
    for n in (1, 5):
        lvl = reg.level(n)
        assert (lvl.max_steps, lvl.stall_threshold) == (
            l4.max_steps,
            l4.stall_threshold,
        )


# A death decided from the lives byte, in any of the shapes it has been written in.
_LIVES_DEATH = re.compile(
    r"lives\w*\s*<\s*\S*prev_lives|prev_lives\w*\s*>\s*\S*lives\b|"
    r"\"lives\"\]\s*<\s*prev_lives"
)


def test_no_python_decides_death_from_lives():
    hits = [
        f"{p.relative_to(REPO)}:{i}"
        for root in ("python", "scripts")
        for p in sorted((REPO / root).rglob("*.py"))
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
        # Prose may quote the old test as ``code``; only code counts.
        if _LIVES_DEATH.search(re.sub(r"``[^`]*``", "", line))
        and not line.lstrip().startswith("#")
    ]
    assert not hits, f"lives-based death detection is back: {hits}"


# --- the trainer itself, on level 1 ------------------------------------------------

_SRC = REPO / "scripts/mo5/yeti/train_checkpoint_curriculum.py"
_L1_CFG = (
    REPO
    / "experiments/003-yeti/configs/yeti_curriculum_l1_v20_l4settings_warm_v17_6m.yaml"
)


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location("tcc_death", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - ML deps absent (CI)
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


class _Rows:
    def __init__(self):
        self.rows = []

    def log(self, **row):
        self.rows.append(row)


def test_a_level_1_death_ends_the_episode_as_a_death(tcc):
    """From the L1 reset, holding NOOP, the agent is hit by a snowball. The episode must
    end on the death flag and be logged "death", as on every other level."""
    import os

    if not os.environ.get("RETRO_AI_ROM_DIR"):
        pytest.skip("RETRO_AI_ROM_DIR not set")
    from retro_ai.games import yeti
    from retro_ai.training.rewards import create as create_reward
    from retro_ai.training.run_config import RunConfig
    from retro_ai.training.yeti_registry import Registry

    cfg = RunConfig.from_yaml(str(_L1_CFG))
    l1 = Registry.load().level(1)
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env, max_steps=l1.max_steps, stall_threshold=l1.stall_threshold
        ),
    )
    ladder, rungs = tcc._progress_ladder(cfg)
    mgr = tcc._make_manager(cfg, ladder, rungs)
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
        env.reset(seed=0)
        for _ in range(cfg.env.max_steps + 1):
            _obs, _r, done, trunc, _info = env.step([0, 0, 0])
            if done or trunc:
                break
        assert yeti.is_dead(env.iface), "holding NOOP from the L1 reset no longer dies"
        assert [r["end_reason"] for r in rows.rows] == ["death"]
    finally:
        tcc._manager, tcc._groups[:] = saved[0], saved[1]
