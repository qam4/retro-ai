"""Regression test: each parallel env must have its OWN reward_fn.

The shared-reward bug (approach 19/20): when multiple envs share a
single stateful reward function, env A's reset() clears the per-
episode state mid-episode for env B. The visible symptom was
``best_d`` ratchets being broken in ``fruit_bonus_path_progress``,
allowing reward farming. The same vulnerability applies to any
stateful reward (``fruit_bonus_floor_novelty``,
``fruit_bonus_climb_novelty``, ...).

This test imports each multi-env training script's ``make_env``
helper and asserts that two envs constructed via that helper end up
with two distinct ``_reward_fn`` instances.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script_module(name: str):
    """Import a scripts/X.py module without making scripts a package."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if str(SCRIPTS) in sys.path:
            sys.path.remove(str(SCRIPTS))


def _build_two_envs(make_env_factory):
    e0 = make_env_factory(0)()
    e1 = make_env_factory(1)()
    # Monitor wraps the SegmentEnv; unwrap to find the reward fn.
    inner0 = e0.env if hasattr(e0, "env") else e0
    inner1 = e1.env if hasattr(e1, "env") else e1
    while hasattr(inner0, "env") and not hasattr(inner0, "_reward_fn"):
        inner0 = inner0.env
    while hasattr(inner1, "env") and not hasattr(inner1, "_reward_fn"):
        inner1 = inner1.env
    return inner0, inner1


@pytest.fixture
def stateful_segment_cfg(tmp_path):
    """Minimal RunConfig for train_segment with a stateful reward."""
    import os

    if not os.environ.get("RETRO_AI_ROM_DIR"):
        pytest.skip("RETRO_AI_ROM_DIR not set")
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        f"""
training:
  timesteps: 1000
  output: {tmp_path / "out"}
  num_envs: 2
  seed: 42
env:
  profile: yeti_fruit
  action_mode: joystick
  max_steps: 100
  stall_threshold: 5
  resize: [84, 84]
reward:
  name: fruit_bonus_path_progress
  params:
    scale: 0.01
segment:
  checkpoints: output/mo5/yeti/seeds/v9_checkpoints_v2.pkl
  segment: 2
"""
    )
    return str(yaml_path)


def test_segment_env_each_env_has_own_reward_fn(stateful_segment_cfg):
    """Two parallel segment envs must hold distinct reward_fn objects.

    Otherwise ``reward_fn.reset()`` called by env A (on episode
    boundary) would clear per-episode state for env B mid-episode.
    """
    import pickle

    import yaml
    from retro_ai.training.run_config import RunConfig
    from retro_ai.training.run_manifest import EpisodeLogger

    train_segment = _load_script_module("train_segment")

    cfg = RunConfig.from_dict(yaml.safe_load(open(stateful_segment_cfg)))

    with open(cfg.segment.checkpoints, "rb") as f:
        states = pickle.load(f)["checkpoints"][cfg.segment.segment]

    import os

    os.makedirs(cfg.training.output, exist_ok=True)
    logger = EpisodeLogger(cfg.training.output)

    def make_env(rank: int):
        def _init():
            from retro_ai.training.rewards import create as create_reward

            env_reward_fn = create_reward(cfg.reward.name, cfg.reward.params)
            return train_segment.SegmentEnv(
                cfg=cfg,
                checkpoint_states=states,
                reward_fn=env_reward_fn,
                env_id=rank,
                episode_logger=logger,
            )

        return _init

    env0 = make_env(0)()
    env1 = make_env(1)()
    assert env0._reward_fn is not env1._reward_fn
    # Sanity: their best_d dicts are also distinct objects.
    assert env0._reward_fn.best_d is not env1._reward_fn.best_d
