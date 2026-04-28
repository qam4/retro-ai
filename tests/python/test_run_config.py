"""Tests for RunConfig YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from retro_ai.training.run_config import (
    BackwardCurriculumConfig,
    CurriculumConfig,
    EnvConfig,
    PPOConfig,
    RewardConfig,
    RunConfig,
    SegmentConfig,
    TrainingConfig,
)


SEGMENT_YAML = {
    "training": {
        "timesteps": 5_000_000,
        "num_envs": 8,
        "seed": 42,
        "output": "output/mo5/yeti/training/segment_1to2_v3",
    },
    "env": {
        "profile": "yeti_fruit",
        "action_mode": "joystick",
        "max_steps": 1000,
        "stall_threshold": 15,
    },
    "ppo": {
        "learning_rate": 3.0e-4,
        "batch_size": 64,
        "n_epochs": 4,
        "ent_coef": 0.01,
    },
    "reward": {
        "name": "fruit_bonus",
        "params": {"scale": 0.01},
    },
    "segment": {
        "checkpoints": "output/mo5/yeti/training/curriculum_v5/checkpoints.pkl",
        "segment": 1,
    },
}


def _write(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "run.yaml"
    with p.open("w") as f:
        yaml.safe_dump(data, f)
    return str(p)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_segment_config_roundtrip(tmp_path):
    path = _write(tmp_path, SEGMENT_YAML)
    cfg = RunConfig.from_yaml(path)
    assert cfg.training.timesteps == 5_000_000
    assert cfg.training.num_envs == 8
    assert cfg.training.seed == 42
    assert cfg.env.profile == "yeti_fruit"
    assert cfg.env.max_steps == 1000
    assert cfg.ppo.learning_rate == 3.0e-4
    assert cfg.reward.name == "fruit_bonus"
    assert cfg.reward.params == {"scale": 0.01}
    assert isinstance(cfg.segment, SegmentConfig)
    assert cfg.segment.segment == 1
    assert cfg.curriculum is None
    assert cfg.backward_curriculum is None


def test_to_dict_omits_none_script_sections(tmp_path):
    path = _write(tmp_path, SEGMENT_YAML)
    cfg = RunConfig.from_yaml(path)
    d = cfg.to_dict()
    assert "segment" in d
    assert "curriculum" not in d
    assert "backward_curriculum" not in d


def test_minimal_config_uses_defaults(tmp_path):
    minimal = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_flat"},
    }
    path = _write(tmp_path, minimal)
    cfg = RunConfig.from_yaml(path)
    # Defaults populated from dataclass
    assert cfg.training.num_envs == 8
    assert cfg.training.seed is None
    assert cfg.env.action_mode == "joystick"
    assert cfg.env.max_steps == 1000
    assert cfg.ppo.learning_rate == 3e-4
    assert cfg.reward.params == {}


def test_curriculum_config(tmp_path):
    data = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_bonus"},
        "curriculum": {
            "reset_fraction": 0.5,
            "frontier_fraction": 0.3,
            "earlier_fraction": 0.2,
        },
    }
    cfg = RunConfig.from_yaml(_write(tmp_path, data))
    assert isinstance(cfg.curriculum, CurriculumConfig)
    assert cfg.curriculum.reset_fraction == 0.5
    assert cfg.segment is None


def test_backward_curriculum_config(tmp_path):
    data = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti"},
        "reward": {"name": "score_delta_survival"},
        "backward_curriculum": {
            "archive": "output/mo5/yeti/go_explore_v8/archive.pkl",
        },
    }
    cfg = RunConfig.from_yaml(_write(tmp_path, data))
    assert isinstance(cfg.backward_curriculum, BackwardCurriculumConfig)
    assert (
        cfg.backward_curriculum.archive == "output/mo5/yeti/go_explore_v8/archive.pkl"
    )
    assert cfg.backward_curriculum.advance_threshold == 20.0  # default


def test_resize_coerced_to_tuple(tmp_path):
    data = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti_fruit", "resize": [84, 84]},
        "reward": {"name": "fruit_flat"},
    }
    cfg = RunConfig.from_yaml(_write(tmp_path, data))
    assert cfg.env.resize == (84, 84)
    assert isinstance(cfg.env.resize, tuple)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path):
    data = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_flat"},
        "typoed_section": {"foo": 1},
    }
    with pytest.raises(ValueError, match="Unknown top-level"):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_unknown_nested_key_rejected(tmp_path):
    data = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti_fruit", "not_a_field": True},
        "reward": {"name": "fruit_flat"},
    }
    with pytest.raises(ValueError, match="Unknown keys in 'env'"):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_missing_required_section_raises(tmp_path):
    data = {
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_flat"},
    }
    with pytest.raises(ValueError, match="training"):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_missing_required_field_raises(tmp_path):
    # training.timesteps is required (no default)
    data = {
        "training": {"output": "/tmp/x"},
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_flat"},
    }
    with pytest.raises(TypeError):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_multiple_script_sections_rejected(tmp_path):
    data = {
        "training": {"timesteps": 1000, "output": "/tmp/x"},
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_flat"},
        "segment": {
            "checkpoints": "/tmp/cps.pkl",
            "segment": 1,
        },
        "curriculum": {},
    }
    with pytest.raises(ValueError, match="multiple script-specific"):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_non_mapping_section_raises(tmp_path):
    data = {
        "training": "not a mapping",
        "env": {"profile": "yeti_fruit"},
        "reward": {"name": "fruit_flat"},
    }
    with pytest.raises(ValueError, match="must be a mapping"):
        RunConfig.from_yaml(_write(tmp_path, data))


def test_non_mapping_yaml_file_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        RunConfig.from_yaml(str(p))


# ---------------------------------------------------------------------------
# Dataclass sanity (cheap guardrails)
# ---------------------------------------------------------------------------


def test_training_config_is_frozen():
    cfg = TrainingConfig(timesteps=1000, output="/tmp/x")
    with pytest.raises(Exception):
        cfg.timesteps = 2000  # type: ignore[misc]


def test_reward_config_params_defaults_empty():
    r = RewardConfig(name="x")
    assert r.params == {}


def test_ppo_config_defaults_sane():
    p = PPOConfig()
    assert p.learning_rate > 0
    assert p.batch_size > 0
    assert p.n_steps is None
    assert 0 < p.ent_coef < 1
