"""`mask_top_rows`: hide the HUD, change nothing else.

Why it exists. Policies read the HUD. Pasting another start's score digits or lives
stars into the observation, and nothing else, took l2_v10, l3_v15 and l4_v32 from their
own start's success rate to 0 of 20 episodes. Yeti's first HUD line (score, lives) is
raw rows 0-6; the mask zeroes those rows of the raw frame before anything else runs.

What must hold:

* masked rows are zero in the observation, every other output pixel is unchanged
* the default (0) changes nothing, including the frame-stack signature stored pools
  carry
* a stack captured unmasked is not restored into a masked pipeline (it shows the HUD)
* the caller's frame is not written to
* the value reaches the pipeline from the run config, in training and in evaluation

No emulator needed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.training import env_builder
from retro_ai.training.game_profile import GameProfile
from retro_ai.training.run_config import EnvConfig

H, W, OUT = 200, 320, 84


def _frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 256, (H, W, 3), dtype=np.uint8)


def _row_blocks():
    """Source rows [lo, hi) behind each output row: the resize's own block rule."""
    return [(i * H // OUT, (i + 1) * H // OUT) for i in range(OUT)]


@pytest.mark.parametrize("mode", ["nearest", "max"])
@pytest.mark.parametrize("n", [1, 7, 16])
def test_mask_zeroes_its_rows_and_nothing_else(mode, n):
    f = _frame()
    plain = PreprocessingPipeline(grayscale=True, resize=(OUT, OUT), resize_mode=mode)
    masked = PreprocessingPipeline(
        grayscale=True, resize=(OUT, OUT), resize_mode=mode, mask_top_rows=n
    )
    a, b = plain.process(f), masked.process(f)
    assert a.shape == b.shape == (OUT, OUT, 1)
    for i, (lo, hi) in enumerate(_row_blocks()):
        if hi <= n:  # block entirely masked
            assert (b[i] == 0).all(), f"output row {i} (source {lo}..{hi - 1})"
        elif lo >= n:  # block entirely visible
            assert np.array_equal(a[i], b[i]), f"output row {i} changed"


def test_seven_rows_is_exactly_the_first_three_output_rows():
    """Rows 0-6 are output rows 0-2 under 200 -> 84, with no partial block: output row
    3 starts at source row 7. So masking 7 rows blanks 3 whole output rows and leaves
    every playfield pixel under both resize modes."""
    assert [b for b in _row_blocks()[:4]] == [(0, 2), (2, 4), (4, 7), (7, 9)]


# How far a jump lifts the sprite top above its standing y. Measured by holding JUMP
# (up, left, right; 1-12 steps) from the L2 start (y 30) and the L4 start (y 182): the
# smallest y was 20 and 172, a 10 px rise in every case.
JUMP_RISE = 10
HUD_LINE_ROWS = 7  # Yeti's first HUD line (score, lives stars) is raw rows 0-6


def test_the_hud_mask_never_covers_the_agent():
    """The highest the sprite top ever gets is the highest standing y minus a jump. It
    must stay below the masked rows on every level, or the mask hides the agent."""
    from retro_ai.training.yeti_map import LEVELS

    for n, lvl in LEVELS.items():
        top = min(lvl.floor_top_y.values()) - JUMP_RISE
        assert top >= HUD_LINE_ROWS, f"level {n}: sprite top reaches row {top}"


def test_the_callers_frame_is_not_modified():
    f = _frame()
    before = f.copy()
    PreprocessingPipeline(grayscale=True, resize=(OUT, OUT), mask_top_rows=7).process(f)
    assert np.array_equal(f, before)


def test_default_changes_nothing():
    f = _frame()
    a = PreprocessingPipeline(grayscale=True, resize=(OUT, OUT)).process(f)
    b = PreprocessingPipeline(grayscale=True, resize=(OUT, OUT), mask_top_rows=0)
    assert np.array_equal(a, b.process(f))
    assert EnvConfig(profile="x").mask_top_rows == 0


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_bad_values_are_rejected(bad):
    with pytest.raises(ValueError):
        PreprocessingPipeline(resize=(OUT, OUT), mask_top_rows=bad)


class _FakeEnv:
    def reset(self, seed=None):
        return _frame(), {}

    def step(self, action):
        return _frame(1), 0.0, False, False, {}


def _sig(**kw):
    p = PreprocessingPipeline(grayscale=True, resize=(OUT, OUT), frame_stack=4, **kw)
    env = PreprocessedEnv(_FakeEnv(), p)
    env.reset()
    return env, env._stack_signature()


def test_default_stack_signature_is_unchanged():
    assert _sig()[1] == (True, (OUT, OUT), 4, None, (OUT, OUT, 1))


def test_an_unmasked_stack_is_not_restored_into_a_masked_pipeline():
    plain, _ = _sig()
    masked, _ = _sig(mask_top_rows=7)
    assert not masked.restore_frame_stack(plain.export_frame_stack())
    assert masked.restore_frame_stack(masked.export_frame_stack())


def test_masked_stack_frames_are_masked():
    env, _ = _sig(mask_top_rows=7)
    obs, *_ = env.step(0)
    assert (obs[:3] == 0).all() and obs[3:].any()


class _StubBaseEnv(_FakeEnv):
    """Stands in for BaseEnv in build_training_env (no ROM, no native module)."""

    def __init__(self, *args, **kwargs):
        pass

    def get_observation_space(self):
        return {"width": W, "height": H, "channels": 3, "bits_per_channel": 8}

    def get_action_space(self):
        return {"type": "discrete", "shape": [4]}


def test_env_builder_applies_the_mask_from_the_config(monkeypatch):
    profile = GameProfile(
        name="stub",
        emulator_type="mo5",
        rom_path="/nonexistent.rom",
        reward_mode="stub",
        grayscale=True,
        frame_stack=1,
        frame_skip=1,
    )
    monkeypatch.setattr(env_builder, "BaseEnv", _StubBaseEnv)
    monkeypatch.setattr(
        env_builder.GameProfileRegistry, "load", lambda self, name: profile
    )
    stack = env_builder.build_training_env(
        "stub", EnvConfig(profile="stub", mask_top_rows=7)
    )
    assert stack.preprocessed.preprocessing.mask_top_rows == 7
    obs, _ = stack.preprocessed.reset()
    assert (obs[:3] == 0).all() and obs[3:].any()


def test_evaluation_takes_the_mask_from_the_run_config():
    """The referee (--config) and registry evals build the env from these settings,
    so a masked run is scored on the masked picture it trained on."""
    from retro_ai.training import yeti_registry as reg

    registry = reg.Registry.load()
    cfg = registry.config("l4_v32")
    masked = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, mask_top_rows=7))
    assert registry.config_eval_settings(masked, 4).env.mask_top_rows == 7
    assert registry.config_eval_settings(cfg, 4).env.mask_top_rows == 0


def test_run_config_yaml_accepts_the_field(tmp_path):
    from retro_ai.training import yeti_registry as reg
    from retro_ai.training.run_config import RunConfig

    registry = reg.Registry.load()
    text = (registry.repo / registry.policy("l4_v32").config).read_text()
    assert "\nenv:\n" in text
    p = tmp_path / "c.yaml"
    p.write_text(text.replace("\nenv:\n", "\nenv:\n  mask_top_rows: 7\n", 1))
    assert RunConfig.from_yaml(str(p)).env.mask_top_rows == 7
