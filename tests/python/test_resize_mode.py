"""`resize_mode`: a thin feature must survive the resize, and nothing else may change.

Why it exists. L4's rope 2 is a one-pixel-wide red line. Shrinking 320x200 to 84x84 by
nearest-neighbour keeps one source pixel per output pixel, so the rope keeps 0-4 pixels
and in some frames none at all -- the swing the crossing has to be timed on can vanish
from the observation. "max" keeps the brightest pixel of the same block instead.

What must NOT change is everything already trained. The L4 v29 champion scores mean
rung 8.17 under "nearest" and 0.02 under "max", so:

* the default must stay byte-identical to the old nearest-neighbour code
* the frame-stack signature of a default pipeline must be unchanged, or every stored
  pool's stack is silently re-seeded on load
* the referee must forward the mode, or a "max" run is scored on the wrong picture and
  reads as a total collapse with no error anywhere

No emulator needed.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.training.run_config import EnvConfig

_SWEEP = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "keep_best_sweep.py"
)


def _frame_with_thin_line(col: int, h: int = 200, w: int = 320) -> np.ndarray:
    """A black RGB frame with a one-pixel-wide pure-red vertical line at ``col``."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[20:110, col, 0] = 255
    return f


def _old_nearest(frame: np.ndarray, th: int, tw: int) -> np.ndarray:
    """The pre-change resize, copied verbatim, as the byte-identity reference."""
    gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
    frame = np.expand_dims(gray.astype(np.uint8), axis=-1)
    sh, sw = frame.shape[0], frame.shape[1]
    r = (np.arange(th) * sh // th).astype(int)
    c = (np.arange(tw) * sw // tw).astype(int)
    return frame[np.ix_(r, c)]


def test_nearest_drops_a_thin_line_that_max_keeps():
    # Column 161 is not one nearest-neighbour samples for 320 -> 84 (it keeps
    # j*320//84, which runs ..., 160, 163, ...), so the line falls between samples.
    sampled = set((np.arange(84) * 320 // 84).tolist())
    assert 161 not in sampled
    f = _frame_with_thin_line(161)
    near = PreprocessingPipeline(grayscale=True, resize=(84, 84)).process(f)
    mx = PreprocessingPipeline(grayscale=True, resize=(84, 84), resize_mode="max")
    mx = mx.process(f)
    assert near.max() == 0, "nearest kept a line it should have stepped over"
    # Pure red is 76 after grayscale, the darkest colour on L4's screen.
    assert mx.max() == 76, "max lost the one-pixel line"
    assert near.shape == mx.shape == (84, 84, 1)


def test_default_is_byte_identical_to_the_old_nearest_resize():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 256, (200, 320, 3), dtype=np.uint8)
    got = PreprocessingPipeline(grayscale=True, resize=(84, 84)).process(f)
    assert np.array_equal(got, _old_nearest(f, 84, 84))


def test_max_keeps_the_brightest_pixel_of_each_nearest_block():
    """Same blocks as nearest; max can only ever be >= it, pixel by pixel."""
    rng = np.random.default_rng(1)
    f = rng.integers(0, 256, (200, 320, 3), dtype=np.uint8)
    near = PreprocessingPipeline(grayscale=True, resize=(84, 84)).process(f)
    mx = PreprocessingPipeline(grayscale=True, resize=(84, 84), resize_mode="max")
    mx = mx.process(f)
    assert (mx >= near).all()
    gray = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(np.uint8)
    # Spot-check one block by hand: output (5, 7) covers rows 11..13, cols 26..29.
    r0, r1 = 5 * 200 // 84, 6 * 200 // 84
    c0, c1 = 7 * 320 // 84, 8 * 320 // 84
    assert mx[5, 7, 0] == gray[r0:r1, c0:c1].max()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        PreprocessingPipeline(resize=(84, 84), resize_mode="bilinear")


class _FakeEnv:
    def reset(self, seed=None):
        return np.zeros((200, 320, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((200, 320, 3), dtype=np.uint8), 0.0, False, False, {}


def _sig(**kw):
    p = PreprocessingPipeline(grayscale=True, resize=(84, 84), frame_stack=4, **kw)
    env = PreprocessedEnv(_FakeEnv(), p)
    env.reset()
    return env._stack_signature()


def test_default_stack_signature_is_unchanged_so_stored_pools_still_restore():
    """Every pool captured before this change stored a 5-tuple signature. A default
    pipeline must produce exactly that shape, or restore_frame_stack rejects them all
    and every seed silently falls back to re-seeding."""
    sig = _sig()
    assert len(sig) == 5
    assert sig == (True, (84, 84), 4, None, (84, 84, 1))


def test_a_max_stack_cannot_be_restored_into_a_nearest_pipeline():
    assert _sig() != _sig(resize_mode="max")


def test_envconfig_defaults_to_nearest():
    assert EnvConfig(profile="x").resize_mode == "nearest"


def _sweep():
    spec = importlib.util.spec_from_file_location("_yeti_keepbest_rm", _SWEEP)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"keep_best_sweep not importable here: {exc}")
    return mod


def _captured_eval_cmd(monkeypatch, tmp_path, **kw):
    mod = _sweep()
    seen = {}

    def fake_run(cmd, **_):
        seen["cmd"] = list(cmd)
        out = cmd[cmd.index("--out") + 1]
        pathlib.Path(out).write_text(
            '{"episodes": 1, "princess_touches": 0, "max_cp_counts": {}, "rows": []}'
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._eval_snapshot(
        "m.zip",
        1,
        "cpu",
        str(tmp_path / "e.json"),
        profile="p",
        fruits_total=1,
        start_state=None,
        stall_threshold=40,
        max_steps=10,
        level=4,
        **kw,
    )
    return seen["cmd"]


def test_referee_forwards_a_non_default_resize_mode(monkeypatch, tmp_path):
    cmd = _captured_eval_cmd(monkeypatch, tmp_path, resize_mode="max")
    assert cmd[cmd.index("--resize-mode") + 1] == "max"


def test_referee_command_is_unchanged_for_the_default(monkeypatch, tmp_path):
    cmd = _captured_eval_cmd(monkeypatch, tmp_path)
    assert "--resize-mode" not in cmd
