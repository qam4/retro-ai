"""End a run when the parallel evaluator says the policy got worse.

WHY THIS EXISTS. Measured on v24 by the parallel evaluator (keep_best_sweep.py
--watch, 56 snapshots at 30 episodes each): 34 of 56 evals were below the best already
seen, the run ended 45% below its own peak, and at 4.6M the frontier collapsed from
`Low2_launch` back to `Lfruit_bot` -- route position 5. The tail has never produced the
champion on this project either: v18's best snapshot was at 900k of 15M, and v21
matched v18 in 6M rather than 15M. Stopping at the peak loses nothing measured and
returns hours per run.

It reads a FILE rather than evaluating in-process, because the Crayon emulator keeps
in-process global state and evaluation must not share a process with training. That
also makes the coupling honest: the evaluator runs whether or not training watches,
and training degrades to a no-op when the evaluator is absent.

It only STOPS. Reverting weights is the next increment and this is its control arm --
reverting alone is measured not to work here (control arm A0: a champion at mean depth
9.55 put back into training read 1.20 / 2.43 / 7.53 / 1.03 over the next 1M steps).

No emulator needed.
"""

import importlib.util
import json
import pathlib

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)


def _mod():
    spec = importlib.util.spec_from_file_location("_yeti_regstop", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _cb(mod, path, patience=3, check_freq=10):
    cb = mod.RegressionStopCallback(str(path), patience=patience, check_freq=check_freq)
    cb.num_timesteps = 0
    return cb


def _write(path, n, step=100000):
    path.write_text(
        json.dumps(
            {
                "step": step,
                "consecutive_regressions": n,
                "frontier": "Low2_launch",
                "frontier_rate": 0.40,
                "best_step": 3800000,
                "best_frontier": "Low2_launch",
                "best_frontier_rate": 0.73,
                "best_model": "best/best_model.zip",
            }
        )
    )


def _step(cb, to):
    """Advance to ``to`` timesteps and return the callback's verdict."""
    cb.num_timesteps = to
    return cb._on_step()


def test_a_missing_status_file_is_a_no_op():
    """The evaluator is a separate process and may simply not be running."""
    mod = _mod()
    cb = _cb(mod, pathlib.Path("/nonexistent/eval_status.json"))
    assert _step(cb, 10) is True
    assert _step(cb, 20) is True


def test_a_half_written_file_is_a_no_op(tmp_path):
    """It is rewritten after every eval, so a read can land mid-write."""
    mod = _mod()
    p = tmp_path / "eval_status.json"
    p.write_text('{"step": 1, "consecutive_re')
    cb = _cb(mod, p)
    assert _step(cb, 10) is True


def test_below_patience_keeps_training(tmp_path):
    mod = _mod()
    p = tmp_path / "eval_status.json"
    cb = _cb(mod, p, patience=3)
    for n in (0, 1, 2):
        _write(p, n, step=100000 + n)
        assert _step(cb, 10 * (n + 1)) is True, n


def test_at_patience_stops(tmp_path):
    mod = _mod()
    p = tmp_path / "eval_status.json"
    _write(p, 3)
    cb = _cb(mod, p, patience=3)
    assert _step(cb, 10) is False


def test_patience_is_respected_not_hardcoded(tmp_path):
    mod = _mod()
    p = tmp_path / "eval_status.json"
    _write(p, 3)
    assert _step(_cb(mod, p, patience=5), 10) is True
    assert _step(_cb(mod, p, patience=2), 10) is False


def test_it_does_not_read_on_every_step(tmp_path):
    """One file read per `check_freq` timesteps, not per step -- this runs inside the
    PPO loop."""
    mod = _mod()
    p = tmp_path / "eval_status.json"
    _write(p, 99)
    cb = _cb(mod, p, patience=3, check_freq=1000)
    assert _step(cb, 10) is True, "should not have read the file yet"
    assert _step(cb, 1000) is False, "should have read it once past check_freq"


def test_a_recovered_run_is_not_stopped(tmp_path):
    """`consecutive_regressions` resets to 0 in the evaluator when a snapshot is no
    longer regressing, so a run that recovers must survive."""
    mod = _mod()
    p = tmp_path / "eval_status.json"
    cb = _cb(mod, p, patience=3)
    _write(p, 2, step=1)
    assert _step(cb, 10) is True
    _write(p, 0, step=2)
    assert _step(cb, 20) is True
    _write(p, 1, step=3)
    assert _step(cb, 30) is True


def test_off_by_default_in_the_config():
    from retro_ai.training.run_config import TrainingConfig

    cfg = TrainingConfig(timesteps=1, output="x")
    assert cfg.on_regression == "off"
    assert cfg.regression_patience == 3
