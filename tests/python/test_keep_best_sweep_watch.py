"""`keep_best_sweep --watch` must tolerate a snapshots dir that does not exist yet.

WHY THIS EXISTS. The documented way to use the parallel evaluator is to launch it
alongside a fresh training run, but the trainer creates `snapshots/` lazily, when it
writes its FIRST snapshot. So the evaluator races it, and `_snapshots()` called
`os.listdir` on a directory that was not there yet.

Measured cost of that race on 2026-09-26: the evaluator died with FileNotFoundError 10
seconds after launch, the v26 run trained for 1h40m with no referee, and
`training.on_regression` could not fire because nothing was writing
`best/eval_status.json`. Recoverable only because snapshots stay on disk and could be
swept afterwards. Nothing in the status file said "crashed" loudly enough to notice: a
check 10 s after launch still read RUNNING.

`--watch` exists precisely to wait for snapshots that do not exist yet, so the empty
case is normal operation, not an error.

No emulator needed.
"""

import importlib.util
import pathlib

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "keep_best_sweep.py"
)


def _mod():
    spec = importlib.util.spec_from_file_location("_yeti_keepbest", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"keep_best_sweep not importable here: {exc}")
    return mod


def test_snapshots_of_missing_dir_is_empty_not_an_error(tmp_path):
    """The race that killed a 6M run's referee. Empty, not FileNotFoundError."""
    mod = _mod()
    assert mod._snapshots(str(tmp_path / "run" / "snapshots")) == []


def test_snapshots_of_empty_dir_is_empty(tmp_path):
    """The dir exists but the first snapshot has not landed yet -- also normal."""
    mod = _mod()
    d = tmp_path / "snapshots"
    d.mkdir()
    assert mod._snapshots(str(d)) == []


def test_snapshots_are_returned_sorted_by_step(tmp_path):
    """Guards the guard: an early return must not break the normal path.

    Sorted by STEP as an integer, not lexically -- `model_900000` must come before
    `model_1000000`, which string ordering gets wrong.
    """
    mod = _mod()
    d = tmp_path / "snapshots"
    d.mkdir()
    for step in (1000000, 900000, 100000):
        (d / f"model_{step}_steps.zip").write_bytes(b"")
    (d / "not_a_snapshot.txt").write_bytes(b"")
    got = mod._snapshots(str(d))
    assert [s for s, _ in got] == [100000, 900000, 1000000]
    assert all(p.endswith(".zip") for _, p in got)


def test_watch_keeps_going_when_snapshots_arrive_during_a_long_batch(
    tmp_path, monkeypatch
):
    """The idle timer must measure IDLE time, not time spent evaluating.

    `last_new` was stamped once per batch, and the idle check is `now - last_new`, so a
    batch taking longer than `--max-idle-min` made the loop exit after one pass. The
    snapshots it lost were the ones training wrote WHILE it was evaluating: the dir
    listing is taken at the top of the pass, so anything arriving mid-batch is only
    picked up on the next pass -- which never came.

    Measured on v26: it evaluated 36 of 60, printed "idle 32 min, stopping" after 32
    minutes of continuous work, and left 24 unevaluated, including the window where the
    CONTROL run had its best snapshot. The comparison the run existed for was missing.

    Uses a FAKE CLOCK, not sleeps. A first version slept 0.4 s per eval against a 0.3 s
    threshold and passed or failed depending on machine load -- it failed, then passed
    on unchanged code. A guard that flickers is worse than none, because it teaches you
    to ignore it.
    """
    import sys

    mod = _mod()
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for step in (100000, 200000):
        (snaps / f"model_{step}_steps.zip").write_bytes(b"")

    class Clock:
        """One eval costs 100 s; --max-idle-min is 1 (60 s). So a single eval outlasts
        the idle threshold, which is the condition that exposed the bug."""

        def __init__(self):
            self.t = 0.0

        def time(self):
            return self.t

        def sleep(self, n):
            # MUST advance the clock, or the idle check can never fire once the work
            # runs out and --watch spins forever. A first version left this a no-op and
            # the test hung.
            self.t += n

    clock = Clock()
    monkeypatch.setattr(mod, "time", clock)

    calls = []

    def fake_eval(path, episodes, device, tmp_json, **kw):
        calls.append(path)
        if len(calls) == 1:
            # training writes another snapshot while we are busy
            (snaps / "model_300000_steps.zip").write_bytes(b"")
        clock.t += 100.0
        return 0.0, 1.0, 6.0, 13, {"Lfruit_bot": 1.0}

    monkeypatch.setattr(mod, "_eval_snapshot", fake_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "keep_best_sweep.py",
            "--snapshots-dir",
            str(snaps),
            "--watch",
            # poll MUST be non-zero: the fake clock only advances via eval and sleep, so
            # a 0 poll leaves idle_min pinned at 0 and --watch spins forever.
            "--poll-sec",
            "30",
            "--max-idle-min",
            "1",
            "--episodes",
            "1",
            "--level",
            "4",
            "--profile",
            "yeti_fruit_level4",
            "--fruits-total",
            "1",
        ],
    )
    mod.main()

    assert sorted(pathlib.Path(p).name for p in calls) == [
        "model_100000_steps.zip",
        "model_200000_steps.zip",
        "model_300000_steps.zip",
    ], f"snapshot arriving mid-batch was never evaluated; evaluated {calls}"
