"""Unit tests for retro_ai.training.run_manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from retro_ai.training.run_manifest import (
    EPISODE_COLUMNS,
    EpisodeLogger,
    RunManifest,
    iter_inner_envs,
    seed_everything,
)


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


def test_seed_everything_returns_given_seed():
    assert seed_everything(1234) == 1234


def test_seed_everything_picks_random_when_none():
    a = seed_everything(None)
    b = seed_everything(None)
    assert isinstance(a, int) and isinstance(b, int)
    # Extremely unlikely to collide on two consecutive nanosecond reads.
    assert a != b or a > 0


def test_seed_everything_is_reproducible():
    import random

    seed_everything(42)
    first = [random.random() for _ in range(5)]
    seed_everything(42)
    second = [random.random() for _ in range(5)]
    assert first == second


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------


def test_run_manifest_writes_yaml_and_json(tmp_path: Path):
    args = argparse.Namespace(
        timesteps=1000, profile="yeti_fruit", output=str(tmp_path / "out")
    )
    out_dir = tmp_path / "out"
    manifest = RunManifest.capture(args, str(out_dir), extras={"reward": "flat_10"})

    run_yaml = out_dir / "run.yaml"
    env_json = out_dir / "env.json"
    assert run_yaml.exists()
    assert env_json.exists()

    with env_json.open() as f:
        env = json.load(f)
    assert env["status"] == "RUNNING"
    assert env["exit_code"] is None
    assert env["finished_at"] is None
    assert "git" in env and "versions" in env
    assert "python" in env["versions"]

    manifest.finalize(status="COMPLETED", exit_code=0)
    with env_json.open() as f:
        env2 = json.load(f)
    assert env2["status"] == "COMPLETED"
    assert env2["exit_code"] == 0
    assert env2["finished_at"] is not None
    assert env2["wall_clock_sec"] is not None


def test_run_manifest_captures_extras(tmp_path: Path):
    args = {"foo": 1, "bar": "baz"}
    RunManifest.capture(args, str(tmp_path), extras={"ppo": {"lr": 3e-4}, "seed": 7})
    text = (tmp_path / "run.yaml").read_text()
    assert "ppo" in text and "seed" in text
    # args block present
    assert "foo" in text and "bar" in text


def test_run_manifest_accepts_dict_or_namespace(tmp_path: Path):
    ns = argparse.Namespace(a=1)
    d = {"a": 1}
    m1 = RunManifest.capture(ns, str(tmp_path / "ns"))
    m2 = RunManifest.capture(d, str(tmp_path / "d"))
    assert m1.args == m2.args == {"a": 1}


# ---------------------------------------------------------------------------
# EpisodeLogger
# ---------------------------------------------------------------------------


def _read_rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_episode_logger_writes_header_and_rows(tmp_path: Path):
    logger = EpisodeLogger(str(tmp_path))
    logger.log(
        env_id=0,
        episode_id=1,
        global_step=1000,
        start_level=1,
        reached_level=2,
        n_fruits_collected=1,
        length=250,
        total_reward=8.0,
        end_reason="death",
    )
    logger.close()

    csv_path = tmp_path / "episodes.csv"
    assert csv_path.exists()
    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["env_id"] == "0"
    assert rows[0]["start_level"] == "1"
    assert rows[0]["reached_level"] == "2"
    assert rows[0]["n_fruits_collected"] == "1"
    # Missing optional fields should be blank, not crash.
    assert rows[0]["start_x"] == ""


def test_episode_logger_appends_on_reopen(tmp_path: Path):
    log1 = EpisodeLogger(str(tmp_path))
    log1.log(env_id=0, episode_id=1, global_step=100)
    log1.close()

    log2 = EpisodeLogger(str(tmp_path))
    log2.log(env_id=0, episode_id=2, global_step=200)
    log2.close()

    rows = _read_rows(tmp_path / "episodes.csv")
    assert [r["episode_id"] for r in rows] == ["1", "2"]
    # Header should not be duplicated
    assert (tmp_path / "episodes.csv").read_text().count("episode_id") == 1


def test_episode_logger_ignores_unknown_keys(tmp_path: Path):
    logger = EpisodeLogger(str(tmp_path))
    logger.log(
        env_id=0,
        episode_id=1,
        global_step=100,
        not_a_real_column="should be dropped silently",
    )
    logger.close()
    rows = _read_rows(tmp_path / "episodes.csv")
    assert "not_a_real_column" not in rows[0]


def test_episode_logger_columns_stable():
    # Guard against accidental re-ordering of columns; downstream analysis
    # relies on this list being stable across refactors.
    expected_prefix = ["timestamp", "global_step", "env_id", "episode_id"]
    assert EPISODE_COLUMNS[: len(expected_prefix)] == expected_prefix
    assert "start_state_hash" in EPISODE_COLUMNS


def test_episode_logger_thread_safe(tmp_path: Path):
    import threading

    logger = EpisodeLogger(str(tmp_path))
    n_threads = 8
    per_thread = 50

    def worker(tid: int):
        for i in range(per_thread):
            logger.log(
                env_id=tid,
                episode_id=i,
                global_step=tid * per_thread + i,
                length=10,
                total_reward=1.0,
                end_reason="death",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.close()

    rows = _read_rows(tmp_path / "episodes.csv")
    assert len(rows) == n_threads * per_thread
    # Every row should have valid env_id and no row corruption.
    for r in rows:
        assert 0 <= int(r["env_id"]) < n_threads


# ---------------------------------------------------------------------------
# iter_inner_envs
# ---------------------------------------------------------------------------


class _FakeEnv:
    def __init__(self, name):
        self.name = name


class _FakeWrapper:
    def __init__(self, env):
        self.env = env


class _FakeVecEnv:
    def __init__(self, envs):
        self._envs = envs


class _FakeVecEnvWrapper:
    def __init__(self, venv):
        self.venv = venv


def test_iter_inner_envs_through_plain_vec_env():
    envs = [_FakeEnv("a"), _FakeEnv("b")]
    vec = _FakeVecEnv(envs)
    found = list(iter_inner_envs(vec))
    assert [e.name for e in found] == ["a", "b"]


def test_iter_inner_envs_peels_monitor_layers():
    envs = [_FakeWrapper(_FakeWrapper(_FakeEnv("inner")))]
    vec = _FakeVecEnv(envs)
    found = list(iter_inner_envs(vec))
    assert [e.name for e in found] == ["inner"]


def test_iter_inner_envs_walks_vec_wrapper_chain():
    # This is the original bug: SB3 wraps ThreadedVecEnv in VecTransposeImage,
    # which exposes .venv but not ._envs.
    envs = [_FakeEnv("x")]
    inner_vec = _FakeVecEnv(envs)
    outer = _FakeVecEnvWrapper(_FakeVecEnvWrapper(inner_vec))
    found = list(iter_inner_envs(outer))
    assert [e.name for e in found] == ["x"]


def test_iter_inner_envs_returns_nothing_if_no_inner():
    # Some non-vec object with no _envs and no venv.
    class _Bare:
        pass

    assert list(iter_inner_envs(_Bare())) == []


# --- native (emulator) provenance -------------------------------------------
# The git SHA cannot identify which EMULATOR produced a run: the compiled module
# is unversioned, so it may be stale or ahead of HEAD, and checking out an older
# commit does not revert it. That gap cost a real investigation — an L1 champion
# documented at 99.7% scored 0% because it had been trained against a core whose
# state-restore was broken (experiments/003-yeti/core_provenance_2b0a45d.md).


def test_native_info_has_expected_keys():
    from retro_ai.training.run_manifest import _native_info

    info = _native_info()
    assert set(info) == {"path", "sha256", "size", "mtime"}


def test_native_info_hashes_the_loaded_module_when_available():
    """When the native module resolves, we must record a content hash — a path
    and mtime alone do not pin the binary."""
    import importlib.util

    from retro_ai.training.run_manifest import _native_info

    if importlib.util.find_spec("retro_ai_native") is None:
        pytest.skip("native module not built in this environment")
    info = _native_info()
    assert info["path"] and info["path"].endswith((".so", ".pyd"))
    assert info["sha256"] and len(info["sha256"]) == 64
    assert isinstance(info["size"], int) and info["size"] > 0
    assert info["mtime"]


def test_native_info_never_raises(monkeypatch):
    """Provenance is best-effort: a broken environment must not fail a run."""
    import sys as _sys

    from retro_ai.training import run_manifest as rm

    monkeypatch.setitem(_sys.modules, "retro_ai_native", None)
    monkeypatch.setattr(
        rm.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert rm._native_info()["sha256"] is None


def test_manifest_records_native_alongside_git(tmp_path):
    """The manifest must carry both, so a policy is traceable to its emulator."""
    from retro_ai.training.run_manifest import RunManifest

    RunManifest.capture(args={"config_path": "x.yaml"}, output_dir=str(tmp_path))
    env = json.loads((tmp_path / "env.json").read_text())
    assert "git" in env and "native" in env
    assert set(env["native"]) == {"path", "sha256", "size", "mtime"}


def test_git_info_dumps_the_diff_when_dirty(tmp_path, monkeypatch):
    """A SHA plus ``dirty: true`` does not identify the code a run used.

    L4's v24, v25, v26 and v27 -- four 6M runs and the readout the rope-2 work rests on --
    ALL recorded commit b6cba78 with ``dirty: true``, so they cannot be told apart from
    their artefacts and a 1.2-rung gap between them cannot be attributed to any lever.
    This pins the patch dump that makes a dirty run reconstructible, and the untracked
    list that states what the patch CANNOT cover.
    """
    import subprocess as sp

    from retro_ai.training.run_manifest import _git_info

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        sp.check_call(
            ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )

    git("init", "-q")
    (repo / "tracked.py").write_text("x = 1\n")
    git("add", "tracked.py")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")

    (repo / "tracked.py").write_text("x = 2\n")
    (repo / "untracked.py").write_text("secret = 3\n")

    monkeypatch.chdir(repo)
    out = tmp_path / "out"
    out.mkdir()
    info = _git_info(dump_dir=str(out))

    assert info["dirty"] is True
    assert info["diff_file"] == "run_dirty.patch"
    # Read back with the SAME encoding and no newline translation the writer used, so
    # this assertion means "the bytes round-tripped" on Windows as well as here.
    patch = (out / "run_dirty.patch").read_text(encoding="utf-8")
    assert "tracked.py" in patch and "x = 2" in patch
    assert info["diff_bytes"] == len(patch.encode())
    assert info["untracked"] == ["untracked.py"]
    # Untracked CONTENTS are deliberately not captured: dumping arbitrary untracked
    # files into a run directory is how secrets and unrelated work in progress escape.
    assert "secret = 3" not in patch


def test_git_info_on_a_clean_tree_writes_no_patch(tmp_path, monkeypatch):
    """The dump must be inert when there is nothing uncommitted."""
    import subprocess as sp

    from retro_ai.training.run_manifest import _git_info

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        sp.check_call(
            ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )

    git("init", "-q")
    (repo / "tracked.py").write_text("x = 1\n")
    git("add", "tracked.py")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")

    monkeypatch.chdir(repo)
    out = tmp_path / "out"
    out.mkdir()
    info = _git_info(dump_dir=str(out))

    assert info["dirty"] is False
    assert "diff_file" not in info
    assert not list(out.iterdir())


def test_finalize_keeps_the_launch_commit_and_patch(tmp_path, monkeypatch):
    """env.json must describe the code a run LAUNCHED with, not the tree at the end.

    finalize() used to re-capture git state, so a run that launched clean and finished
    after an unrelated edit recorded the later commit, `dirty: true`, and a
    run_dirty.patch of code it never executed. Measured on L1 v17: launched from a
    clean 571c359, recorded 811e35a dirty.
    """
    import json
    import subprocess as sp

    from retro_ai.training.run_manifest import RunManifest

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        sp.check_call(
            ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )

    ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    git("init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git(*ident, "commit", "-qm", "init")
    launch_sha = sp.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    monkeypatch.chdir(repo)
    out = tmp_path / "run"
    m = RunManifest.capture({"x": 1}, str(out))

    # Mid-run: someone commits, then leaves an uncommitted edit.
    (repo / "a.py").write_text("x = 2\n")
    git(*ident, "commit", "-qam", "later")
    (repo / "a.py").write_text("x = 3\n")

    m.finalize(status="COMPLETED", exit_code=0)
    env = json.loads((out / "env.json").read_text())
    assert env["status"] == "COMPLETED"
    assert env["git"]["commit"] == launch_sha
    assert env["git"]["dirty"] is False
    assert not (out / "run_dirty.patch").exists()
    assert env["wall_clock_sec"] is not None
