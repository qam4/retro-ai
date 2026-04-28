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
