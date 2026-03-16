"""Tests for RunComparator in retro_ai.training.compare."""

import json
import os
import tempfile

import pytest

from retro_ai.training.compare import RunComparator


def _make_summary_dir(tmp_path, name, summary_data):
    """Create a temp output dir with a summary.json file."""
    d = os.path.join(tmp_path, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary_data, f)
    return d


class TestRunComparatorInit:
    def test_stores_output_dirs(self):
        rc = RunComparator(["a", "b"])
        assert rc.output_dirs == ["a", "b"]


class TestLoadSummaries:
    def test_loads_valid_summary(self, tmp_path):
        d = _make_summary_dir(
            str(tmp_path),
            "run1",
            {
                "total_episodes": 10,
                "mean_reward": 5.0,
                "best_reward": 20.0,
                "wall_clock_seconds": 100.0,
            },
        )
        rc = RunComparator([d])
        summaries = rc.load_summaries()
        assert len(summaries) == 1
        assert summaries[0]["total_episodes"] == 10
        assert summaries[0]["config_name"] == "run1"
        assert summaries[0]["functional"] is True

    def test_skips_missing_directory(self, tmp_path, capsys):
        rc = RunComparator([str(tmp_path / "nonexistent")])
        summaries = rc.load_summaries()
        assert len(summaries) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_skips_malformed_json(self, tmp_path, capsys):
        d = os.path.join(str(tmp_path), "bad")
        os.makedirs(d)
        with open(os.path.join(d, "summary.json"), "w") as f:
            f.write("{invalid json")
        rc = RunComparator([d])
        summaries = rc.load_summaries()
        assert len(summaries) == 0
        captured = capsys.readouterr()
        assert "failed to read" in captured.err

    def test_functional_field_false_when_zero_episodes(self, tmp_path):
        d = _make_summary_dir(
            str(tmp_path),
            "dead_run",
            {
                "total_episodes": 0,
                "mean_reward": 0.0,
                "best_reward": 0.0,
                "wall_clock_seconds": 50.0,
            },
        )
        rc = RunComparator([d])
        summaries = rc.load_summaries()
        assert summaries[0]["functional"] is False

    def test_loads_multiple_dirs(self, tmp_path):
        d1 = _make_summary_dir(
            str(tmp_path),
            "run_a",
            {
                "total_episodes": 5,
                "mean_reward": 3.0,
                "best_reward": 10.0,
                "wall_clock_seconds": 60.0,
            },
        )
        d2 = _make_summary_dir(
            str(tmp_path),
            "run_b",
            {
                "total_episodes": 15,
                "mean_reward": 8.0,
                "best_reward": 30.0,
                "wall_clock_seconds": 120.0,
            },
        )
        rc = RunComparator([d1, d2])
        summaries = rc.load_summaries()
        assert len(summaries) == 2


class TestFlagNonfunctional:
    def test_zero_episodes_is_nonfunctional(self):
        rc = RunComparator([])
        assert rc.flag_nonfunctional({"total_episodes": 0}) is True

    def test_positive_episodes_is_functional(self):
        rc = RunComparator([])
        assert rc.flag_nonfunctional({"total_episodes": 1}) is False

    def test_missing_key_treated_as_nonfunctional(self):
        rc = RunComparator([])
        assert rc.flag_nonfunctional({}) is True


class TestCompare:
    def test_exits_with_code_1_when_no_summaries(self):
        rc = RunComparator([])
        rc._summaries = []
        with pytest.raises(SystemExit) as exc_info:
            rc.compare()
        assert exc_info.value.code == 1

    def test_ranked_by_mean_reward_descending(self, tmp_path):
        d1 = _make_summary_dir(
            str(tmp_path),
            "low",
            {
                "total_episodes": 5,
                "mean_reward": 2.0,
                "best_reward": 5.0,
                "wall_clock_seconds": 50.0,
            },
        )
        d2 = _make_summary_dir(
            str(tmp_path),
            "high",
            {
                "total_episodes": 10,
                "mean_reward": 20.0,
                "best_reward": 50.0,
                "wall_clock_seconds": 100.0,
            },
        )
        rc = RunComparator([d1, d2])
        rc.load_summaries()
        table = rc.compare()
        lines = table.strip().split("\n")
        # First data line (after header + separator) should be the higher reward
        data_lines = lines[2:]
        assert "high" in data_lines[0]
        assert "low" in data_lines[1]

    def test_table_contains_expected_columns(self, tmp_path):
        d = _make_summary_dir(
            str(tmp_path),
            "test_run",
            {
                "total_episodes": 42,
                "mean_reward": 18.5,
                "best_reward": 70.0,
                "wall_clock_seconds": 312.4,
            },
        )
        rc = RunComparator([d])
        rc.load_summaries()
        table = rc.compare()
        assert "Rank" in table
        assert "Config" in table
        assert "Episodes" in table
        assert "Mean Reward" in table
        assert "Best Reward" in table
        assert "Wall Clock" in table
        assert "test_run" in table
        assert "18.5" in table
        assert "70.0" in table

    def test_nonfunctional_run_flagged_with_star(self, tmp_path):
        d = _make_summary_dir(
            str(tmp_path),
            "dead",
            {
                "total_episodes": 0,
                "mean_reward": 0.0,
                "best_reward": 0.0,
                "wall_clock_seconds": 10.0,
            },
        )
        rc = RunComparator([d])
        rc.load_summaries()
        table = rc.compare()
        # The non-functional run should have a '*' marker
        assert "*" in table


class TestToJson:
    def test_empty_summaries(self):
        rc = RunComparator([])
        rc._summaries = []
        result = json.loads(rc.to_json())
        assert result["runs"] == []
        assert result["ranked_by"] == "mean_reward"

    def test_includes_functional_field(self, tmp_path):
        d = _make_summary_dir(
            str(tmp_path),
            "run1",
            {
                "total_episodes": 10,
                "mean_reward": 5.0,
                "best_reward": 20.0,
                "mean_length": 100.0,
                "wall_clock_seconds": 60.0,
            },
        )
        rc = RunComparator([d])
        rc.load_summaries()
        result = json.loads(rc.to_json())
        assert len(result["runs"]) == 1
        assert result["runs"][0]["functional"] is True
