"""Metrics tracking, CSV persistence, and summary generation."""

import csv
import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional


class MetricsTracker:
    """Track episode metrics, persist to CSV, and generate JSON summaries."""

    def __init__(self, output_dir: str, rolling_window: int = 100):
        self._output_dir = output_dir
        self._rolling_window = rolling_window
        self._episodes: List[Dict[str, Any]] = []
        self._buffer: List[Dict[str, Any]] = []
        self._rewards: deque = deque(maxlen=rolling_window)
        self._lengths: deque = deque(maxlen=rolling_window)
        self._best_reward: float = float("-inf")
        self._start_time: float = time.monotonic()
        self._total_timesteps: int = 0
        self._csv_path = os.path.join(output_dir, "metrics.csv")
        self._summary_path = os.path.join(output_dir, "summary.json")

    def record_episode(self, reward: float, length: int, info: dict) -> None:
        """Buffer an episode with reward, length, score, and timestamp."""
        episode = {
            "episode": len(self._episodes) + 1,
            "reward": reward,
            "length": length,
            "score": info.get("score"),
            "timestamp": time.time(),
        }
        self._episodes.append(episode)
        self._buffer.append(episode)
        self._rewards.append(reward)
        self._lengths.append(length)
        self._total_timesteps += length
        if reward > self._best_reward:
            self._best_reward = reward

    def rolling_reward(self) -> Optional[float]:
        """Mean reward over the last rolling_window episodes."""
        if not self._rewards:
            return None
        return sum(self._rewards) / len(self._rewards)

    def rolling_length(self) -> Optional[float]:
        """Mean length over the last rolling_window episodes."""
        if not self._lengths:
            return None
        return sum(self._lengths) / len(self._lengths)

    def best_reward(self) -> float:
        """Maximum reward seen across all episodes."""
        return self._best_reward

    def flush_csv(self) -> None:
        """Append buffered episodes to CSV, creating the file if needed."""
        if not self._buffer:
            return
        os.makedirs(self._output_dir, exist_ok=True)
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            fields = ["episode", "reward", "length", "score", "timestamp"]
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(self._buffer)
        self._buffer.clear()

    def write_summary(self) -> None:
        """Write a JSON summary of all recorded episodes."""
        os.makedirs(self._output_dir, exist_ok=True)
        rewards = [e["reward"] for e in self._episodes]
        lengths = [e["length"] for e in self._episodes]
        wall_clock = time.monotonic() - self._start_time
        summary = {
            "total_episodes": len(self._episodes),
            "total_timesteps": self._total_timesteps,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "std_reward": _std(rewards),
            "best_reward": self._best_reward if rewards else 0.0,
            "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "wall_clock_seconds": round(wall_clock, 1),
        }
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def load_existing(self, csv_path: str) -> None:
        """Resume from an existing CSV file, restoring episode history."""
        if not os.path.exists(csv_path):
            return
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_score = row.get("score", "")
                score: Optional[float] = None
                if raw_score and raw_score != "None":
                    try:
                        score = float(raw_score)
                    except ValueError:
                        pass
                reward = float(row["reward"])
                length = int(row["length"])
                episode: Dict[str, Any] = {
                    "episode": int(row["episode"]),
                    "reward": reward,
                    "length": length,
                    "score": score,
                    "timestamp": float(row["timestamp"]),
                }
                self._episodes.append(episode)
                self._rewards.append(reward)
                self._lengths.append(length)
                self._total_timesteps += length
                if reward > self._best_reward:
                    self._best_reward = reward
        # Point CSV path to the loaded file for subsequent flushes
        self._csv_path = csv_path


def _std(values: list) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5
