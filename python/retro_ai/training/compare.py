"""Run comparison utilities for comparing training run results."""

import json
import os
import sys
from typing import Dict, List


class RunComparator:
    """Load and compare summary.json files from multiple training runs."""

    def __init__(self, output_dirs: List[str]) -> None:
        self.output_dirs = output_dirs
        self._summaries: List[Dict] = []

    def load_summaries(self) -> List[Dict]:
        """Load summary.json from each output directory.

        Skips missing or malformed files with a warning to stderr.
        Each dict includes output_dir, config_name, plus all summary.json fields.
        """
        self._summaries = []
        for d in self.output_dirs:
            path = os.path.join(d, "summary.json")
            if not os.path.isfile(path):
                print(f"Warning: {path} not found, skipping", file=sys.stderr)
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"Warning: failed to read {path}: {exc}, skipping",
                    file=sys.stderr,
                )
                continue
            data["output_dir"] = d
            data["config_name"] = os.path.basename(d.rstrip("/"))
            self._summaries.append(data)
        return self._summaries

    def flag_nonfunctional(self, summary: Dict) -> bool:
        """Return True when total_episodes == 0."""
        return summary.get("total_episodes", 0) == 0

    def compare(self) -> str:
        """Return a formatted comparison table ranked by mean_reward descending.

        Flags non-functional runs (0 episodes) with a '*' marker.
        Returns a message if no valid runs are found.
        """
        if not self._summaries:
            return "No valid training runs found"

        ranked = sorted(
            self._summaries, key=lambda s: s.get("mean_reward", 0), reverse=True
        )

        header = (
            f"{'Rank':>4} | {'Config':<40} | {'Episodes':>8} | "
            f"{'Mean Reward':>11} | {'Best Reward':>11} | {'Wall Clock':>10}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for i, s in enumerate(ranked, 1):
            nf = "*" if self.flag_nonfunctional(s) else " "
            config = s.get("config_name", "unknown")
            episodes = s.get("total_episodes", 0)
            mean_r = s.get("mean_reward", 0.0)
            best_r = s.get("best_reward", 0.0)
            wall = s.get("wall_clock_seconds", 0.0)
            lines.append(
                f"{i:>4} | {config:<40} | {episodes:>8} | "
                f"{mean_r:>11.1f} | {best_r:>11.1f} | {wall:>9.1f}s{nf}"
            )

        return "\n".join(lines)

    def to_json(self) -> str:
        """Return JSON output per the design's comparison output schema."""
        if not self._summaries:
            return json.dumps({"runs": [], "ranked_by": "mean_reward"}, indent=2)

        ranked = sorted(
            self._summaries, key=lambda s: s.get("mean_reward", 0), reverse=True
        )

        runs = []
        for s in ranked:
            runs.append(
                {
                    "output_dir": s.get("output_dir", ""),
                    "config_name": s.get("config_name", "unknown"),
                    "total_episodes": s.get("total_episodes", 0),
                    "mean_reward": s.get("mean_reward", 0.0),
                    "best_reward": s.get("best_reward", 0.0),
                    "mean_length": s.get("mean_length", 0.0),
                    "wall_clock_seconds": s.get("wall_clock_seconds", 0.0),
                    "functional": not self.flag_nonfunctional(s),
                }
            )

        return json.dumps({"runs": runs, "ranked_by": "mean_reward"}, indent=2)
