"""Run manifest + per-episode logging for training scripts.

Every long training run should be reproducible from artifacts alone.
This module writes three files alongside a run's output:

- ``run.yaml``   — full CLI args + any extras the script wants to record
                    (reward formula id, PPO hyperparams, etc.).
- ``env.json``   — machine context: git SHA, dirty flag, library versions,
                    hostname, timestamps, seed, Python argv.
- ``episodes.csv`` — one row per terminated episode (via ``EpisodeLogger``).

Typical use
-----------
>>> manifest = RunManifest.capture(args, args.output, extras={"reward": "bonus*0.01"})
>>> episode_logger = EpisodeLogger(args.output)
>>> # ... inside env.step on done/truncated:
>>> episode_logger.log(env_id=0, episode_id=12, global_step=45_000, ...)
>>> # ... at end of training:
>>> manifest.finalize(status="COMPLETED", exit_code=0)
>>> episode_logger.close()
"""

from __future__ import annotations

import collections
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is in requirements
    yaml = None


EPISODE_COLUMNS = [
    "timestamp",
    "global_step",
    "env_id",
    "episode_id",
    "start_level",
    "reached_level",
    "n_fruits_collected",
    "length",
    "total_reward",
    "end_reason",
    "start_x",
    "start_y",
    "start_score",
    "start_bonus",
    "final_x",
    "final_y",
    "final_score",
    "final_bonus",
    "start_state_hash",
]


def _git_info() -> Dict[str, Any]:
    """Capture git SHA, dirty flag, and branch. Best-effort, never raises."""
    info: Dict[str, Any] = {
        "commit": None,
        "dirty": None,
        "branch": None,
    }
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return info
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        info["dirty"] = bool(status.strip())
    except Exception:
        pass
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass
    return info


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": platform.python_version(),
    }
    for mod_name in ("numpy", "torch", "stable_baselines3", "gymnasium"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not_installed"
    return versions


def seed_everything(seed: Optional[int]) -> Optional[int]:
    """Seed ``random``, ``numpy``, ``torch``, and SB3. Returns the resolved seed.

    If ``seed`` is ``None``, picks one from ``time.time_ns() & 0xFFFFFFFF``.
    """
    if seed is None:
        seed = int(time.time_ns() & 0xFFFFFFFF)
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except ImportError:
        pass
    return seed


@dataclass
class RunManifest:
    """Persisted metadata for a training run.

    Writes ``run.yaml`` and ``env.json`` to ``output_dir`` on capture.
    Call :meth:`finalize` to record end time + status.
    """

    output_dir: str
    args: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    _env_path: str = field(init=False)
    _run_path: str = field(init=False)
    _started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self._env_path = os.path.join(self.output_dir, "env.json")
        self._run_path = os.path.join(self.output_dir, "run.yaml")

    @classmethod
    def capture(
        cls,
        args: Any,
        output_dir: str,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> "RunManifest":
        """Create a manifest, capture machine context, and write initial files.

        Parameters
        ----------
        args : argparse.Namespace or dict
            Run configuration to persist.
        output_dir : str
            Where to write ``run.yaml`` and ``env.json``.
        extras : mapping, optional
            Additional fields to persist in ``run.yaml`` (e.g. resolved reward
            formula, PPO hyperparameters). Merged under the ``extras`` key.
        """
        args_dict = vars(args) if not isinstance(args, dict) else dict(args)
        manifest = cls(
            output_dir=output_dir,
            args=args_dict,
            extras=dict(extras) if extras else {},
        )
        manifest._write_run_yaml()
        manifest._write_env_json(status="RUNNING", exit_code=None)
        return manifest

    def finalize(self, status: str = "COMPLETED", exit_code: Optional[int] = 0) -> None:
        """Write finish timestamp + status to ``env.json``."""
        self._write_env_json(status=status, exit_code=exit_code)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _write_run_yaml(self) -> None:
        payload = {
            "args": self.args,
            "extras": self.extras,
        }
        with open(self._run_path, "w") as f:
            if yaml is not None:
                yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
            else:  # fallback: JSON is valid YAML
                json.dump(payload, f, indent=2, default=str)

    def _write_env_json(self, status: str, exit_code: Optional[int]) -> None:
        payload = {
            "status": status,
            "exit_code": exit_code,
            "started_at": self._started_at,
            "started_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(self._started_at)
            ),
            "finished_at": None,
            "wall_clock_sec": None,
            "git": _git_info(),
            "versions": _library_versions(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        }
        if status != "RUNNING":
            now = time.time()
            payload["finished_at"] = now
            payload["finished_at_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(now)
            )
            payload["wall_clock_sec"] = round(now - self._started_at, 3)
        with open(self._env_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


class EpisodeLogger:
    """Thread-safe per-episode CSV logger.

    Call :meth:`log` from inside each env's ``step()`` when the episode
    terminates (done or truncated). Columns are defined by
    :data:`EPISODE_COLUMNS`; any unknown keys are ignored silently, any
    missing keys are written as empty strings.

    A small in-memory ring buffer of recently-logged episodes is also
    maintained, so callers like a live-metrics TensorBoard callback can
    pull the last N episodes without re-reading the CSV.
    """

    def __init__(
        self,
        output_dir: str,
        filename: str = "episodes.csv",
        ring_size: int = 4096,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, filename)
        self._lock = threading.Lock()
        new_file = not os.path.exists(self._path) or os.path.getsize(self._path) == 0
        # Open in append mode so resumed runs keep adding rows.
        self._fh = open(self._path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(
            self._fh, fieldnames=EPISODE_COLUMNS, extrasaction="ignore"
        )
        if new_file:
            self._writer.writeheader()
            self._fh.flush()
        # Ring buffer of recent episodes. Uses a deque for O(1) append/pop;
        # ring_size is chosen to comfortably cover any reasonable TB window.
        self._recent: "collections.deque[Dict[str, Any]]" = collections.deque(
            maxlen=ring_size
        )

    def log(self, **row: Any) -> None:
        """Write one episode row. Keys must match :data:`EPISODE_COLUMNS`."""
        row.setdefault("timestamp", time.time())
        with self._lock:
            self._writer.writerow(row)
            # buffering=1 is line-buffered for text files; explicit flush is
            # still cheap insurance against interpreter-crash loss.
            self._fh.flush()
            self._recent.append(dict(row))

    def recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a snapshot of the most recently logged episodes.

        If ``n`` is ``None`` (default), returns all episodes currently in
        the ring buffer (up to ``ring_size``). Otherwise returns up to the
        last ``n`` episodes.
        """
        with self._lock:
            if n is None or n >= len(self._recent):
                return list(self._recent)
            return list(self._recent)[-n:]

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    # Context-manager convenience
    def __enter__(self) -> "EpisodeLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def iter_inner_envs(training_env: Any):
    """Yield the user-defined envs inside an SB3 VecEnv chain.

    SB3 may wrap our :class:`ThreadedVecEnv` in ``VecTransposeImage`` (or
    other ``VecEnvWrapper`` subclasses), which hides the ``_envs`` attribute.
    Walk down the ``.venv`` chain until we find a vec env that exposes
    ``_envs``, then peel off any ``Monitor`` / ``gym.Wrapper`` layers to
    return the user env.
    """
    vec = training_env
    for _ in range(16):  # paranoia cap
        if hasattr(vec, "_envs"):
            break
        vec = getattr(vec, "venv", None)
        if vec is None:
            return
    if not hasattr(vec, "_envs"):
        return
    for env in vec._envs:
        inner = env
        for _ in range(16):
            nxt = getattr(inner, "env", None)
            if nxt is None:
                break
            inner = nxt
        yield inner


__all__ = [
    "EPISODE_COLUMNS",
    "EpisodeLogger",
    "RunManifest",
    "iter_inner_envs",
    "seed_everything",
]
