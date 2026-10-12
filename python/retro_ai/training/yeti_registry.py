"""Yeti champion registry: which policy plays which game level, and how to evaluate it.

WHY. Every evaluation used to be set up by hand, with six or seven flags per call
(level, profile, start state, fruit count, stop rules, resize mode, reach mode), and a
wrong flag does not error, it produces a wrong number. Each of these happened on this
project: the referee hardcoded to L4 for every level's runs; a resize-mode mismatch that
read the L4 v29 champion at 0.02 mean rung instead of 8.17; a settle of 5 instead of 1
that read 4/40 instead of 32/40; box instead of sprite reach. A full game run needs four
policies set up correctly at once, so the settings must come from one place.

THREE SOURCES, each owning one kind of setting, never copied between them:
  * a LEVEL's settings (start state, map layout, game profile, fruit count, stop rules)
    live in ``experiments/003-yeti/registry.yaml`` under ``levels``;
  * a POLICY's view of the game (resize, resize mode, crop, frame settings, action mode,
    reach test) comes from the config it trained with, checked against the archived run
    record (tests/python/test_yeti_registry.py);
  * RESULTS come from the files ``eval_from_reset.py --policy`` writes, read back with
    the model hash checked, never from numbers typed into the registry.

No emulator or model is needed to load the registry, so its consistency tests run in CI.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from retro_ai.training.run_config import EnvConfig, RunConfig

REPO = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPO / "experiments/003-yeti/registry.yaml"
RESULTS_DIR = REPO / "experiments/003-yeti/data/champion_recheck"

# eval_from_reset.py flags whose value the registry owns. Passing any of them together
# with --policy is refused rather than silently overridden or silently ignored.
HAND_SETTINGS = (
    "model",
    "profile",
    "max_steps",
    "stall_threshold",
    "fruits_total",
    "start_state",
    "level",
    "resize_mode",
    "reach_mode",
)

# The reach test a run used when its record predates the `waypoint_reach_mode` field
# (added in c666500, 2026-09-08). Box was the only test before then. Reading such a
# config through today's RunConfig would report today's default, "sprite" (de21939),
# which is not what those runs trained with.
PRE_FLAG_REACH_MODE = "box"


@dataclass(frozen=True)
class LevelSpec:
    game_level: int
    layout: int
    start_state: Optional[str]
    profile: str
    fruits_total: int
    max_steps: int
    stall_threshold: int


@dataclass(frozen=True)
class PolicySpec:
    id: str
    model: str
    model_sha256: str
    config: str
    run_record: str


@dataclass(frozen=True)
class EvalSettings:
    """Everything an evaluation of ``policy`` on ``game_level`` needs, resolved."""

    policy: str
    game_level: int
    model: str
    model_sha256: str
    layout: int
    start_state: Optional[str]
    fruits_total: int
    reach_mode: str
    env: EnvConfig


@dataclass(frozen=True)
class Result:
    file: str
    policy: str
    game_level: int
    princess: int
    episodes: int
    seed: int
    model_sha256: str
    native_sha256: Optional[str]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Registry:
    def __init__(
        self,
        levels: Dict[int, LevelSpec],
        policies: Dict[str, PolicySpec],
        champions: Dict[int, str],
        repo: Path = REPO,
        results_dir: Path = RESULTS_DIR,
    ):
        self.levels = levels
        self.policies = policies
        self.champions = champions
        self.repo = Path(repo)
        self.results_dir = Path(results_dir)
        for lvl, pid in champions.items():
            if lvl not in levels:
                raise ValueError(f"champion for unknown game level {lvl}")
            if pid not in policies:
                raise ValueError(f"champion of level {lvl} is unknown policy {pid!r}")

    @classmethod
    def load(
        cls, path: Path = REGISTRY_PATH, results_dir: Path = RESULTS_DIR
    ) -> "Registry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        levels = {
            int(n): LevelSpec(
                game_level=int(n),
                layout=int(v["layout"]),
                start_state=v.get("start_state"),
                profile=str(v["profile"]),
                fruits_total=int(v["fruits_total"]),
                max_steps=int(v["max_steps"]),
                stall_threshold=int(v["stall_threshold"]),
            )
            for n, v in (raw.get("levels") or {}).items()
        }
        policies = {
            str(pid): PolicySpec(
                id=str(pid),
                model=str(v["model"]),
                model_sha256=str(v["model_sha256"]),
                config=str(v["config"]),
                run_record=str(v["run_record"]),
            )
            for pid, v in (raw.get("policies") or {}).items()
        }
        champions = {int(n): str(p) for n, p in (raw.get("champions") or {}).items()}
        return cls(levels, policies, champions, REPO, results_dir)

    # --- lookups ---------------------------------------------------------------

    def level(self, game_level: int) -> LevelSpec:
        try:
            return self.levels[int(game_level)]
        except KeyError:
            raise KeyError(
                f"game level {game_level} is not in the registry "
                f"(known: {sorted(self.levels)})"
            ) from None

    def policy(self, policy_id: str) -> PolicySpec:
        try:
            return self.policies[policy_id]
        except KeyError:
            raise KeyError(
                f"policy {policy_id!r} is not in the registry "
                f"(known: {sorted(self.policies)})"
            ) from None

    def config(self, policy_id: str) -> RunConfig:
        return RunConfig.from_yaml(str(self.repo / self.policy(policy_id).config))

    def run_record(self, policy_id: str) -> dict:
        path = self.repo / self.policy(policy_id).run_record
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def reach_mode(self, policy_id: str) -> str:
        """The reach test the policy's run actually used (see PRE_FLAG_REACH_MODE)."""
        cur = (self.run_record(policy_id).get("extras") or {}).get("curriculum") or {}
        return str(cur.get("waypoint_reach_mode", PRE_FLAG_REACH_MODE))

    def training_level(self, policy_id: str) -> int:
        """The game level whose start and layout the policy's config trains on."""
        cfg = self.config(policy_id)
        layout = int((cfg.reward.params or {}).get("level", 1))
        start = cfg.curriculum.start_state if cfg.curriculum else None
        hits = [
            n
            for n, spec in self.levels.items()
            if spec.layout == layout and spec.start_state == start
        ]
        if len(hits) != 1:
            raise ValueError(
                f"{policy_id}: config trains on layout {layout} from {start!r}, which "
                f"matches game levels {hits}; expected exactly one"
            )
        return hits[0]

    # --- evaluation --------------------------------------------------------------

    def eval_settings(self, policy_id: str, game_level: int) -> EvalSettings:
        """Resolve every setting an evaluation of ``policy_id`` on ``game_level`` needs.

        The level supplies profile and stop rules; the policy's config supplies the
        rest of the environment (resize, resize mode, crop, frame settings, action
        mode), so a policy is always shown the picture it trained on.
        """
        lvl = self.level(game_level)
        pol = self.policy(policy_id)
        env = dataclasses.replace(
            self.config(policy_id).env,
            profile=lvl.profile,
            max_steps=lvl.max_steps,
            stall_threshold=lvl.stall_threshold,
        )
        return EvalSettings(
            policy=policy_id,
            game_level=lvl.game_level,
            model=pol.model,
            model_sha256=pol.model_sha256,
            layout=lvl.layout,
            start_state=lvl.start_state,
            fruits_total=lvl.fruits_total,
            reach_mode=self.reach_mode(policy_id),
            env=env,
        )

    def default_result_path(
        self, policy_id: str, game_level: int, episodes: int, seed: int
    ) -> Path:
        return (
            self.results_dir
            / policy_id
            / f"level{int(game_level)}_{int(episodes)}ep_seed{int(seed)}.json"
        )

    # --- results -----------------------------------------------------------------

    def results(self, policy_id: str, game_level: Optional[int] = None) -> List[Result]:
        """Stochastic results for ``policy_id`` read from its result files.

        Only files the registry-driven evaluator wrote are accepted: they name the
        policy and game level and carry the model hash, which must match the
        registry's (a result for a different model is not this policy's result).
        """
        pol = self.policy(policy_id)
        out: List[Result] = []
        folder = self.results_dir / policy_id
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("policy") != policy_id or "game_level" not in d:
                continue
            if d.get("model_sha256") != pol.model_sha256 or d.get("deterministic"):
                continue
            if game_level is not None and int(d["game_level"]) != int(game_level):
                continue
            try:
                shown = str(path.relative_to(self.repo))
            except ValueError:
                shown = str(path)
            out.append(
                Result(
                    file=shown,
                    policy=policy_id,
                    game_level=int(d["game_level"]),
                    princess=int(d["princess_touches"]),
                    episodes=int(d["episodes"]),
                    seed=int(d["seed"]),
                    model_sha256=str(d["model_sha256"]),
                    native_sha256=d.get("native_sha256"),
                )
            )
        return out

    def rate(
        self, policy_id: str, game_level: int, native_sha256: Optional[str]
    ) -> Tuple[int, int]:
        """Pooled (princess, episodes) on ``game_level``, on one emulator build.

        Results from another emulator build are excluded: the L1 champion that read
        99.7% on the pre-``2b0a45d`` core reads 0/40 on the fixed one. Two files with
        the same seed are the same episodes, so a seed is counted once.
        """
        seen: Set[int] = set()
        k = n = 0
        for r in self.results(policy_id, game_level):
            if r.native_sha256 != native_sha256 or r.seed in seen:
                continue
            seen.add(r.seed)
            k += r.princess
            n += r.episodes
        return k, n


def explicit_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> Set[str]:
    """The ``dest`` of every option actually given on ``argv``.

    Parses ``argv`` a second time with every default replaced by a sentinel, so it is
    right for abbreviated options (``--resize`` for ``--resize-mode``), ``--opt=value``
    and flags. ``parser`` is modified; pass a freshly built one.
    """
    sentinel = object()
    parser.set_defaults(
        **{a.dest: sentinel for a in parser._actions if a.dest != "help"}
    )
    for a in parser._actions:
        a.required = False
    ns = parser.parse_args(list(argv))
    return {k for k, v in vars(ns).items() if v is not sentinel}


def hand_settings_given(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    owned: Iterable[str] = HAND_SETTINGS,
) -> List[str]:
    """Option strings on ``argv`` whose value the registry owns, sorted."""
    given = explicit_dests(parser, argv) & set(owned)
    flags = {a.dest: a.option_strings[0] for a in parser._actions if a.option_strings}
    return sorted(flags.get(d, d) for d in given)


def current_native_sha256() -> Optional[str]:
    """sha256 of the native emulator module this process would load, or None."""
    from retro_ai.training.run_manifest import _native_info

    return _native_info().get("sha256")


__all__ = [
    "REGISTRY_PATH",
    "RESULTS_DIR",
    "HAND_SETTINGS",
    "PRE_FLAG_REACH_MODE",
    "LevelSpec",
    "PolicySpec",
    "EvalSettings",
    "Result",
    "Registry",
    "sha256_of",
    "explicit_dests",
    "hand_settings_given",
    "current_native_sha256",
]
