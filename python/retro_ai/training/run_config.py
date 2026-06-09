"""Typed training-run configuration.

Every training run is specified by a single YAML file. This module defines
the nested dataclass schema, loads YAML into it, and round-trips back to
a plain dict for manifest persistence.

Design principles
-----------------

1. **One YAML, no inheritance.** Every value either has a dataclass default
   or is required. Keeps resolution trivial; no mystery "this field came
   from somewhere else".

2. **Fail-loud on unknown keys.** A typo in a config should raise, not
   silently use the default. Helps catch stale configs after refactors.

3. **Round-trippable.** ``RunConfig.from_yaml(p).to_dict()`` yields the
   same structure the YAML describes (plus resolved defaults). That dict
   is what the run manifest persists, so replaying a run never needs the
   original YAML.

Typical use
-----------

>>> cfg = RunConfig.from_yaml("experiments/003-yeti/configs/segment_1to2.yaml")
>>> cfg.training.timesteps
5000000
>>> cfg.reward.name
'fruit_bonus'
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level run controls."""

    timesteps: int
    output: str
    num_envs: int = 8
    seed: Optional[int] = None
    resume: Optional[str] = None


@dataclass(frozen=True)
class EnvConfig:
    """Environment wiring.

    Most fields default to ``None`` meaning "use whatever the game profile
    specifies". That way a config only has to mention what it overrides.
    """

    profile: str
    action_mode: str = "joystick"
    max_steps: int = 1000
    stall_threshold: int = 15
    resize: Optional[Tuple[int, int]] = (84, 84)
    frame_skip: Optional[int] = None
    frame_stack: Optional[int] = None
    frame_maxpool: Optional[bool] = None
    grayscale: Optional[bool] = None


@dataclass(frozen=True)
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters.

    ``n_steps = None`` signals "compute from num_envs" downstream
    (typically ``max(1, 128 // num_envs)`` to keep rollout buffer size
    roughly constant).
    """

    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: Optional[int] = None
    n_epochs: int = 4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass(frozen=True)
class RewardConfig:
    """Reference to a named reward formula + its parameters."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumConfig:
    """Settings used only by ``train_checkpoint_curriculum.py``."""

    reset_fraction: float = 0.4
    frontier_fraction: float = 0.4
    earlier_fraction: float = 0.2
    max_states_per_checkpoint: int = 100
    min_states_to_advance: int = 20
    seed_archive: Optional[str] = None
    # Minimum number of frames the agent must survive *after* a
    # checkpoint snapshot, under its own policy, for that state to be
    # admitted to the seed pool. Replaces the old passive-noop probe
    # (state_validator) which rejected ~99% of real mid-action
    # pickups. A snapshot that led to the next checkpoint in the same
    # episode is always admitted regardless of this threshold.
    min_survival_frames: int = 30


@dataclass(frozen=True)
class SegmentConfig:
    """Settings used only by ``train_segment.py``."""

    checkpoints: str
    segment: int


@dataclass(frozen=True)
class BackwardCurriculumConfig:
    """Settings used only by ``go_explore_phase2.py``."""

    archive: str
    advance_threshold: float = 20.0
    advance_window: int = 100
    frontier_ratio: float = 0.5


@dataclass(frozen=True)
class RunConfig:
    """Fully-resolved training-run configuration.

    Only one of ``curriculum``, ``segment``, ``backward_curriculum`` should
    be populated per run — each script validates the one it needs is
    present (and that the others are absent).
    """

    training: TrainingConfig
    env: EnvConfig
    reward: RewardConfig
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: Optional[CurriculumConfig] = None
    segment: Optional[SegmentConfig] = None
    backward_curriculum: Optional[BackwardCurriculumConfig] = None

    # -- construction -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load RunConfig from YAML")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path!r} must be a YAML mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        kwargs: Dict[str, Any] = {}

        # Required sub-configs
        kwargs["training"] = _build(TrainingConfig, data.get("training"), "training")
        kwargs["env"] = _build(EnvConfig, data.get("env"), "env")
        kwargs["reward"] = _build(RewardConfig, data.get("reward"), "reward")

        # Optional / defaulted sub-configs
        if "ppo" in data:
            kwargs["ppo"] = _build(PPOConfig, data.get("ppo"), "ppo")

        # Script-specific — at most one should be present
        script_sections = [
            ("curriculum", CurriculumConfig),
            ("segment", SegmentConfig),
            ("backward_curriculum", BackwardCurriculumConfig),
        ]
        present = [name for name, _ in script_sections if name in data]
        if len(present) > 1:
            raise ValueError(
                f"Config has multiple script-specific sections: {present}. "
                "Only one of (curriculum, segment, backward_curriculum) is allowed."
            )
        for name, klass in script_sections:
            if name in data:
                kwargs[name] = _build(klass, data.get(name), name)

        # Reject unknown top-level keys.
        known_top_level = {
            "training",
            "env",
            "reward",
            "ppo",
            "curriculum",
            "segment",
            "backward_curriculum",
        }
        unknown = set(data) - known_top_level
        if unknown:
            raise ValueError(
                f"Unknown top-level config keys: {sorted(unknown)}. "
                f"Expected one or more of {sorted(known_top_level)}."
            )

        return cls(**kwargs)

    # -- persistence --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation, omitting ``None`` sub-configs."""
        result = dataclasses.asdict(self)
        for key in ("curriculum", "segment", "backward_curriculum"):
            if result.get(key) is None:
                result.pop(key, None)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(klass, payload, section_name: str):
    """Instantiate ``klass`` from a mapping, rejecting unknown keys."""
    if payload is None:
        raise ValueError(f"Config missing required section {section_name!r}")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Config section {section_name!r} must be a mapping, "
            f"got {type(payload).__name__}"
        )
    known = {f.name for f in fields(klass)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in {section_name!r}: {sorted(unknown)}. "
            f"Expected one or more of {sorted(known)}."
        )
    # Tuples need explicit coercion from YAML lists for the type-annotated
    # fields (``resize`` is the only one currently; keep this generic).
    coerced = {}
    for name, value in payload.items():
        annotation = klass.__dataclass_fields__[name].type
        coerced[name] = _coerce(value, annotation)
    return klass(**coerced)


def _coerce(value, annotation):
    """Best-effort coercion for a handful of types we expect from YAML."""
    # YAML represents tuples as lists. Only coerce when the annotation
    # names a tuple; otherwise return as-is and let the dataclass accept
    # whatever YAML loaded.
    anno_str = str(annotation)
    if value is not None and "Tuple" in anno_str and isinstance(value, list):
        return tuple(value)
    return value


__all__ = [
    "BackwardCurriculumConfig",
    "CurriculumConfig",
    "EnvConfig",
    "PPOConfig",
    "RewardConfig",
    "RunConfig",
    "SegmentConfig",
    "TrainingConfig",
]
