"""Named reward formulas for training scripts.

Reward formulas used to live inline inside each training script's
``step()`` method. That made them invisible to the run manifest and
hard to diff across runs. This module gives each formula a stable
name + parameter schema, addressable from YAML configs:

.. code-block:: yaml

    reward:
      name: fruit_bonus
      params:
        scale: 0.01

Registered formulas
-------------------

- ``fruit_flat``          — +per_fruit reward for each fruit collected this step.
                            Matches the pre-Apr-2026 checkpoint-curriculum reward.
- ``fruit_bonus``         — +bonus * scale per fruit (faster = more reward).
                            Matches the post-Apr-2026 reward and segment training.
- ``fruit_princess_bonus`` — same as ``fruit_bonus`` plus a one-shot reward
                             when the princess is reached (level complete).
                             Detected by fruits_remaining repopulating from
                             0 to 4 while lives is preserved and bonus resets
                             upward — uses prev_bonus so a fast finish pays
                             more than a slow one.
- ``score_delta_survival`` — clipped score delta + constant per-step bonus.
                            Matches go_explore_phase2.py.

Adding a new formula
--------------------

Register a factory that returns a callable taking a :class:`RewardContext`:

.. code-block:: python

    @register("my_reward")
    def _my_reward(params):
        weight = params.get("weight", 1.0)
        def fn(ctx: RewardContext) -> float:
            return weight * ...
        return fn

Factories receive only the ``params`` dict from the config. Raise
``ValueError`` for bad params; prefer explicit defaults so omissions
have predictable meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping


@dataclass(frozen=True)
class RewardContext:
    """Per-step inputs made available to every reward function.

    Fields are intentionally broad so a single context suits all current
    and expected future formulas. Adding fields is backward-compatible;
    removing fields would break existing formulas, so treat this as
    append-only.
    """

    prev_fruits: int
    curr_fruits: int
    prev_bonus: int
    curr_bonus: int
    prev_score: int
    curr_score: int
    prev_lives: int
    curr_lives: int
    step_count: int


RewardFn = Callable[[RewardContext], float]
RewardFactory = Callable[[Mapping[str, Any]], RewardFn]


_REGISTRY: Dict[str, RewardFactory] = {}


def register(name: str) -> Callable[[RewardFactory], RewardFactory]:
    """Decorator to register a reward factory under ``name``.

    The factory receives the ``params`` dict from the YAML config and
    must return a ``RewardFn``.
    """

    def deco(factory: RewardFactory) -> RewardFactory:
        if name in _REGISTRY:
            raise ValueError(f"Reward formula {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def create(name: str, params: Mapping[str, Any] | None = None) -> RewardFn:
    """Instantiate a reward function by name.

    Parameters
    ----------
    name : str
        Registry key.
    params : mapping, optional
        Per-formula parameters (passed through to the factory).

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown reward formula: {name!r}. Available: {available}")
    return _REGISTRY[name](params or {})


def available() -> list[str]:
    """Return all registered formula names (sorted)."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in formulas
# ---------------------------------------------------------------------------


@register("fruit_flat")
def _fruit_flat(params: Mapping[str, Any]) -> RewardFn:
    """Flat reward per fruit collected this step.

    Parameters
    ----------
    per_fruit : float, default 10.0
        Reward added per fruit (multiplied by number collected on the
        step, though in practice that's always 0 or 1).
    """
    per_fruit = float(params.get("per_fruit", 10.0))

    def fn(ctx: RewardContext) -> float:
        if ctx.curr_fruits < ctx.prev_fruits:
            return (ctx.prev_fruits - ctx.curr_fruits) * per_fruit
        return 0.0

    return fn


@register("fruit_bonus")
def _fruit_bonus(params: Mapping[str, Any]) -> RewardFn:
    """Fruit reward scaled by the in-game bonus countdown.

    The Yeti bonus starts near 1000 and decreases every frame. Tying the
    reward to the bonus value means "collect fruit quickly" pays more
    than "collect fruit eventually".

    Parameters
    ----------
    scale : float, default 0.01
        Multiplier applied to the current bonus when a fruit is
        collected.
    """
    scale = float(params.get("scale", 0.01))

    def fn(ctx: RewardContext) -> float:
        if ctx.curr_fruits < ctx.prev_fruits:
            collected = ctx.prev_fruits - ctx.curr_fruits
            return collected * ctx.curr_bonus * scale
        return 0.0

    return fn


@register("fruit_princess_bonus")
def _fruit_princess_bonus(params: Mapping[str, Any]) -> RewardFn:
    """Fruit reward plus a one-shot princess reward on level complete.

    Same fruit term as ``fruit_bonus``. On a level-complete transition
    (princess reached), the game in the same frame:

    - Repopulates ``fruits_remaining`` from 0 back to 4
    - Resets the bonus countdown to ~1000
    - Preserves ``lives`` (the transition costs no life)

    Those three conditions together are taken to mean the princess was
    reached (a death-respawn also repopulates fruits but decrements
    lives). The princess term uses ``prev_bonus`` — the bonus value the
    agent earned *as* they touched the princess, before the game reset
    it — so a fast finish pays more than a slow one, same as the
    per-fruit term.

    Parameters
    ----------
    fruit_scale : float, default 0.01
        Per-fruit multiplier (matches ``fruit_bonus.scale``).
    princess_scale : float, default 0.05
        Per-princess multiplier. Default 0.05 makes a full-level
        princess (prev_bonus ≈ 500) worth ≈ 25 reward, comparable to
        the cumulative fruit rewards within one level (~20-30).
    """
    fruit_scale = float(params.get("fruit_scale", 0.01))
    princess_scale = float(params.get("princess_scale", 0.05))

    def fn(ctx: RewardContext) -> float:
        # Fruit pickup: fruits_remaining went down.
        if ctx.curr_fruits < ctx.prev_fruits:
            collected = ctx.prev_fruits - ctx.curr_fruits
            return collected * ctx.curr_bonus * fruit_scale
        # Princess / level complete: fruits repopulated, no life lost,
        # bonus jumped back up.
        if (
            ctx.curr_fruits > ctx.prev_fruits
            and ctx.curr_lives >= ctx.prev_lives
            and ctx.curr_bonus > ctx.prev_bonus
        ):
            return ctx.prev_bonus * princess_scale
        return 0.0

    return fn


@register("score_delta_survival")
def _score_delta_survival(params: Mapping[str, Any]) -> RewardFn:
    """Clipped score delta plus a constant per-step survival bonus.

    Used by Go-Explore Phase 2. The step bonus makes the agent prefer
    longer episodes; the score term rewards progress.

    Parameters
    ----------
    score_scale : float, default 0.1
    step_bonus  : float, default 0.01
    clip_min    : float, default 0.0
        Delta is clipped to ``>= clip_min`` before scaling. Prevents a
        negative reward from a mid-episode score reset (e.g. level up).
    """
    score_scale = float(params.get("score_scale", 0.1))
    step_bonus = float(params.get("step_bonus", 0.01))
    clip_min = float(params.get("clip_min", 0.0))

    def fn(ctx: RewardContext) -> float:
        delta = ctx.curr_score - ctx.prev_score
        if delta < clip_min:
            delta = clip_min
        return delta * score_scale + step_bonus

    return fn


__all__ = [
    "RewardContext",
    "RewardFn",
    "RewardFactory",
    "available",
    "create",
    "register",
]
