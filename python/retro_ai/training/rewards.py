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

    ``curr_y`` defaults to 0 when the caller has no positional reading
    to supply — rewards that don't read it are unaffected. Same for
    ``fruits_present`` (default empty tuple = "unknown, behave as if
    no per-fruit signal is available").
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
    curr_y: int = 0
    # Per-fruit presence at this step, tuple of bools for fruits 1..4.
    # Empty tuple means "not provided". When provided, ``True`` means
    # the fruit is still on the map, ``False`` means collected.
    fruits_present: tuple = ()


RewardFn = Callable[[RewardContext], float]
RewardFactory = Callable[[Mapping[str, Any]], RewardFn]


def reset_reward(fn: RewardFn) -> None:
    """Call ``fn.reset()`` if the reward function is stateful.

    Stateful formulas (e.g. floor-novelty) need to clear per-episode
    bookkeeping on reset. Stateless formulas don't define ``reset``
    and this is a no-op for them. Callers should invoke this on every
    episode boundary.
    """
    reset = getattr(fn, "reset", None)
    if callable(reset):
        reset()


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


@register("fruit_bonus_floor_novelty")
def _fruit_bonus_floor_novelty(params: Mapping[str, Any]) -> RewardFn:
    """Fruit reward plus a one-shot bonus per new floor visited per episode.

    Yeti's map has four floors, each 32 px tall and anchored at the
    bottom of the screen (y=182 is floor 1, y=150 floor 2, etc.). This
    formula pays a small bonus the first time the agent enters a floor
    it hasn't visited yet this episode.

    Motivation: approach 14 showed the per-segment CP2->CP3 policy
    almost never climbs past its starting floor (3-4% climb rate
    regardless of starting floor). Plain ``fruit_bonus`` only pays when
    the fruit is in reach; there's no gradient pointing toward
    climbing. The floor-novelty term provides a one-shot exploration
    incentive without rewarding in-place jumping (it fires on arrival
    at a new floor, not per-frame while standing there).

    One-shot per floor per episode keeps the signal from dominating
    fruit reward: visiting all 4 floors pays ``4 * novelty_bonus``
    (default = 4.0) vs a full-level fruit run of ~10-40 from
    ``fruit_bonus``.

    Uses internal per-episode state (``visited_floors``). Callers MUST
    invoke :func:`reset_reward` at episode boundaries — otherwise the
    visited set carries over between episodes.

    Parameters
    ----------
    scale : float, default 0.01
        Multiplier for the fruit term (same as ``fruit_bonus.scale``).
    novelty_bonus : float, default 1.0
        Reward paid on first visit to each new floor.
    """
    scale = float(params.get("scale", 0.01))
    novelty_bonus = float(params.get("novelty_bonus", 1.0))

    class _FloorNoveltyReward:
        def __init__(self) -> None:
            self.visited: set[int] = set()

        def reset(self) -> None:
            self.visited = set()

        def __call__(self, ctx: RewardContext) -> float:
            reward = 0.0
            if ctx.curr_fruits < ctx.prev_fruits:
                collected = ctx.prev_fruits - ctx.curr_fruits
                reward += collected * ctx.curr_bonus * scale
            # Floor bucket: 32px tall, anchored at bottom of 200px screen.
            # bucket 0 ~ floor 1 (spawn), bucket 1 ~ floor 2, etc.
            # Clamp bucket >= 4 (the death-animation region) so it
            # doesn't count as a "new floor visit".
            floor = (200 - int(ctx.curr_y)) // 32
            if 0 <= floor <= 3 and floor not in self.visited:
                self.visited.add(floor)
                reward += novelty_bonus
            return reward

    return _FloorNoveltyReward()


# Fruit (x, y) pixel-centre positions, measured from a CP0 state.
# Sprite is 16x16; these are the centres so distance math is
# consistent with the agent centre (ram_x*4 + 8, ram_y + 8).
# See debug/cp0_fruits_annotated.png for the verification overlay.
FRUIT_CENTRES_PX: dict[int, tuple[int, int]] = {
    1: (184, 184),
    2: (80, 150),
    3: (144, 120),
    4: (272, 88),
}


@register("fruit_bonus_climb_novelty")
def _fruit_bonus_climb_novelty(params: Mapping[str, Any]) -> RewardFn:
    """Fruit reward plus a one-shot bonus per floor climbed toward
    a remaining fruit that sits above the agent.

    Motivation: approach 15 showed ``fruit_bonus_floor_novelty`` helped
    descent but not climbing — agents starting at the spawn floor or
    game-floor-3 still failed to climb to a remaining fruit above.
    This variant gates the novelty bonus three ways:

    1. Direction: only awarded on a floor HIGHER than any floor
       previously visited this episode. Descending into a new-to-the-
       episode lower floor does not pay.
    2. Target: only awarded if at least one *remaining* fruit sits
       strictly above the agent's current pixel y. If every remaining
       fruit is on or below the agent's floor, climbing is away from
       all targets and shouldn't be pushed.
    3. One-shot: once credited, re-visits to the same or lower floor
       don't repay; only crossing to an even higher new floor pays
       again.

    This avoids past failures:
    - Plain delta(y): rewards jumping-in-place, wins by oscillation.
    - Milestone height thresholds: rewards repeat crossings at the
      same boundary.
    - Undirected novelty (approach 15): pays equally for up and down
      travel, helps descent but not climbing.

    Needs per-fruit presence via ``ctx.fruits_present``. If that's
    empty (unknown), falls back to the coarse "any remaining fruit
    means fruit could be above" signal — less safe but better than
    nothing.

    Parameters
    ----------
    scale : float, default 0.01
        Per-fruit multiplier (matches ``fruit_bonus.scale``).
    climb_bonus : float, default 2.0
        Reward on first arrival at each new HIGHER floor, when a
        remaining fruit is above. Spawn->top climb pays 3 * 2.0 = 6.0.
    """
    scale = float(params.get("scale", 0.01))
    climb_bonus = float(params.get("climb_bonus", 2.0))

    class _ClimbNoveltyReward:
        def __init__(self) -> None:
            # Highest floor-bucket visited this episode. Bucket 0 =
            # bottom of screen (spawn), 3 = top. "Climbing" means
            # moving to a higher bucket.
            self.best_floor: int | None = None

        def reset(self) -> None:
            self.best_floor = None

        @staticmethod
        def _fruit_above(ctx: RewardContext) -> bool:
            """True if at least one remaining fruit is strictly above
            the agent's current pixel y."""
            agent_pix_y = int(ctx.curr_y) + 8  # sprite is 16x16, use centre
            if ctx.fruits_present:
                for i, present in enumerate(ctx.fruits_present, start=1):
                    if not present:
                        continue
                    fy = FRUIT_CENTRES_PX.get(i, (0, 0))[1]
                    if fy < agent_pix_y:
                        return True
                return False
            # Fallback: no per-fruit info — if any fruit remains, pay.
            return ctx.curr_fruits > 0

        def __call__(self, ctx: RewardContext) -> float:
            reward = 0.0
            if ctx.curr_fruits < ctx.prev_fruits:
                collected = ctx.prev_fruits - ctx.curr_fruits
                reward += collected * ctx.curr_bonus * scale

            floor = (200 - int(ctx.curr_y)) // 32
            if not (0 <= floor <= 3):
                return reward
            if self.best_floor is None:
                self.best_floor = floor
                return reward
            # Higher bucket number = higher on screen = climbing.
            if floor <= self.best_floor:
                return reward
            self.best_floor = floor
            if self._fruit_above(ctx):
                reward += climb_bonus
            return reward

    return _ClimbNoveltyReward()


__all__ = [
    "RewardContext",
    "RewardFn",
    "RewardFactory",
    "available",
    "create",
    "register",
    "reset_reward",
]
