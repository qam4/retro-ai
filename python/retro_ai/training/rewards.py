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
                             Fires when the caller flags
                             ``ctx.princess_touched`` — uses prev_bonus
                             so a fast finish pays more than a slow one.
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

from retro_ai.training.targets import reaches


@dataclass(frozen=True)
class RewardContext:
    """Per-step inputs made available to every reward function.

    Fields are intentionally broad so a single context suits all current
    and expected future formulas. Adding fields is backward-compatible;
    removing fields would break existing formulas, so treat this as
    append-only.

    ``curr_y`` / ``curr_x`` default to 0 when the caller has no
    positional reading to supply — rewards that don't read them are
    unaffected. Same for ``fruits_present`` (default empty tuple =
    "unknown, behave as if no per-fruit signal is available").
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
    curr_x: int = 0
    # Per-fruit presence at this step, tuple of bools for fruits 1..4.
    # Empty tuple means "not provided". When provided, ``True`` means
    # the fruit is still on the map, ``False`` means collected.
    fruits_present: tuple = ()
    # True on the single step where the agent just touched the
    # princess (rising edge of the level-cleared flag in RAM). The
    # caller is responsible for the rising-edge check; reward
    # functions can treat this as authoritative.
    princess_touched: bool = False
    # Player sprite-pose index (MO5 Yeti RAM 0x2B54). -1 = not provided.
    # Pose-gated rewards freeze shaping while this is an airborne/falling
    # code; ungated rewards ignore it. See experiments/003-yeti-training.md
    # "run 3" for the measured pose table.
    pose: int = -1
    # True on the step where the player has just died (cause-agnostic MO5
    # Yeti death flag 0x2AFC == 65; lives byte is inert on level 2). Default
    # False = "not provided / alive". Used by the grounded reward to suppress
    # shaping credit on a fatal transition (so a fall that ends in death is
    # never credited even if it briefly grounds). Reward functions that don't
    # read it are unaffected.
    died: bool = False


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


def restore_reached_waypoints(fn: RewardFn, idents) -> None:
    """Mark mandatory-waypoint MILESTONES as already reached on a seeded start.

    Why this exists: milestone "reached" state is POSITIONAL — it is computed by
    walking within a tol-box — so unlike CP progress (fruit presence / princess
    flag, which live in the emulator RAM and therefore survive a ``load_state``)
    it lives only in the reward object and is wiped by the per-episode reset. An
    episode seeded high on the route then treats every milestone BELOW it as
    still pending, so the potential sums path-distance to targets BEHIND the
    agent and the shaping pays it to RETREAT. Measured on L3: the milestone-sum
    was globally minimised at SN3, so seeds above SN3 were paid ~+3.7 to climb
    back down and a seed AT SN3 lost reward for leaving — which is why the
    SN3->A1 hand-off measured 0%.

    ``idents`` is the set of route-point ids the seed had already reached when it
    was captured (accumulated transitively along the reverse-curriculum chain).
    No-op for rewards without milestones (all L1/L2 formulas), so behaviour
    there is unchanged.
    """
    restore = getattr(fn, "restore_reached_waypoints", None)
    if callable(restore) and idents:
        restore(idents)


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

    Same fruit term as ``fruit_bonus``. The princess term fires when
    the caller flags ``ctx.princess_touched=True`` — typically the
    rising edge of the Yeti level-cleared RAM flag (byte 11050). See
    ``scripts/train_segment.py`` for the canonical detection.

    The princess term uses ``prev_bonus`` — the bonus value the agent
    earned *as* they touched the princess, before the game zeroes it
    on level transition — so a fast finish pays more than a slow one,
    same as the per-fruit term.

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
        # Princess / level complete: caller flagged the rising edge.
        if ctx.princess_touched:
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


@register("fruit_bonus_path_progress")
def _fruit_bonus_path_progress(params: Mapping[str, Any]) -> RewardFn:
    """Fruit reward plus shortest-path progress toward any remaining fruit.

    Uses the hand-coded Yeti navigation graph
    (:mod:`retro_ai.training.yeti_map`) to compute path distance in
    pixels along walkable floor segments and ladder climbs.

    Per step, the reward:
      1. Resolves the agent's current floor from pixel y (with
         tolerance). Falls back to the last-known floor if the agent
         is mid-jump; if unknown, skips the progress term.
      2. For **every remaining fruit**, computes the shortest-path
         distance from the agent through the graph.
      3. If that distance is strictly smaller than the per-fruit best
         seen this episode so far, pays ``(best - new) * scale`` and
         updates best. A pickup clears that fruit's entry (it's
         collected, no longer a target).
      4. A fruit pickup also fires the usual fruit-bonus reward.

    Tracking progress per-fruit rather than per-closest-only means
    the agent gets shaping no matter which remaining fruit it decides
    to head toward. The "best_d per fruit" bookkeeping still prevents
    ratcheting / oscillation: once the agent has been within distance
    D of fruit F, it can't re-earn reward for reaching distance D
    again — only for getting strictly closer.

    Parameters
    ----------
    scale : float, default 0.01
        Per-pixel multiplier on path-distance progress. A full spawn
        -> F4 route is 496 px; at scale=0.01 that pays 4.96 over the
        entire climb, less than a single fruit pickup (~6-10) but
        enough to hint direction.
    fruit_scale : float, default 0.01
        Multiplier for the fruit pickup term (matches fruit_bonus).

    Notes
    -----
    - When there are no remaining fruits, the progress term is 0
      (this reward doesn't shape for the princess yet).
    - Jumping briefly changes the agent's y but pixel x is unchanged,
      so path distance doesn't drop — zero reward for jumps.
    - A ladder climb changes the floor, which drops path distance to
      fruits on the new floor and beyond — so ladder travel pays.
    """
    progress_scale = float(params.get("scale", 0.01))
    fruit_scale = float(params.get("fruit_scale", params.get("scale", 0.01)))

    from retro_ai.training.yeti_map import (
        agent_floor_from_pixel_y,
        build_navigation_map,
    )

    nav = build_navigation_map()

    class _PathProgressReward:
        def __init__(self) -> None:
            # Per-fruit best distance seen this episode. None = not
            # initialised yet. The first time we see a fruit's
            # distance we store it as the baseline and pay nothing;
            # subsequent strictly smaller distances pay
            # ``(prev_best - new) * scale``.
            self.best_d: dict[int, int | None] = {}
            self.last_floor: int | None = None

        def reset(self) -> None:
            self.best_d = {}
            self.last_floor = None

        def __call__(self, ctx: RewardContext) -> float:
            reward = 0.0
            if ctx.curr_fruits < ctx.prev_fruits:
                collected = ctx.prev_fruits - ctx.curr_fruits
                reward += collected * ctx.curr_bonus * fruit_scale
                # Re-baseline every remaining fruit's best_d at the new
                # post-pickup position. Without this, a fruit's best_d
                # is the closest the agent ever drifted to it across the
                # WHOLE episode (e.g. passing the L23 ladder en route to
                # F2), so the next leg toward it starts already "spent"
                # and pays nothing for the first stretch — a dead zone
                # exactly where the agent must commit to a long
                # traversal. Resetting on pickup gives each inter-fruit
                # leg a fresh full-distance budget.
                self.best_d = {}
            pixel_y = int(ctx.curr_y)
            floor = agent_floor_from_pixel_y(pixel_y)
            if floor is None:
                floor = self.last_floor
            else:
                self.last_floor = floor
            if floor is None:
                return reward

            if not ctx.fruits_present:
                return reward

            # Clear tracking for fruits that have been collected, so a
            # later princess-touch (which re-populates fruits for the
            # next level) doesn't get charged stale best_d values.
            for fid, is_present in enumerate(ctx.fruits_present, start=1):
                if not is_present:
                    self.best_d[fid] = None

            agent_pix_x = int(ctx.curr_x) * 4 + 8

            # Accumulate progress across every remaining fruit.
            # Each fruit has its own best-seen lock, so oscillation
            # between two targets ratchets each fruit's best_d down
            # and then pays nothing further.
            for fid, is_present in enumerate(ctx.fruits_present, start=1):
                if not is_present:
                    continue
                d = nav.path_distance_from_agent(floor, agent_pix_x, f"F{fid}")
                prev_best = self.best_d.get(fid)
                if prev_best is None:
                    self.best_d[fid] = d
                    continue
                if d < prev_best:
                    reward += (prev_best - d) * progress_scale
                    self.best_d[fid] = d
            return reward

    return _PathProgressReward()


@register("fruit_bonus_path_progress_universal")
def _fruit_bonus_path_progress_universal(params: Mapping[str, Any]) -> RewardFn:
    """Path-progress reward that handles both fruit-collection AND
    princess-touch segments uniformly.

    When some fruits remain, behaves identically to
    ``fruit_bonus_path_progress`` (per-fruit best_d ratchet, pickup
    term, fall-back-on-last_floor while mid-jump).

    When ALL fruits are collected (CP4), targets the princess node
    in the navigation graph instead. Pays best_d ratchet toward
    princess. The princess term fires when the caller flags
    ``ctx.princess_touched=True`` — typically the rising edge of the
    Yeti level-cleared RAM flag (byte 11050). When that fires, this
    pays ``prev_bonus * princess_scale`` and resets all best_d
    entries (the game just reloaded the level).

    Parameters
    ----------
    scale : float, default 0.01
        Per-pixel multiplier on path-distance progress (both fruits
        and princess use the same scale).
    fruit_scale : float, default 0.01
        Per-fruit multiplier on the pickup term.
    princess_scale : float, default 0.05
        Per-princess multiplier on the touch term. Default 0.05
        makes a fast princess-touch worth ≈ 25 reward
        (prev_bonus=500 × 0.05), comparable to the cumulative
        fruits in one level.
    """
    progress_scale = float(params.get("scale", 0.01))
    fruit_scale = float(params.get("fruit_scale", params.get("scale", 0.01)))
    princess_scale = float(params.get("princess_scale", 0.05))

    from retro_ai.training.yeti_map import (
        agent_floor_from_pixel_y,
        build_navigation_map,
    )

    nav = build_navigation_map()

    class _UniversalPathProgressReward:
        def __init__(self) -> None:
            self.best_d: dict[int, int | None] = {}
            self.best_d_princess: int | None = None
            self.last_floor: int | None = None

        def reset(self) -> None:
            self.best_d = {}
            self.best_d_princess = None
            self.last_floor = None

        def __call__(self, ctx: RewardContext) -> float:
            reward = 0.0

            # Fruit pickup term.
            if ctx.curr_fruits < ctx.prev_fruits:
                collected = ctx.prev_fruits - ctx.curr_fruits
                reward += collected * ctx.curr_bonus * fruit_scale
                # Re-baseline all remaining targets (fruits + princess)
                # at the new post-pickup position, so the next leg gets a
                # fresh full-distance progress budget rather than
                # inheriting a leaked-low best_d from earlier travel
                # (e.g. passing the L23 ladder while heading to F2). See
                # the F2->F3 reward audit (approach 33).
                self.best_d = {}
                self.best_d_princess = None

            # Princess touch: caller flagged the rising edge of the
            # level-cleared flag. Pay one-shot reward and reset all
            # best_d trackers (game is about to reload the level with
            # fruits=4 and bonus=1000).
            if ctx.princess_touched:
                reward += ctx.prev_bonus * princess_scale
                self.best_d = {}
                self.best_d_princess = None

            # Resolve agent floor.
            pixel_y = int(ctx.curr_y)
            floor = agent_floor_from_pixel_y(pixel_y)
            if floor is None:
                floor = self.last_floor
            else:
                self.last_floor = floor
            if floor is None:
                return reward

            agent_pix_x = int(ctx.curr_x) * 4 + 8

            # Decide target: any remaining fruit, OR princess if none.
            any_fruit = bool(ctx.fruits_present) and any(ctx.fruits_present)
            if any_fruit:
                # Clear best_d for fruits no longer present (post-pickup).
                for fid, is_present in enumerate(ctx.fruits_present, start=1):
                    if not is_present:
                        self.best_d[fid] = None
                # Per-fruit ratchet.
                for fid, is_present in enumerate(ctx.fruits_present, start=1):
                    if not is_present:
                        continue
                    d = nav.path_distance_from_agent(floor, agent_pix_x, f"F{fid}")
                    prev_best = self.best_d.get(fid)
                    if prev_best is None:
                        self.best_d[fid] = d
                        continue
                    if d < prev_best:
                        reward += (prev_best - d) * progress_scale
                        self.best_d[fid] = d
            else:
                # No fruits: target princess.
                d = nav.path_distance_from_agent(floor, agent_pix_x, "princess")
                prev_best = self.best_d_princess
                if prev_best is None:
                    self.best_d_princess = d
                elif d < prev_best:
                    reward += (prev_best - d) * progress_scale
                    self.best_d_princess = d

            return reward

    return _UniversalPathProgressReward()


@register("fruit_bonus_path_progress_pbrs")
def _fruit_bonus_path_progress_pbrs(params: Mapping[str, Any]) -> RewardFn:
    """Potential-based reward shaping (PBRS) variant of the path-progress
    reward — the Markovian replacement for the ``best_d`` ratchet.

    Motivation (approach 34). The ``..._universal`` reward shapes via a
    per-fruit ``best_d`` ratchet: it pays only when the agent beats its
    *closest distance ever* to a fruit this episode. That makes the
    reward **non-Markovian** — the reward at a given state depends on
    episode history (the best distance seen so far), which the policy
    can't observe. PBRS gives the *same* anti-oscillation property
    without the history dependence.

    Shaping term per step::

        F(s, s') = gamma * Phi(s') - Phi(s),   Phi(s) = -scale * sum_f D_f(s)

    where ``D_f`` is the nav-graph path distance to remaining fruit ``f``
    (princess when none remain). Moving closer pays ``+``; moving away
    pays a symmetric ``-`` so round-trips cancel (no farming) with no
    ratchet. With ``gamma`` equal to the agent's discount, the optimal
    policy is provably unchanged (Ng, Harada & Russell 1999).

    The sparse fruit-pickup and princess terms are identical to
    ``..._universal``.

    Known, deliberately-retained approximations (see backlog):
    - ``last_floor`` fallback while mid-jump/ladder. This is a *small,
      bounded* non-Markovian residue (current-floor inferred with one
      step of memory because pixel-y alone is ambiguous). It is what
      keeps jumps from being rewarded (floor pinned + x-based distance
      => a jump is a no-op in the graph), so it is kept on purpose. A
      future change may derive current-floor from RAM (ladder/velocity
      flag) to remove it cleanly.
    - Phi uses the SUM over all remaining fruits (matching the legacy
      target), not nearest/next-fruit. Target selection is a separate
      future change.

    Parameters
    ----------
    scale : float, default 0.01
        Potential scale (per-pixel). Phi = -scale * sum of distances.
    fruit_scale : float, default = scale
        Sparse per-fruit pickup multiplier.
    princess_scale : float, default 0.05
        Sparse princess-touch multiplier.
    gamma : float, default 0.99
        Shaping discount. Should equal the agent's PPO gamma for the
        policy-invariance guarantee; the training script injects
        ``cfg.ppo.gamma`` here by default so they can't drift.
    """
    progress_scale = float(params.get("scale", 0.01))
    fruit_scale = float(params.get("fruit_scale", params.get("scale", 0.01)))
    princess_scale = float(params.get("princess_scale", 0.05))
    gamma = float(params.get("gamma", 0.99))
    level = int(params.get("level", 1))
    # Segment-aware shaping (default OFF => byte-identical to the shipped
    # reward). When ON, a point on a VERTICAL ladder/escalator edge resolves to
    # that edge and earns continuous path-progress toward the target as it
    # descends/climbs, instead of the floor-quantized "credit on arrival". This
    # is what un-flattens the L3 escalator ride (see yeti_map segment helpers).
    segment_shaping = bool(params.get("ladder_segment_shaping", False))
    _UNREACHABLE = 10**8  # path_distance INF sentinel is 10^9; treat >= as INF

    from retro_ai.training.yeti_map import (
        agent_floor_from_pixel_xy,
        agent_ladder_from_pixel_xy,
        build_navigation_map,
        get_level_map,
    )

    nav = build_navigation_map(level)

    # ROUTE POTENTIAL (opt-in). Phi = -scale * shortest remaining travel that
    # collects every outstanding fruit and ends at the princess, instead of a SUM
    # of independent distances to each outstanding target.
    #
    # Why. A sum cannot express "do A, then B" when A and B lie in opposite
    # directions -- the terms fight. L1-L3 hid that because their targets are
    # roughly co-directional; L4's route doubles back (fruit far right, princess
    # far left) and the sum paid the agent to walk AWAY from the mandatory fruit:
    # 0 fruits collected in 3000 reset-origin episodes.
    #
    # The fix is that the princess leg becomes d(fruit -> princess), which does not
    # move as the agent walks, rather than d(agent -> princess), which does.
    # Measured on L3's A1..A5 ascent (the stretch v11's milestones were added for),
    # per-step slope over 20 samples: route 19 down / 0 flat / 1 up, versus the
    # milestone sum's 16 / 1 / 3. It is also CONTINUOUS across a fruit pickup
    # (route -16 where the sum jumps +48), because the fruit->princess leg was
    # already counted -- so it needs no rebaseline there.
    #
    # NOTE the per-step magnitude is ~3.5x smaller than the milestone sum's, so
    # ``scale`` must be raised to keep the shaping-to-sparse balance.
    route_potential = bool(params.get("route_potential", False))
    _fruit_ids = sorted(get_level_map(level).fruit_centre_px) if route_potential else []
    # _suffix[(remaining_set, first)] = distance from `first`, through the rest of
    # the set in the cheapest order, ending at the princess. Position-independent,
    # so it is precomputed once; at most 4 fruits, so brute force is fine.
    _suffix: Dict[Any, float] = {}
    if route_potential:
        import itertools as _it

        _ni = nav.node_by_ident
        _pn = _ni.get("princess")
        for _r in range(1, len(_fruit_ids) + 1):
            for _sub in _it.combinations(_fruit_ids, _r):
                _S = frozenset(_sub)
                for _first in _sub:
                    _rest = [i for i in _sub if i != _first]
                    _best = None
                    for _order in _it.permutations(_rest):
                        _seq = [_first, *_order]
                        _tot = 0.0
                        _ok = True
                        for _a, _b in zip(_seq, _seq[1:]):
                            _ia, _ib = _ni.get(f"F{_a}"), _ni.get(f"F{_b}")
                            if _ia is None or _ib is None:
                                _ok = False
                                break
                            _tot += nav.dist[_ia][_ib]
                        _il = _ni.get(f"F{_seq[-1]}")
                        if not _ok or _il is None or _pn is None:
                            continue
                        _tot += nav.dist[_il][_pn]
                        _best = _tot if _best is None else min(_best, _tot)
                    if _best is not None:
                        _suffix[(_S, _first)] = _best

    class _PBRSPathProgressReward:
        def __init__(self) -> None:
            self.last_floor: int | None = None
            self.prev_phi: float | None = None
            # (floor, ladder, agent_x, pixel_y) resolved on the last _potential
            # call, so the grounded wrapper's waypoint-sum reuses the SAME
            # segment resolution. ladder = (name, y_top, y_bot) or None.
            self._seg = (None, None, 0, 0)

        def reset(self) -> None:
            self.last_floor = None
            self.prev_phi = None
            self._seg = (None, None, 0, 0)

        def _potential(self, ctx: RewardContext) -> float | None:
            """Phi(s) = -scale * sum of path distances to remaining
            targets, or None if the position can't be resolved."""
            agent_pix_x = int(ctx.curr_x) * 4 + 8
            curr_y = int(ctx.curr_y)
            floor = agent_floor_from_pixel_xy(agent_pix_x, curr_y, level)
            ladder = None
            if floor is not None:
                self.last_floor = floor
            elif segment_shaping:
                # Off a platform but possibly on a vertical edge (ladder /
                # escalator): resolve it so the descent earns progress. Only
                # fall back to last_floor when not on any edge.
                ladder = agent_ladder_from_pixel_xy(agent_pix_x, curr_y, level)
                if ladder is None:
                    floor = self.last_floor
            else:
                floor = self.last_floor
            self._seg = (floor, ladder, agent_pix_x, curr_y)
            if floor is None and ladder is None:
                return None
            # Drop targets the graph can't reach (>= sentinel), exactly as the
            # grounded wrapper's waypoint-sum does. On L1/L2 every fruit/princess
            # is graph-reachable so nothing is ever dropped -> byte-identical;
            # on L3 this removes the dead ~1e9 term for the fruit that sits past
            # the un-modelled A1-A5 ascent (see TODO "Yeti Level 3"), which was
            # a constant offset that only made phi unreadable.
            any_fruit = bool(ctx.fruits_present) and any(ctx.fruits_present)
            total = 0
            if route_potential:
                # One tour: to the cheapest first fruit, then through the rest,
                # ending at the princess. The tail is position-independent, so
                # walking toward a fruit always shortens the whole thing.
                left = frozenset(
                    fid
                    for fid, present in enumerate(ctx.fruits_present, start=1)
                    if present
                )
                if not left:
                    d = nav.path_distance_from_pos(
                        floor, ladder, agent_pix_x, curr_y, "princess"
                    )
                    total = d if d < _UNREACHABLE else 0
                else:
                    best = None
                    for first in left:
                        d0 = nav.path_distance_from_pos(
                            floor, ladder, agent_pix_x, curr_y, f"F{first}"
                        )
                        if d0 >= _UNREACHABLE:
                            continue
                        tail = _suffix.get((left, first))
                        if tail is None or tail >= _UNREACHABLE:
                            continue
                        cand = d0 + tail
                        best = cand if best is None else min(best, cand)
                    total = best if best is not None else 0
                return -progress_scale * total
            if any_fruit:
                for fid, present in enumerate(ctx.fruits_present, start=1):
                    if present:
                        d = nav.path_distance_from_pos(
                            floor, ladder, agent_pix_x, curr_y, f"F{fid}"
                        )
                        if d < _UNREACHABLE:
                            total += d
            else:
                d = nav.path_distance_from_pos(
                    floor, ladder, agent_pix_x, curr_y, "princess"
                )
                if d < _UNREACHABLE:
                    total += d
            return -progress_scale * total

        def __call__(self, ctx: RewardContext) -> float:
            reward = 0.0

            picked = ctx.curr_fruits < ctx.prev_fruits
            if picked:
                collected = ctx.prev_fruits - ctx.curr_fruits
                reward += collected * ctx.curr_bonus * fruit_scale
            if ctx.princess_touched:
                reward += ctx.prev_bonus * princess_scale

            phi = self._potential(ctx)

            # Re-baseline (no shaping this step) across any discontinuity:
            # episode start (prev None), unresolved floor (phi None), a
            # fruit pickup or a princess touch (the set of remaining
            # targets — and on princess, the whole level — changes). The
            # sparse terms cover those events; shaping resumes next step.
            if picked or ctx.princess_touched or self.prev_phi is None or phi is None:
                self.prev_phi = phi
                return reward

            reward += gamma * phi - self.prev_phi
            self.prev_phi = phi
            return reward

    return _PBRSPathProgressReward()


# Player sprite-pose codes (0x2B54) that mean "on a surface" (grounded floor
# or ladder) and are therefore creditable for path-progress shaping. Every
# other code (jump 9/10, fall 11, death-anim 12, and any unseen code) is
# treated as airborne/off-surface -> shaping frozen (fails safe). Measured
# table: experiments/003-yeti-training.md "run 3".
SURFACE_POSES = frozenset({0, 1, 2, 3, 4, 5, 8})

# Poses at which a waypoint may NOT be marked reached: 11 = fall, 12 = death anim.
# Falling PAST a waypoint is not reaching it. Everything else counts, INCLUDING the
# jump poses 9/10 -- see the marking block for why that matters.
#
# Kept as a local constant rather than importing yeti (this module stays game-agnostic,
# same as SURFACE_POSES above). test_reward_mark_blocklist_matches_yeti pins it equal to
# yeti.NON_TRAVERSAL_POSES so the two cannot drift.
MARK_BLOCKED_POSES = frozenset({11, 12})


@register("fruit_bonus_path_progress_pbrs_grounded")
def _fruit_bonus_path_progress_pbrs_grounded(params: Mapping[str, Any]) -> RewardFn:
    """Pose-gated variant of ``fruit_bonus_path_progress_pbrs``.

    Identical to the PBRS path-progress reward EXCEPT shaping is frozen
    (potential -> None -> re-baseline, no credit) whenever the player sprite
    pose (``ctx.pose``, RAM 0x2B54) is not a surface code. This stops the
    reward from crediting a *fall* into a lower floor (the level-2 failure
    mode: falling out-pays crossing at every one of ~14 gaps) while still
    crediting ladder descents (pose 8) and jump-landings (grounded on
    arrival). ``last_floor`` is preserved (deliberately not removed).

    Reuses the ungated reward object verbatim and only overrides its
    ``_potential`` with a pose guard, so the base reward's logic is
    untouched. Only active when a pose is supplied (``ctx.pose >= 0``); with
    no pose it behaves exactly like the ungated reward.
    """
    base = _fruit_bonus_path_progress_pbrs(params)
    gamma = float(params.get("gamma", 0.99))
    fruit_scale = float(params.get("fruit_scale", params.get("scale", 0.01)))
    princess_scale = float(params.get("princess_scale", 0.05))
    # (D4) When True, DEFER the sparse fruit credit to the next grounded-alive
    # frame (mirroring the D2 shaping deferral) instead of paying it at the
    # pickup step. A fruit grabbed mid-air that never lands alive (the L2
    # fruit-2 grab-and-fall-to-death) is then never rewarded, so the fatal
    # early jump stops being locally optimal. Default False = pay at pickup
    # (byte-identical to the shipped reward; L1 unaffected).
    defer_fruit = bool(params.get("defer_fruit_credit", False))
    # (D5) One rule for every target type: credit only for progress you SURVIVE.
    # Implies the fruit deferral (a fruit grabbed mid-fatal-fall must not pay
    # either), so enabling this turns that on regardless of the legacy flag —
    # they were the same idea applied to only one target type.
    credit_requires_survival = bool(params.get("credit_requires_survival", False))
    if credit_requires_survival:
        defer_fruit = True

    # Mandatory-waypoint reward targets (LevelMap.reward_waypoints): the
    # reward sums path-distance to these exactly like it sums distance to
    # remaining fruits (unordered), min over an OR-group's members, dropping
    # any that are currently unreachable (10^9 sentinel). Reaching any member
    # of a group (within tol) marks it done. Empty on L1/L2 -> every WP branch
    # below is skipped, so behavior is byte-identical to the shipped reward.
    from retro_ai.training.yeti_map import build_navigation_map, get_level_map

    progress_scale = float(params.get("scale", 0.01))
    level = int(params.get("level", 1))
    wp_tol = int(params.get("waypoint_reward_tol", 2))
    # Geometry of the reach test; forced to match the curriculum by the trainer.
    # See CurriculumConfig.waypoint_reach_mode.
    wp_reach_mode = str(params.get("waypoint_reach_mode", "sprite"))
    # Where milestone marking happens. See the marking block in __call__. Default True
    # (mark while airborne, so a jump landing counts). False reproduces the placement
    # from before 2026-09-14, and exists so the A/B's control arm is a config rather
    # than a git checkout.
    mark_airborne = bool(params.get("mark_airborne", True))
    # (D6) PRICE A FRAME AGAINST THE LIST IT STARTED WITH.
    #
    # The waypoint term is a sum of distances to the groups still on the list. When a
    # group is marked reached it leaves the list, so the sum drops for a reason that is
    # not movement, and paying that drop would pay for bookkeeping. Today's guard is to
    # skip the frame entirely (the ``active_wp != _prev_active_wp`` arm of the D3 `if`).
    #
    # WHY THAT GUARD IS NOT FREE. The D2 freeze banks a whole jump into the landing
    # frame: shaping is suppressed while airborne and ``prev_phi`` is held, so the
    # landing pays the entire distance covered since take-off. If the waypoint is marked
    # mid-air, the code cannot notice the list change until the next grounded frame --
    # which IS the landing -- so the skip lands on the most valuable frame of the
    # episode. Measured on L4 climb2 -> Spring (v16a policy, 29/29 crossings, replaying
    # one trajectory through both variants): the floor-9 arrival pays +3.200 with
    # marking on the ground and exactly 0.000 with marking in the air. The f7 `Rope1`
    # arrival loses 5.120 the same way. That is what sank v16b (`Spring` mean reach
    # 0.63 -> 0.03) -- not the marking itself, but the skip landing on the landing.
    #
    # THE FIX. Split the frame in two: pay it against the set it STARTED with
    # (``_prev_active_wp``, priced at the CURRENT position, so only movement is
    # charged), then rebaseline to the set it ENDS with. Nothing is paid for a deletion
    # and nothing earned is lost, so marking position stops mattering. When the list did
    # not change the two sets are equal and this is arithmetically a no-op.
    #
    # SCOPE. Only the waypoint arm. The other arms of that `if` (death, fruit pickup,
    # deferred-fruit credit, princess) still skip: pricing the old set there means
    # reconstructing which fruits were uncollected, which is a bigger change with no
    # measured failure behind it. Default False = today's skip, byte-identical.
    pay_on_target_change = bool(params.get("pay_on_target_change", False))
    segment_shaping = bool(params.get("ladder_segment_shaping", False))
    # When segment shaping is on, the escalator RIDE pose (13) is a controlled
    # vertical traversal, not a fall, so it counts as on-surface (un-frozen) so
    # the descent can be shaped. Off => shipped surface set (byte-identical).
    _surf = (SURFACE_POSES | {13}) if segment_shaping else SURFACE_POSES
    _WP_UNREACHABLE = 10**8  # path_distance sentinel is 10^9; treat >= as unreachable
    _lvl_map = get_level_map(level)
    _wp_nav = build_navigation_map(level)
    # _wp_groups: list of OR-groups; each = list of (ident, x_ram, y).
    # TWO INDEPENDENT LEVERS. ``route_potential`` changes the SHAPE of the base
    # potential (remaining route length instead of a sum of distances);
    # ``drop_milestones`` removes the mandatory-waypoint term. They were briefly
    # coupled -- route_potential implied drop_milestones -- and control D therefore
    # moved both at once and its result was unattributable. Same mistake v14 made.
    # Keep them separate so each can be measured alone.
    _drop_milestones = bool(params.get("drop_milestones", False))
    _wp_groups: list = []
    # MARKING POSITION comes from the shared Target, not from the graph node.
    #
    # These two used to be the same expression, and where they still agree this is a
    # no-op (verified: Target.pos matched the node-derived position for every target on
    # L3 and L4 before any override existed). It matters where a level supplies a
    # MEASURED anchor via `LevelMap.jump_waypoint_pos`, because a node is placed on a
    # platform EDGE while the agent lands 3-4 units inside -- which is how L4 `Fr1`'s
    # tol-2 box ended up somewhere the agent never stands, so the milestone was never
    # marked and its distance term never switched off.
    #
    # NOTE this changes only WHERE WE TEST FOR ARRIVAL. The distance term below is
    # still computed from `_ident` against the navigation graph, so shaping geometry is
    # untouched.
    _pos_by_node = {}
    try:
        from retro_ai.training.targets import build_targets as _bt

        for _t in _bt(level):
            if _t.node_ident and _t.pos is not None and _t.trigger == "position":
                _pos_by_node[_t.node_ident] = _t.pos
    except Exception:  # pragma: no cover - level without a target table
        _pos_by_node = {}

    for _group in (
        [] if _drop_milestones else (getattr(_lvl_map, "reward_waypoints", None) or [])
    ):
        _members = []
        for _ident in _group:
            _idx = _wp_nav.node_by_ident.get(_ident)
            if _idx is None:
                continue
            _node = _wp_nav.nodes[_idx]
            _mark = _pos_by_node.get(
                _ident, ((_node.x - 8) // 4, _lvl_map.floor_top_y[_node.floor])
            )
            _members.append((_ident, _mark[0], _mark[1]))
        if _members:
            _wp_groups.append(_members)

    # _wp_after_fruit[gi]: this group only enters the potential once every fruit
    # is collected (LevelMap.waypoints_after_fruit). Mirrors the phase rule the
    # BASE potential already applies to fruits-then-princess; the waypoint sum
    # never inherited it, which on L4 made the level unwinnable (see the field's
    # docstring). Empty on L1/L2/L3 -> every group always active -> byte-identical.
    _after_fruit_idents = set(getattr(_lvl_map, "waypoints_after_fruit", None) or ())
    _wp_after_fruit: list = [
        any(ident in _after_fruit_idents for ident, _wx, _wy in _members)
        for _members in _wp_groups
    ]

    # _wp_group_names[gi]: every NAME that refers to group gi — its graph idents
    # PLUS any curriculum waypoint id at the same position. The two naming
    # schemes coexist: the ascent milestones are graph nodes (``J10_11_b``) while
    # the seeder/curriculum calls the same point ``A1``. Resolving both here lets
    # ``restore_reached_waypoints`` accept whichever the caller has, instead of
    # silently failing to match (which would leave the backward pull in place).
    _wp_group_names: list = []
    try:
        from retro_ai.training.yeti_map import jump_waypoints as _jump_wps

        _pos_to_name = {
            (x, y): name for name, (x, y, _f) in _jump_wps(_lvl_map).items()
        }
    except Exception:  # pragma: no cover - level without jump waypoints
        _pos_to_name = {}
    for _members in _wp_groups:
        _names = {ident for ident, _wx, _wy in _members}
        for _ident, _wx, _wy in _members:
            _alias = _pos_to_name.get((_wx, _wy))
            if _alias:
                _names.add(_alias)
        _wp_group_names.append(_names)

    class _GroundedPBRS:
        """PBRS path-progress shaping with an airborne freeze + death gate.

        Reuses the base reward's ``_potential`` (and its deliberate
        ``last_floor`` residue) verbatim, but reimplements the per-step
        shaping bookkeeping.

        ~~~ NON-MARKOVIAN DEVIATIONS (all deliberate; documented so we don't
        re-litigate them) ~~~

        PBRS is Markovian by construction: shaping is ``gamma*Phi(s') -
        Phi(s)``, a function of the (s, s') transition only, and round trips
        telescope to zero so nothing can be farmed. The level-2 problem
        ("don't reward a fall") is fundamentally a statement about *how* the
        agent got somewhere, which state alone cannot express (falling to
        floor 2 and laddering to floor 2 are the SAME state). So a correct
        fix MUST look at the transition, i.e. be slightly non-Markovian. The
        three deviations, each bounded and intentional:

        (D1) ``last_floor`` fallback (inside base ``_potential``): current
             floor is inferred with one step of memory because pixel-y is
             ambiguous mid-jump. Keeps a jump from looking like graph
             progress.

        (D2) AIRBORNE FREEZE + HOLD. While the sprite pose is airborne
             (jump/fall/death-anim, i.e. pose not in ``SURFACE_POSES``) we
             credit nothing AND hold ``prev_phi`` unchanged (we do NOT
             rebaseline it, and we do NOT sample the potential). Holding is
             what restores telescoping across a jump: the pre-jump baseline
             survives the airborne frames, so the return leg is charged on
             landing and an approach-then-jump-back round trip nets exactly 0.
             (The earlier version returned ``Phi=None`` while airborne, which
             rebaselined ``prev_phi`` to None and DELETED the return-leg debt
             -> a free +Phi per approach/jump-back cycle. PPO farmed that over
             15M steps and sat at spawn; see experiments/003 H-AH.)

        (D3) DEATH GATE. On a grounded frame that is a death (``ctx.died``,
             the cause-agnostic 0x2AFC flag), we credit nothing and
             rebaseline. Combined with (D2) deferring all credit to the
             grounded landing, this guarantees a fatal transition is never
             rewarded, for ANY cause (fall, goat, snowball) and on any level
             direction. A *survived* descent (alive on landing) IS credited
             (it is real progress and cannot be farmed -- climbing back up is
             grounded and charged symmetrically at gamma=1).

        (D4) DEFERRED FRUIT CREDIT (opt-in via ``defer_fruit_credit``). The
             SPARSE fruit reward was previously added at the pickup step and
             returned even on an airborne/dying frame (measured: the L2 agent
             grabs fruit 2 while already falling -> banks the reward -> dies,
             so the fatal early jump is locally optimal). When enabled we
             instead carry ``prev_fruits_grounded`` (the fruit count at the
             last grounded-alive frame) and credit fruit progress only on
             grounded-alive frames — never on a death frame. A fruit grabbed
             mid-air pays out only once the agent lands alive; if it dies
             first it pays 0. This is the SAME bounded "last grounded state"
             memory class as ``prev_phi``/``last_floor`` (D1/D2), so it does
             not add a new kind of non-Markovian dependence — it extends the
             deferral scheme already shipped. A grounded grab is credited the
             same step (equivalent to immediate).
        """

        def __init__(self) -> None:
            self._base = base
            self.prev_phi: float | None = None
            # (D4) fruit count at the last grounded-alive frame; None until
            # the first grounded-alive frame. Only used when defer_fruit.
            self._prev_fruits_grounded: int | None = None
            # Mandatory-WP targets: group indices reached this episode, and
            # the set of currently-active (reachable & unreached) groups last
            # step (a change => target set changed => rebaseline).
            self._reached_wp: set = set()
            self._prev_active_wp: frozenset = frozenset()
            # (D5) shaping paid so far this episode, refunded on death when
            # ``credit_requires_survival`` is on.
            self._shaping_acc: float = 0.0

        def reset(self) -> None:
            self._base.reset()
            self.prev_phi = None
            self._prev_fruits_grounded = None
            self._reached_wp = set()
            self._prev_active_wp = frozenset()
            self._shaping_acc = 0.0

        def restore_reached_waypoints(self, idents) -> None:
            """Mark milestone groups containing any of ``idents`` as reached.

            Called right after ``reset()`` on a SEEDED start so milestones the
            seed already banked stop being summed as pending targets (otherwise
            the potential pulls the agent BACKWARD; see the module-level
            ``restore_reached_waypoints``). Call order matters: reset() first,
            then this — the group set must be re-derived, not accumulated.
            """
            wanted = set(idents)
            for gi, names in enumerate(_wp_group_names):
                if names & wanted:
                    self._reached_wp.add(gi)

        @property
        def last_floor(self):  # exposed for probes/tests
            return self._base.last_floor

        def __call__(self, ctx: RewardContext) -> float:
            reward = 0.0

            # Sparse terms. The fruit term is paid at pickup by default; when
            # deferring (D4) it is instead paid on the next grounded-alive
            # frame (below), so an airborne grab that dies before landing
            # pays 0.
            picked = ctx.curr_fruits < ctx.prev_fruits
            if picked and not defer_fruit:
                collected = ctx.prev_fruits - ctx.curr_fruits
                reward += collected * ctx.curr_bonus * fruit_scale
            if ctx.princess_touched:
                reward += ctx.prev_bonus * princess_scale

            pose = getattr(ctx, "pose", -1)
            airborne = pose is not None and pose >= 0 and pose not in _surf

            # MILESTONE MARKING. `mark_airborne` selects WHERE this happens; the test
            # itself is identical either way, so the two arms differ in one thing only.
            #   True  (default) -- here, BEFORE the airborne return, so a jump LANDING
            #                      can be marked.
            #   False           -- after `phi`, where it used to be, i.e. only on frames
            #                      in the reward's SURFACE_POSES. Kept ONLY so the v14
            #                      control arm is reproducible from a config instead of
            #                      from a git checkout.
            #
            # MILESTONE MARKING, ahead of the airborne return (2026-09-14).
            #
            # This used to live below, after `if airborne: return reward`, which made it
            # unreachable on airborne frames however "ungated" the loop itself looked.
            # Every mandatory L4 target that is a JUMP LANDING is therefore nearly
            # unmarkable, because a landing is exactly when the agent is airborne.
            # Measured over 120 from-reset episodes: the sprite test contains `Rope1`
            # (px 108, y 118) in 107 episodes, almost all in pose 9, and the reward
            # marked it in 2. `Spring` was 95 against 14.
            #
            # The cost is not just a missing credit: an unmarked group stays in the sum
            # and keeps pulling. On floor 12 the four unmarked backward terms outweigh
            # the two forward ones and REVERSE the gradient -- sum 984 at px 184 against
            # 888 at px 232, i.e. the shaping pushes the agent AWAY from the rope-2 gap.
            # With groups 0-7 marked it is 320 against 368, pulling toward the gap.
            #
            # Only MARKING moves. The airborne freeze below is untouched: returning
            # Phi=None while airborne rebaselined prev_phi, deleted the return-leg debt
            # and gave a free +Phi per approach-then-jump-back cycle, which PPO farmed
            # for 15M steps (H-AH). A mid-air mark cannot pay anything by itself: it
            # only changes the target set, and the resulting `active_wp` change
            # rebaselines on the next grounded frame -- the same discipline a fruit
            # pickup uses, and required, or the potential jumps when a term leaves.
            def _mark(c, p):
                """Mark every group the agent is currently within reach of."""
                if not _wp_groups or c.died or p in MARK_BLOCKED_POSES:
                    return
                for gi, members in enumerate(_wp_groups):
                    if gi in self._reached_wp:
                        continue
                    for _ident, wx, wy in members:
                        if reaches(
                            (wx, wy), c.curr_x, c.curr_y, wp_tol, mode=wp_reach_mode
                        ):
                            self._reached_wp.add(gi)
                            break

            if mark_airborne:
                _mark(ctx, pose)

            # (D2) Airborne: no credit, HOLD prev_phi (don't sample/rebaseline)
            # and HOLD the deferred-fruit baseline (don't sample it either).
            if airborne:
                return reward

            phi = self._base._potential(ctx)  # grounded; also updates last_floor

            # (D4) Deferred fruit credit: on a grounded-ALIVE frame, credit any
            # fruit collected since the last grounded-alive frame. Never on a
            # death frame (so a fatal airborne grab is never paid). fruit_disc
            # marks the resulting discontinuity so shaping rebaselines (the
            # potential's target set changed), exactly as `picked` does in the
            # immediate case.
            fruit_disc = False
            if defer_fruit and not ctx.died:
                if (
                    self._prev_fruits_grounded is not None
                    and ctx.curr_fruits < self._prev_fruits_grounded
                ):
                    collected = self._prev_fruits_grounded - ctx.curr_fruits
                    reward += collected * ctx.curr_bonus * fruit_scale
                    fruit_disc = True
                self._prev_fruits_grounded = ctx.curr_fruits

            # Mandatory-waypoint targets: add their path-distance to the
            # potential (summed, unordered, like fruits), min over an OR-group,
            # dropping unreachable groups. Mark a group reached when the agent
            # (grounded, alive) is within tol of any member. active_wp = the
            # reachable-and-unreached groups; a change means the target set
            # changed -> rebaseline (same discipline as a fruit pickup).
            if not mark_airborne:
                # v14 control arm: marking only on frames that reach this point, i.e.
                # the reward's SURFACE_POSES. This is what made every mandatory jump
                # LANDING nearly unmarkable (Rope1 2/120 against 107/120 touched).
                _mark(ctx, pose)

            active_wp: frozenset = frozenset()
            # (D6) phi_pay is what THIS frame is paid against; phi is what the NEXT
            # frame is measured from. They differ only on a frame where the waypoint
            # list changed and the old list could be priced -- see pay_on_target_change.
            phi_pay = phi
            pay_old = False
            if _wp_groups and phi is not None and not ctx.died:
                agent_pix_x = int(ctx.curr_x) * 4 + 8
                # Reuse the SAME segment the base _potential just resolved, so
                # the waypoint-sum is shaped along the escalator descent too
                # (not floor-quantized). ladder = None => floor-only, as today.
                seg_floor, seg_ladder, _seg_x, seg_y = self._base._seg
                floor = seg_floor
                ladder = seg_ladder
                if floor is None and ladder is None:
                    floor = self._base.last_floor
                # Marking now happens ABOVE, before the airborne return, so that a jump
                # LANDING can be credited. The reach test there is the shared
                # retro_ai.training.targets.reaches -- the SAME comparison and geometry
                # mode the curriculum uses, so the two cannot drift. NOTE the tolerances
                # still differ in "box" mode: the reward passes `waypoint_reward_tol`
                # (2) while the curriculum passes 6 for jump waypoints; in "sprite" mode
                # the tolerance is ignored entirely, removing that divergence.
                fruits_left = bool(ctx.fruits_present) and any(ctx.fruits_present)

                def _dist(gi):
                    """Path distance to group ``gi``'s nearest member, from here."""
                    return min(
                        _wp_nav.path_distance_from_pos(
                            floor, ladder, agent_pix_x, seg_y, ident
                        )
                        for ident, _wx, _wy in _wp_groups[gi]
                    )

                # dist: every group priced AT THIS POSITION, so the old and the new
                # list can be summed from the same measurements.
                dist: dict = {}
                active = set()
                for gi, members in enumerate(_wp_groups):
                    if gi in self._reached_wp:
                        continue
                    if _wp_after_fruit[gi] and fruits_left:
                        continue  # not this phase yet
                    dmin = _dist(gi)
                    if dmin < _WP_UNREACHABLE:
                        dist[gi] = dmin
                        active.add(gi)
                phi_base = phi
                phi = phi_base - progress_scale * sum(dist[gi] for gi in active)
                active_wp = frozenset(active)
                phi_pay = phi

                # (D6) The list changed: price the frame against the list it STARTED
                # with. Groups that just left are still measurable from here -- being
                # marked reached does not move the agent -- so this charges movement
                # only. If any of them has become UNREACHABLE this frame there is no
                # honest price for the old list, so fall through to today's skip.
                if pay_on_target_change and active_wp != self._prev_active_wp:
                    for gi in self._prev_active_wp - active_wp:
                        dmin = _dist(gi)
                        if dmin < _WP_UNREACHABLE:
                            dist[gi] = dmin
                    if all(gi in dist for gi in self._prev_active_wp):
                        phi_pay = phi_base - progress_scale * sum(
                            dist[gi] for gi in self._prev_active_wp
                        )
                        pay_old = True

            # (D3) death gate + standard PBRS rebaseline on discontinuities
            # (episode start: prev_phi None; unresolved floor: phi None;
            # fruit pickup / princess / a WP target-set change: the summed
            # target set changes -- sparse terms cover those). Shaping resumes
            # on the next grounded, alive step.
            #
            # ``pay_old`` (D6) exempts the waypoint arm: the frame HAS an honest price
            # against the list it started with, so it is paid instead of skipped and
            # only the rebaseline is kept. The other arms are unchanged.
            if (
                ctx.died
                or picked
                or fruit_disc
                or ctx.princess_touched
                or self.prev_phi is None
                or phi is None
                or (active_wp != self._prev_active_wp and not pay_old)
            ):
                # (D5) UNIFIED CREDIT RULE (opt-in via
                # ``credit_requires_survival``): credit only for progress you
                # SURVIVE — applied identically to every target type.
                #
                # The death gate above only suppresses shaping ON the fatal
                # step; progress banked EARLIER was kept. That made arriving
                # recklessly strictly better than waiting: measured on L3,
                # "touch SN3 then die" paid +5.04 while waiting for a safe phase
                # paid 0.00, so the policy learned to arrive and die and never
                # learned to survive there (at SN3 it is no better than random).
                # Fruits already had this protection (``defer_fruit_credit``
                # holds the sparse grab until a grounded-alive frame); milestones
                # and path-progress did not. Refunding the episode's accumulated
                # shaping on death is the same idea for the SHAPING term: it is
                # PBRS with the terminal potential set to the episode baseline,
                # so progress-then-die nets ~0 while progress-then-survive keeps
                # paying. Sparse target bonuses already earned (a fruit banked
                # while grounded-alive) are NOT clawed back — those are real
                # achievements, not approach credit.
                if credit_requires_survival and ctx.died:
                    reward -= self._shaping_acc
                    self._shaping_acc = 0.0
                self.prev_phi = phi
                self._prev_active_wp = active_wp
                return reward

            # PBRS, with both sides of the comparison measured against the SAME target
            # list: prev_phi was recorded under the old list, so phi_pay prices this
            # position under the old list too. The baseline then moves to phi, which is
            # the new list -- that is where the list change is absorbed, unpaid.
            # phi_pay is phi whenever the list did not change (pay_old False).
            shaped = gamma * phi_pay - self.prev_phi
            reward += shaped
            self._shaping_acc += shaped
            self.prev_phi = phi
            self._prev_active_wp = active_wp
            return reward

    return _GroundedPBRS()


__all__ = [
    "RewardContext",
    "RewardFn",
    "RewardFactory",
    "available",
    "create",
    "register",
    "reset_reward",
]
