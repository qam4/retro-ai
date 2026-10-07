"""Shared single-episode rollout harness for Yeti.

Before this, ~19 scripts hand-rolled the same loop (load model -> reset /
load-state -> settle -> per-step predict/transpose/step -> read RAM ->
death/stall/princess termination -> deepest-CP tracking). That duplication is
exactly how the lives-based death bug spread to ~8 copies. This module is the
one place that logic lives; it consumes ``retro_ai.games.yeti`` for RAM layout
and death detection.

Scope: one episode per call. Callers keep their own aggregation / video /
heatmap logic and read what they need off :class:`EpisodeResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from retro_ai.games import yeti
from retro_ai.training.yeti_map import agent_floor_from_pixel_y


@dataclass
class EpisodeResult:
    """Outcome of one rolled-out episode."""

    length: int
    end_reason: str  # "princess" | "death" | "stall" | "env_done" | "max_steps"
    max_cp: int  # deepest checkpoint: fruits collected; princess = fruits_total + 1
    fruits_collected: int
    princess_touched: bool
    deepest_floor: int
    max_y: int
    final_x: int
    final_y: int
    positions: List[Tuple[int, int]] = field(default_factory=list)  # (x_px, y)
    # Sprite pose per step, parallel to ``positions``. Position alone cannot tell a
    # fall from a rope carry from a stalled stand, and L4 turns on poses the codebase
    # does not yet name (6, 14, 16, 17), so any analysis of a rope or spring segment
    # needs this alongside x/y. Detection is also pose-gated, so a step at the right
    # coordinates in the wrong pose does NOT mark a waypoint -- without the pose the
    # trace looks like an unexplained miss.
    poses: List[int] = field(default_factory=list)
    actions: List[Tuple[int, ...]] = field(default_factory=list)
    frames: Optional[List[np.ndarray]] = None  # raw frames if keep_frames
    # ROUTE DEPTH (only populated with track_waypoints=True).
    #
    # Why this exists: ``max_cp`` counts fruits plus the princess, so on a single-fruit
    # level it has three attainable values (0, 1, 2) and cannot distinguish a policy
    # that stops at the first obstacle from one that gets within a rung of the end.
    # Every L4 champion ever selected by keep_best_sweep was therefore arbitrary --
    # v4 kept its 100k snapshot, v5 its 1M, v6 its 200k, all on ties.
    #
    # ``max_rung`` is the SAME quantity training reports as ``reset_reach``: how many
    # MANDATORY route targets the episode has behind it, using the same shared reach
    # test and the same pose gate.
    reached_points: set = field(default_factory=set)
    max_rung: int = 0
    n_rungs: int = 0
    # First time each checkpoint was reached this episode: cp -> (step, bonus).
    # cp = fruits collected; princess = fruits_total + 1. Used by profiling.
    cp_arrival: Dict[int, Tuple[int, int]] = field(default_factory=dict)


def waypoint_frame_reaches(
    pos: Tuple[int, int],
    x_ram: int,
    y_px: int,
    pose: int,
    tol: int,
    mode: str,
    seed_poses: Optional[frozenset] = None,
) -> bool:
    """Does this one frame count as reaching the waypoint at ``pos``?

    The trainer's DETECTION rule, copied so eval and the route table agree
    (train_checkpoint_curriculum.py, the `_pose_ok_detect` block in step()):

    * ``"box"``    -- the agent must be on a surface (``seed_poses``: SURFACE_POSES
      plus the L3 escalator ride) AND within ``tol`` of ``pos``.
    * ``"sprite"`` -- any pose except fall / death (``yeti.NON_TRAVERSAL_POSES``) AND
      the agent's sprite overlaps ``pos``; ``tol`` is ignored.

    The two differ most on jump landings. A rope or jump crossing can fly over a
    landing anchor and touch down beyond it -- L4's rope 2 does exactly that, landing
    at px 88 past `Low2`'s px 128 -- which "sprite" counts and "box" never does.
    """
    from retro_ai.training.targets import reaches

    if mode == "box":
        gate = seed_poses if seed_poses is not None else (yeti.SURFACE_POSES | {13})
        if pose not in gate:
            return False
    elif mode == "sprite":
        if pose in yeti.NON_TRAVERSAL_POSES:
            return False
    else:
        raise ValueError(f"reach mode must be 'box' or 'sprite', got {mode!r}")
    return reaches(pos, x_ram, y_px, tol, mode=mode)


def rollout_episode(
    stack,
    model,
    *,
    level: int,
    fruits_total: int,
    start_state: Optional[bytes] = None,
    settle: int = 1,
    max_steps: int = 1000,
    stall_threshold: int = 15,
    deterministic: bool = True,
    keep_frames: bool = False,
    reset_env: bool = True,
    track_waypoints: bool = False,
    wp_tol: int = 2,
    wp_jump_tol: int = 6,
    reach_mode: str = "sprite",
) -> EpisodeResult:
    """Roll out a single Yeti episode under the training termination rules.

    Termination priority (matches the eval/analysis scripts): princess touch
    (rising edge of the level-cleared flag) -> death (0x2AFC via
    :func:`yeti.is_dead`) -> bonus-stall -> underlying env done/trunc ->
    max_steps.

    Parameters
    ----------
    stack : the object returned by ``build_training_env`` (``.base``, ``.gym``,
        ``.preprocessed``, ``.base._interface``).
    level, fruits_total : level geometry (L1: 4 fruits, L2: 2).
    start_state : optional save-state bytes to load each episode (level 2 boots
        from a save, not a game reset). None => plain game reset.
    reset_env : if True (default) call ``gym_env.reset()`` each episode (fresh
        game reset — required when ``start_state`` is None). Set False for
        seed-pool rollouts that ``load_state`` a different seed each episode:
        this skips the (~32s on MO5) startup per episode. The CALLER must have
        booted the env once (one ``gym_env.reset()``) before the first call,
        and must pass a ``start_state`` with ``settle >= 1``.
    settle : NOOP steps taken after ``load_state`` before the policy acts.

        MUST match training, which takes exactly ONE. This defaulted to 5 while
        the training env was fixed to 1 ("a pure vestige of the
        pre-notify_state_loaded flush that burned ~20 game frames"), and the fix
        never propagated here — so every EVAL episode began by standing still
        for ~20 emulator frames while the level ran on. On L4, whose opening is
        a timed crossing against kangaroos, that is not a small handicap:

            L4 v4 best snapshot, fruit collected from reset, 40 episodes
                settle=5   4/40   (10%)
                settle=1  32/40   (80%)

        The training route table was right and the eval was wrong. Any
        historical from-reset number produced through this harness was measured
        with the handicap and is a LOWER BOUND.
    """
    base = stack.base
    gym_env = stack.gym
    iface = base._interface

    # --- route-depth tracking (opt-in; off => behaviour unchanged) -----------
    # Mirrors the trainer's DETECTION: same waypoint positions, same per-axis
    # tolerance (ladders `wp_tol`, jump landings `wp_jump_tol`, used by "box" only),
    # and -- through `waypoint_frame_reaches` -- the same reach test and pose gate for
    # the given `reach_mode`. Anything that diverges here makes eval numbers
    # incomparable with the route table, which is the whole point of it.
    #
    # It DID diverge, for four weeks. This comment used to say "mirrors the trainer
    # exactly", naming `targets.within_tol` and the grounded pose gate, while the
    # trainer switched to sprite overlap with a fail-open pose gate in de21939 and this
    # function never followed. The cost surfaced only when an agent finally crossed
    # rope 2: v30's champion touches the princess in 208/300 episodes, and every one
    # read `Low2` as unreached and `max_rung` 11 instead of 12, because the crossing
    # FLIES over `Low2`'s anchor, lands at px 88 and walks left -- it is never GROUNDED
    # inside the box. `reach_mode` now defaults to the trainer's default and
    # test_eval_reach_mode pins the two together.
    if reach_mode not in ("box", "sprite"):
        raise ValueError(f"reach_mode must be 'box' or 'sprite', got {reach_mode!r}")
    wps: dict = {}
    tol_of: dict = {}
    # The trainer's progress ladder, from the SAME function (targets.progress_ladder):
    # one rung per route STEP, satisfied by any member of its group. This counted
    # mandatory IDS until 2026-10-07, so reaching both members of an OR-group, or a
    # target under both its names, scored two rungs for one step. On L4 that was
    # invisible until rope 2 was crossed: the crossing flies over both `Low2` (px 128)
    # and `Lhi_down_bot` (px 104), and every princess episode read 13 of 13 against
    # the trainer's 12-rung ladder. `n_rungs` is the ladder's length, so L4 reports
    # /12 from that date on, where earlier evals said /13.
    ladder_groups: list = []
    n_rungs = 0
    if track_waypoints:
        from retro_ai.training.targets import progress_ladder
        from retro_ai.training.yeti_map import get_level_map, jump_waypoints

        wps = dict(yeti.waypoints(level))
        try:
            jump_ids = set(jump_waypoints(get_level_map(level)))
        except (ValueError, KeyError):
            jump_ids = set()
        tol_of = {w: (wp_jump_tol if w in jump_ids else wp_tol) for w in wps}
        ladder_groups, n_rungs = progress_ladder(level)
    seed_poses = frozenset(yeti.SURFACE_POSES | {13})
    reached_points: set = set()
    fruit_addrs = yeti.fruit_presence_addrs(level) if track_waypoints else {}

    if reset_env:
        obs, _ = gym_env.reset()
    elif start_state is None:
        raise ValueError("reset_env=False requires a start_state (seed-pool mode)")
    else:
        obs = None
    if start_state is not None:
        iface.load_state(start_state)
        stack.preprocessed.notify_state_loaded()  # drop pre-load frames
        for _ in range(settle):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])

    start_fruits = yeti.read_fruits_remaining(iface)
    prev_bonus = yeti.read_bonus(iface)
    prev_princess = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)

    x0, y0 = yeti.read_pos(iface)
    max_cp = fruits_total - start_fruits
    deepest_floor = agent_floor_from_pixel_y(y0, level=level) or 1
    max_y = y0
    stall = 0
    steps = 0
    touched = False
    end_reason = "max_steps"
    positions: List[Tuple[int, int]] = []
    poses: List[int] = []
    actions: List[Tuple[int, ...]] = []
    frames: Optional[List[np.ndarray]] = [] if keep_frames else None
    cp_arrival: Dict[int, Tuple[int, int]] = {}

    while steps < max_steps:
        action, _ = model.predict(
            np.transpose(obs, (2, 0, 1)), deterministic=deterministic
        )
        obs, _, done, trunc, _ = gym_env.step(action)
        steps += 1

        x, y = yeti.read_pos(iface)
        pose = yeti.read_pose(iface)
        fruits = yeti.read_fruits_remaining(iface)
        bonus = yeti.read_bonus(iface)
        princess = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)

        positions.append((x * 4, y))
        poses.append(int(pose))
        actions.append(tuple(int(a) for a in np.ravel(action)))
        if y > max_y:
            max_y = y
        floor = agent_floor_from_pixel_y(y, level=level)
        # Only credit a floor the agent is actually STANDING on (grounded /
        # ladder pose) — a fall passing through a floor line doesn't count.
        if floor is not None and floor > deepest_floor and pose in yeti.SURFACE_POSES:
            deepest_floor = floor
        if track_waypoints:
            for wid, (wx, wy, _f) in wps.items():
                if waypoint_frame_reaches(
                    (wx, wy),
                    x,
                    y,
                    int(pose),
                    tol_of.get(wid, wp_tol),
                    reach_mode,
                    seed_poses=seed_poses,
                ):
                    reached_points.add(wid)

        if keep_frames and base._last_raw_obs is not None:
            frames.append(np.asarray(base._last_raw_obs, dtype=np.uint8).copy())

        cp_now = fruits_total - fruits
        if cp_now > max_cp:
            max_cp = cp_now
            cp_arrival.setdefault(cp_now, (steps, bonus))

        # Termination priority: princess -> death -> stall -> env done.
        if princess == 1 and prev_princess == 0:
            touched = True
            max_cp = fruits_total + 1
            cp_arrival.setdefault(fruits_total + 1, (steps, bonus))
            end_reason = "princess"
            break
        prev_princess = princess

        if yeti.is_dead(iface):
            end_reason = "death"
            break

        if bonus == prev_bonus:
            stall += 1
        else:
            stall = 0
            prev_bonus = bonus
        if stall >= stall_threshold:
            end_reason = "stall"
            break

        if done or trunc:
            end_reason = "env_done"
            break

    x, y = yeti.read_pos(iface)
    # Collected fruits count toward the rung exactly as they do in training, where the
    # reached-set is waypoints UNION fruits-derived-from-their-presence-bytes.
    if track_waypoints:
        for fid, addr in fruit_addrs.items():
            if iface.read_ram_byte(addr) == 0:
                reached_points.add(f"F{fid}")
    if track_waypoints:
        from retro_ai.training.targets import rung_of

        max_rung = rung_of(ladder_groups, reached_points)
    else:
        max_rung = 0
    return EpisodeResult(
        length=steps,
        end_reason=end_reason,
        max_cp=max_cp,
        fruits_collected=fruits_total - yeti.read_fruits_remaining(iface),
        princess_touched=touched,
        deepest_floor=deepest_floor,
        max_y=max_y,
        final_x=x,
        final_y=y,
        positions=positions,
        poses=poses,
        actions=actions,
        frames=frames,
        cp_arrival=cp_arrival,
        reached_points=reached_points,
        max_rung=max_rung,
        n_rungs=n_rungs,
    )
