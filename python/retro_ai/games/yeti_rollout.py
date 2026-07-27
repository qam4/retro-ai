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
from typing import List, Optional, Tuple

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
    actions: List[Tuple[int, ...]] = field(default_factory=list)
    frames: Optional[List[np.ndarray]] = None  # raw frames if keep_frames


def rollout_episode(
    stack,
    model,
    *,
    level: int,
    fruits_total: int,
    start_state: Optional[bytes] = None,
    settle: int = 5,
    max_steps: int = 1000,
    stall_threshold: int = 15,
    deterministic: bool = True,
    keep_frames: bool = False,
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
    """
    base = stack.base
    gym_env = stack.gym
    iface = base._interface

    obs, _ = gym_env.reset()
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
    actions: List[Tuple[int, ...]] = []
    frames: Optional[List[np.ndarray]] = [] if keep_frames else None

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
        actions.append(tuple(int(a) for a in np.ravel(action)))
        if y > max_y:
            max_y = y
        floor = agent_floor_from_pixel_y(y, level=level)
        # Only credit a floor the agent is actually STANDING on (grounded /
        # ladder pose) — a fall passing through a floor line doesn't count.
        if floor is not None and floor > deepest_floor and pose in yeti.SURFACE_POSES:
            deepest_floor = floor
        if keep_frames and base._last_raw_obs is not None:
            frames.append(np.asarray(base._last_raw_obs, dtype=np.uint8).copy())

        max_cp = max(max_cp, fruits_total - fruits)

        # Termination priority: princess -> death -> stall -> env done.
        if princess == 1 and prev_princess == 0:
            touched = True
            max_cp = fruits_total + 1
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
        actions=actions,
        frames=frames,
    )
