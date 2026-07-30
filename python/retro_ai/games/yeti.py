"""Yeti (MO5, 1984 Loriciels) — single source of truth for RAM layout,
sprite poses, per-level data, and death detection.

Before this module these facts were re-declared in ~20 scripts, and some
copies drifted into bugs — most importantly, lives-based death detection
(``lives < prev_lives``) is copy-pasted across the eval/profile/render
scripts but the lives byte is INERT on level 2, so those all silently miss
L2 deaths. The authoritative death signal is the cause-agnostic 0x2AFC flag
(validated on L1 ladder/jump/snowball + L2 falls); use :func:`is_dead`.

Scope note: the navigation graph and pixel-y -> floor mapping still live in
``retro_ai.training.yeti_map`` (imported by the nav/reward code). They should
eventually move here too; kept separate for now to limit churn.
"""

from __future__ import annotations

from typing import Mapping, Tuple

# --- RAM addresses (decimal; hex in comments) ------------------------------
FRUITS_ADDR = 11055  # 0x2B2F: fruits remaining (game-global counter)
# 0x2B57: lives. NOTE: does NOT decrement at the death frame on EITHER level
# (measured — it stays put through death→termination; it likely only changes at
# respawn). Do not use for prompt death detection; use is_dead() (0x2AFC).
LIVES_ADDR = 11095
X_ADDR = 11090  # 0x2B52: player X (RAM units; pixel = x*4 + 8)
Y_ADDR = 11089  # 0x2B51: player Y (pixel)
BONUS_HI = 11010  # 0x2B02: bonus countdown, high byte
BONUS_LO = 11011  # 0x2B03: bonus countdown, low byte
SCORE_HI = 11093  # 0x2B55: score, high byte
SCORE_LO = 11094  # 0x2B56: score, low byte
POSE_ADDR = 11092  # 0x2B54: player sprite-pose index (see SURFACE_POSES)
PRINCESS_FLAG_ADDR = 11050  # 0x2B2A: level-cleared flag (0->1 rising edge)

# Fast, cause-agnostic death flag: 0x2AFC == 65 => dead, 32 => alive. Flips at
# the TRUE death frame regardless of cause. Measured on BOTH levels: on L2 it
# is the only working signal (lives inert); on L1 it fires exactly when the
# bonus freezes — ~1 gym-step before the native bonus-stall termination, and
# the lives byte never moves. Same address the native interface uses on L2
# (game_profiles/mo5_yeti_fruit_level2.yaml death_flag_addr/value).
DEATH_FLAG_ADDR = 11004  # 0x2AFC
DEATH_FLAG_VALUE = 65

# Sprite poses where the player is on a surface (grounded floor / ladder):
# {0-3 walk-right, 4-5 walk-left, 8 ladder up/down/idle}. Airborne/freeze
# poses: {9 jump-right, 10 jump-left, 11 fall, 12 death-anim}. 0x2B54 is a
# display sprite index, not the full physics state — key on the whitelist,
# not single values. See experiments/003-yeti-training.md "run 3".
SURFACE_POSES = frozenset({0, 1, 2, 3, 4, 5, 8})

# Per-level fruit-presence bytes (non-zero = on map, zero = collected). L1 has
# 4 fruits, L2 has 2. These are POSITIONAL presence flags, distinct from the
# FRUITS_ADDR remaining-counter.
FRUIT_PRESENCE_BY_LEVEL: Mapping[int, Mapping[int, int]] = {
    1: {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8},
    2: {1: 11950, 2: 11975},  # 0x2EAE, 0x2EC7
    # L3 has a single fruit, so the global fruits-remaining counter IS that
    # fruit's presence (1 = on map, 0 = collected). Avoids needing a dedicated
    # per-fruit presence byte.
    3: {1: FRUITS_ADDR},  # 0x2B2F
}


def fruit_presence_addrs(level: int) -> Mapping[int, int]:
    """Per-fruit presence RAM addresses for ``level`` (1 or 2)."""
    return FRUIT_PRESENCE_BY_LEVEL[level]


# --- small read helpers (accept any object with read_ram_byte) -------------
def read_lives(iface) -> int:
    return iface.read_ram_byte(LIVES_ADDR)


def read_fruits_remaining(iface) -> int:
    return iface.read_ram_byte(FRUITS_ADDR)


def read_bonus(iface) -> int:
    return (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)


def read_score(iface) -> int:
    return (iface.read_ram_byte(SCORE_HI) << 8) | iface.read_ram_byte(SCORE_LO)


def read_pos(iface) -> Tuple[int, int]:
    return iface.read_ram_byte(X_ADDR), iface.read_ram_byte(Y_ADDR)


def read_pose(iface) -> int:
    return iface.read_ram_byte(POSE_ADDR)


def is_grounded(iface) -> bool:
    """True when the player sprite is on a surface (grounded floor / ladder)."""
    return read_pose(iface) in SURFACE_POSES


def waypoints(level: int) -> Mapping[str, Tuple[int, int, int]]:
    """Ladder-top/bottom WAYPOINTS for a level, as (x_ram, y_px, floor).

    Positions are DERIVED from the level's tilemap (via yeti_map), not
    captured by play: each vertical ladder gives a top waypoint (on its
    upper floor) and a bottom waypoint (on its lower floor), both at the
    ladder's x. Ladder x is stored in PIXELS; the agent's X RAM byte is in
    4px units offset by 8 (pixel = x_ram*4 + 8), so x_ram = (x_px - 8)//4.
    y is the floor-top pixel (the standing y on that floor).

    These are used only as position DETECTION TARGETS for curriculum
    waypoint capture — a waypoint is reached when the agent is grounded
    within a tolerance of one of these (x_ram, y). Waypoints are NOT goals
    and carry no success semantics (success is always "grab a fruit").
    """
    from retro_ai.training.yeti_map import get_level_map

    m = get_level_map(level)
    # Optional per-ladder placement (L3): expose only the arrival end. Absent
    # (L1/L2) or unlisted ladder -> "both" ends, as before.
    ends = getattr(m, "waypoint_ends", None) or {}
    wps: dict[str, Tuple[int, int, int]] = {}
    for name, upper_floor, lower_floor, x_px in m.ladders:
        x_ram = (int(x_px) - 8) // 4
        which = ends.get(name, "both")
        if which in ("top", "both"):
            wps[f"{name}_top"] = (x_ram, m.floor_top_y[upper_floor], upper_floor)
        if which in ("bot", "both"):
            wps[f"{name}_bot"] = (x_ram, m.floor_top_y[lower_floor], lower_floor)
    return wps


def is_dead(iface) -> bool:
    """Authoritative death check for any Yeti level.

    Reads the 0x2AFC flag (== 65 => dead), which flips at the true death frame
    on both levels regardless of cause (fall/goat/snowball) — measured. This
    is the single death signal; the lives byte is NOT used because it does not
    decrement at death on either level (see LIVES_ADDR).
    """
    return iface.read_ram_byte(DEATH_FLAG_ADDR) == DEATH_FLAG_VALUE
