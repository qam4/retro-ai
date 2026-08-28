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

# EVERY pose code we have identified, and how. 0x2B54 is a display sprite index, not
# the full physics state, so behaviour is keyed on whitelists below rather than on
# single values.
#
# This catalogue exists because an incomplete one cost real training time: the walk
# cycle is FOUR poses per direction, but only the rightward cycle was ever fully
# listed. Poses 6 and 7 are grounded leftward-walk frames and were absent from
# ``SURFACE_POSES``, so roughly half of all leftward walking was invisible to waypoint
# detection, seed capture and reward marking. Use :func:`unknown_poses` to assert that
# a run never sees a code that is not in here.
#
# Measured (debug walk probe, 4 L4 floors, holding a direction and logging pose with
# the per-step lateral delta at the floor's standing y):
#   poses 0,1,2,3 -> dx in {0, +4}   the RIGHTWARD walk cycle
#   poses 4,5,6,7 -> dx in {-4, 0}   the LEFTWARD walk cycle
# and (L4 rope/trampoline traces): 14 carries the agent along a rope, 16/17 lift it
# vertically at constant x off the trampoline below the rope-2 gap.
POSE_NAMES: Mapping[int, str] = {
    0: "walk-right (grounded)",
    1: "walk-right (grounded)",
    2: "walk-right (grounded)",
    3: "walk-right (grounded)",
    4: "walk-left (grounded)",
    5: "walk-left (grounded)",
    6: "walk-left (grounded) -- NOT in SURFACE_POSES, see below",
    7: "walk-left (grounded) -- NOT in SURFACE_POSES, see below",
    8: "ladder up/down/idle (grounded)",
    9: "jump-right (airborne)",
    10: "jump-left (airborne)",
    11: "fall (airborne)",
    12: "death animation",
    13: "escalator ride (L3), a controlled vertical traversal",
    14: "rope carry (L4), lateral motion while held",
    16: "trampoline rise, facing right (L4)",
    17: "trampoline rise, facing left (L4)",
}
KNOWN_POSES = frozenset(POSE_NAMES)

# Sprite poses where the player is on a surface (grounded floor / ladder).
#
# !! KNOWN INCOMPLETE: poses 6 and 7 are grounded leftward-walk frames (measured, see
# POSE_NAMES) and are deliberately NOT added here yet. Adding them is a REWARD change
# -- it alters which frames can mark a milestone -- so it invalidates existing
# champions as warm-starts and must be run as its own lever. Tracked in
# experiments/003-yeti/level4_notes.md.
#
# Consequence while they are absent: on a leftward approach only about half of the
# grounded frames are eligible for detection or capture (measured 54% suppressed on L4
# floor 12, versus 0% walking right).
SURFACE_POSES = frozenset({0, 1, 2, 3, 4, 5, 8})
# Grounded leftward-walk poses missing from SURFACE_POSES. Named so callers and tests
# can refer to the gap explicitly instead of re-deriving it.
SURFACE_POSES_MISSING_LEFT = frozenset({6, 7})

# Per-fruit "is this fruit still on the map" addresses: NON-ZERO means present,
# zero means collected. That predicate is the whole contract — the table does not
# promise any particular KIND of byte.
#
# Why per-fruit at all: FRUITS_ADDR counts how many remain but not WHICH, and the
# path-progress reward shapes toward each remaining fruit individually. On a
# multi-fruit level the count alone would keep pulling the agent toward fruit it
# already ate, so those levels need a positional flag per fruit.
#
# On a SINGLE-fruit level the count and the predicate coincide exactly (1 = the
# one fruit is there, 0 = collected), so FRUITS_ADDR satisfies the contract
# directly. That is the general case collapsing, not a per-level workaround, and
# it is why L3/L4 need no dedicated byte.
FRUIT_PRESENCE_BY_LEVEL: Mapping[int, Mapping[int, int]] = {
    1: {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8},
    2: {1: 11950, 2: 11975},  # 0x2EAE, 0x2EC7
    3: {1: FRUITS_ADDR},  # single fruit -> the counter IS the predicate
    4: {1: FRUITS_ADDR},  # single fruit, same reason
}


def fruit_presence_addrs(level: int) -> Mapping[int, int]:
    """Addresses whose non-zero value means that fruit is still on the map."""
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


def unknown_poses(poses) -> set:
    """Which of ``poses`` are not in :data:`POSE_NAMES`?

    An uncatalogued pose is not a curiosity, it is a silent behaviour change: every
    pose-gated decision (waypoint detection, seed capture, reward milestone marking,
    floor crediting) treats an unrecognised code as "not on a surface", so whatever the
    agent was doing in that frame does not count. That is how poses 6 and 7 -- half the
    leftward walk cycle -- went unnoticed while suppressing ~54% of grounded leftward
    frames.

    Callers should surface the result rather than raise: an unknown pose means the
    catalogue needs extending, not that the run is invalid.
    """
    return {int(p) for p in poses} - KNOWN_POSES


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
    # Ladder tuples are ordered (name, TOP_floor, BOT_floor, x) on every level
    # (top = higher on screen = smaller y) — the same positional rule as
    # yeti_map.build_fixed_nodes, so "<name>_top"/"_bot" agree between the
    # seeder and the reward graph.
    for name, top_floor, bot_floor, x_px in m.ladders:
        x_ram = (int(x_px) - 8) // 4
        which = ends.get(name, "both")
        if which in ("top", "both"):
            wps[f"{name}_top"] = (x_ram, m.floor_top_y[top_floor], top_floor)
        if which in ("bot", "both"):
            wps[f"{name}_bot"] = (x_ram, m.floor_top_y[bot_floor], bot_floor)
    # Jump-edge landing waypoints (L3 A1..A5 ascent): seed/reach targets on the
    # otherwise-unseedable jump platforms. NOT reward targets (reward
    # unchanged). Empty on L1/L2 (no jump_waypoint_names). Ladder/jump names are
    # disjoint by construction, so this never clobbers a ladder waypoint.
    from retro_ai.training.yeti_map import jump_waypoints

    wps.update(jump_waypoints(m))
    return wps


def is_dead(iface) -> bool:
    """Authoritative death check for any Yeti level.

    Reads the 0x2AFC flag (== 65 => dead), which flips at the true death frame
    on both levels regardless of cause (fall/goat/snowball) — measured. This
    is the single death signal; the lives byte is NOT used because it does not
    decrement at death on either level (see LIVES_ADDR).
    """
    return iface.read_ram_byte(DEATH_FLAG_ADDR) == DEATH_FLAG_VALUE
