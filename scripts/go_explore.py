#!/usr/bin/env python3
"""Go-Explore Phase 1: exploration with cell archive.

Based on Ecoffet et al. (2019) "Go-Explore: A New Approach for
Hard-Exploration Problems". Uses save states to return to promising
cells, then explores randomly from there.

Usage:
    python scripts/go_explore.py --profile yeti --steps 500000 \
        --output output/mo5/yeti/go_explore
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
from PIL import Image, ImageDraw

# ── Cell definition ──────────────────────────────────────────────────────────


def make_cell(x, y, fruits_collected):
    """Discretize game state into a cell.

    Cell key = (floor_bucket, x_bucket, fruits_collected_frozenset).

    Y-buckets are 32 px tall anchored at the bottom of the screen so each
    floor lands in exactly one bucket. Game-y 200 is the bottom.
      floor 0 (bottom, floor 1): 168 <= y < 200
      floor 1 (floor 2):         136 <= y < 168
      floor 2 (floor 3):         104 <= y < 136
      floor 3 (floor 4):          72 <= y < 104
      floor 4 (princess area):    y < 72

    X-buckets are 8 px in game-x space (player x is 0..79 → 10 buckets).

    ``fruits_collected`` is a frozenset of floor numbers 1..4 for the
    fruits that have been collected so far — the full subset, not a
    count, so e.g. "fruit 2 collected only" and "fruit 3 collected
    only" are distinct cells.
    """
    y_from_bottom = 200 - y
    if y_from_bottom < 32:
        y_bucket = 0
    elif y_from_bottom < 64:
        y_bucket = 1
    elif y_from_bottom < 96:
        y_bucket = 2
    elif y_from_bottom < 128:
        y_bucket = 3
    else:
        y_bucket = 4

    x_bucket = min(x // 8, 9)
    return (y_bucket, x_bucket, frozenset(fruits_collected))


# ── Per-fruit RAM addresses ──────────────────────────────────────────────────
#
# The MO5 video RAM stores 2-byte sprite-character pairs for each of the
# four fruits. When a fruit is collected the game zeroes out those bytes
# so the sprite disappears. We only need to read ONE byte per fruit to
# detect collection; non-zero = present, zero = collected.
#
# The specific addresses were found by snapshotting RAM at CP0..CP4
# states in curriculum_v5 and looking for bytes that went to zero when
# a specific fruit was collected (see approach 10 in 003-yeti-training).
FRUIT_PRESENCE_ADDR = {
    1: 0x2FAD,  # fruit on floor 1 (+10 score)
    2: 0x2F00,  # fruit on floor 2 (+20 score)
    3: 0x2E68,  # fruit on floor 3 (+30 score)
    4: 0x2DD8,  # fruit on floor 4 (+40 score)
}


def read_fruits_collected(iface):
    """Read the set of floor numbers whose fruit has been collected."""
    collected = set()
    for floor, addr in FRUIT_PRESENCE_ADDR.items():
        if iface.read_ram_byte(addr) == 0:
            collected.add(floor)
    return collected


# ── Archive ──────────────────────────────────────────────────────────────────


class CellArchive:
    """Archive of discovered cells with save states and trajectories."""

    def __init__(self):
        # cell -> {state, score, steps, trajectory, times_chosen,
        #          times_chosen_since_new}
        self.cells = {}
        self.total_cells_found = 0

    def add_or_update(self, cell, state_bytes, score, steps, trajectory, fruits_order):
        """Add a new cell or update if this trajectory is better.

        ``fruits_order`` is the ordered list of floor numbers reflecting
        which fruit was collected at which step of the trajectory —
        preserved for debugging/analysis. It is NOT part of the cell key;
        the cell key already captures the *set* of fruits collected.
        """
        if cell not in self.cells:
            self.cells[cell] = {
                "state": state_bytes,
                "score": score,
                "steps": steps,
                "trajectory": trajectory,
                "fruits_order": list(fruits_order),
                "times_chosen": 0,
                "times_chosen_since_new": 0,
            }
            self.total_cells_found += 1
            return True  # new cell
        else:
            existing = self.cells[cell]
            # Better = higher score, or same score with fewer steps
            if score > existing["score"] or (
                score == existing["score"] and steps < existing["steps"]
            ):
                existing["state"] = state_bytes
                existing["score"] = score
                existing["steps"] = steps
                existing["trajectory"] = trajectory
                existing["fruits_order"] = list(fruits_order)
            return False  # existing cell

    def choose_cell(self):
        """Choose a cell to explore from, preferring newer/less-visited ones."""
        if not self.cells:
            return None

        cells = list(self.cells.items())
        weights = []
        for cell_key, info in cells:
            w = (
                1.0
                / (1.0 + info["times_chosen"])
                / (1.0 + info["times_chosen_since_new"])
            )
            # Bonus for more fruits collected.
            # cell_key[2] is a frozenset of floor numbers whose fruit
            # has been collected.
            n_collected = len(cell_key[2])
            fruit_bonus = 1.0 + n_collected * 2.0
            # Bonus for higher floors (y_bucket 0 = bottom, 4 = princess)
            floor_bonus = 1.0 + cell_key[0] * 0.5
            w *= fruit_bonus * floor_bonus
            weights.append(w)

        weights = np.array(weights)
        weights /= weights.sum()
        idx = np.random.choice(len(cells), p=weights)
        chosen_key, chosen_info = cells[idx]
        chosen_info["times_chosen"] += 1
        chosen_info["times_chosen_since_new"] += 1
        return chosen_key, chosen_info

    def reset_since_new(self):
        """Reset times_chosen_since_new for all cells (called when new cell found)."""
        for info in self.cells.values():
            info["times_chosen_since_new"] = 0


# ── Environment helpers ──────────────────────────────────────────────────────

YETI_ADDRS = {
    "x_pos": 11090,
    "y_pos": 11089,
    "score_hi": 11093,
    "score_lo": 11094,
    "lives": 11095,
    "bonus_hi": 11010,
    "bonus_lo": 11011,
    "fruits_remaining": 11055,
}


def make_env(profile_name):
    """Build a raw (non-preprocessed) env for Go-Explore."""
    from retro_ai.envs.base_env import BaseEnv
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profile = registry.load(profile_name)
    config_dict = {}
    if profile.reward_params:
        config_dict["reward_params"] = profile.reward_params
    base = BaseEnv(
        emulator_type=profile.emulator_type,
        rom_path=profile.rom_path,
        reward_mode=profile.reward_mode,
        config=config_dict or None,
        action_mode="joystick",
    )
    return base


def read_state(env):
    """Read game state from RAM."""
    iface = env._interface
    state = {}
    for name, addr in YETI_ADDRS.items():
        state[name] = iface.read_ram_byte(addr)
    state["score"] = (state["score_hi"] << 8) | state["score_lo"]
    state["bonus"] = (state["bonus_hi"] << 8) | state["bonus_lo"]
    state["fruits_collected"] = read_fruits_collected(iface)
    return state


def is_dead(state, stall_count, prev_lives):
    """Check if the player died (bonus stall or life loss)."""
    if state["lives"] < prev_lives and prev_lives > 0:
        return True
    if stall_count >= 10:
        return True
    return False


# ── Joystick actions ─────────────────────────────────────────────────────────

# All 18 joystick actions: (vertical, horizontal, fire)
# vertical: 0=neutral, 1=up, 2=down
# horizontal: 0=neutral, 1=right, 2=left
# fire: 0=no, 1=jump
ACTIONS = [
    [0, 0, 0],  # noop
    [1, 0, 0],  # up
    [2, 0, 0],  # down
    [0, 1, 0],  # right
    [0, 2, 0],  # left
    [0, 0, 1],  # jump
    [1, 1, 0],  # up+right
    [1, 2, 0],  # up+left
    [2, 1, 0],  # down+right
    [2, 2, 0],  # down+left
    [0, 1, 1],  # right+jump
    [0, 2, 1],  # left+jump
    [1, 0, 1],  # up+jump
    [2, 0, 1],  # down+jump
    [1, 1, 1],  # up+right+jump
    [1, 2, 1],  # up+left+jump
    [2, 1, 1],  # down+right+jump
    [2, 2, 1],  # down+left+jump
]


def random_action(prev_action, sticky_prob=0.85):
    """Random action with high probability of repeating previous action.

    This is key to Go-Explore: sticky actions help the agent commit to
    a direction rather than jittering randomly.
    """
    if prev_action is not None and random.random() < sticky_prob:
        return prev_action
    return random.choice(ACTIONS)


# ── Seeding helpers ──────────────────────────────────────────────────────────


def _seed_archive(env, archive, args):
    """Load seed cells from ``args.seed_archive``, validate, add to archive.

    Skips cells whose ``fruits_collected`` is below ``--seed-min-cp``
    (so callers can say "only seed from CP2 or higher"). Validates each
    candidate state via :func:`state_validator.validate_state` before
    adding it so doomed/frozen states don't become Go-Explore starts.

    Accepts both archive shapes:
      - new: ``cell_key[2]`` is a frozenset of collected floor numbers.
      - old: ``cell_key[2]`` is an int ``fruits_remaining`` (0..4).
    """
    import pickle

    from retro_ai.training.state_validator import validate_state

    print(f"  Seeding from {args.seed_archive}", flush=True)
    with open(args.seed_archive, "rb") as f:
        seed = pickle.load(f)

    # Save the env's current state so validation probes don't disturb it.
    saved = env.save_state()

    def _load(state_bytes):
        env.load_state(state_bytes)

    def _step_noop() -> bool:
        _, _, done, _, _ = env.step([0, 0, 0])
        return bool(done)

    def _cp_of(cell_key):
        """Derive CP level from a cell_key regardless of archive version."""
        if len(cell_key) < 3:
            return None
        v = cell_key[2]
        if isinstance(v, (frozenset, set, list, tuple)):
            return len(v)
        if isinstance(v, int) and 0 <= v <= 4:
            return 4 - v
        return None

    total = len(seed)
    filtered_cp = 0
    rejected = 0
    added = 0
    min_cp = args.seed_min_cp
    for cell_key, entry in seed.items():
        cp = _cp_of(cell_key)
        if min_cp is not None and (cp is None or cp < min_cp):
            filtered_cp += 1
            continue
        state_bytes = entry["state"]
        if args.seed_validate:
            result = validate_state(
                state_bytes=state_bytes,
                load_state=_load,
                step_noop=_step_noop,
            )
            if not result.viable:
                rejected += 1
                continue
        score = entry.get("score", 0)
        steps = entry.get("steps", 0)
        trajectory = entry.get("trajectory", [])
        fruits_order = entry.get("fruits_order", [])
        archive.add_or_update(
            cell_key, state_bytes, score, steps, trajectory, fruits_order
        )
        added += 1

    # Restore env to avoid disturbing the main loop that starts right after.
    env.load_state(saved)

    print(
        f"  Seeded: {added} cells added, {rejected} rejected by validator, "
        f"{filtered_cp} filtered by --seed-min-cp ({total} total)",
        flush=True,
    )


# ── Main exploration loop ────────────────────────────────────────────────────


def explore(args):
    env = make_env(args.profile)
    archive = CellArchive()

    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, "explore_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        [
            "iteration",
            "total_steps",
            "cells_found",
            "new_cells_this_iter",
            "chosen_cell",
            "explore_score",
            "explore_steps",
            "best_score",
            "best_floor",
            "wall_time",
        ]
    )

    total_steps = 0
    iteration = 0
    best_score = 0
    best_floor = 0
    best_trajectory = None
    start_time = time.time()

    # Initial exploration from reset
    env.reset()
    state = read_state(env)
    init_state = env.save_state()
    init_cell = make_cell(state["x_pos"], state["y_pos"], state["fruits_collected"])
    archive.add_or_update(init_cell, init_state, 0, 0, [], [])

    print(f"Go-Explore Phase 1: {args.steps} steps, profile={args.profile}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(flush=True)

    # Optional: seed the archive from a prior archive.pkl. States are
    # run through the viability validator first so we don't teleport
    # Go-Explore into a frozen/doomed state.
    if args.seed_archive:
        _seed_archive(env, archive, args)

    while total_steps < args.steps:
        iteration += 1
        new_cells_this_iter = 0

        # 1. Choose a cell from the archive
        choice = archive.choose_cell()
        if choice is None:
            break
        chosen_key, chosen_info = choice

        # 2. Return to that cell (load save state)
        env.load_state(chosen_info["state"])

        # 3. Explore randomly from there
        prev_action = None
        state = read_state(env)
        prev_bonus = state["bonus"]
        prev_lives = state["lives"]
        stall_count = 0
        # Full trajectory from game start = parent's trajectory + new actions
        base_trajectory = list(chosen_info["trajectory"])
        new_actions = []
        explore_steps = 0
        explore_score = state["score"]
        # Ordered list of fruits picked up along the way, starting from
        # the parent cell's known order. We extend it whenever a new
        # fruit appears in the ``fruits_collected`` set.
        fruits_order = list(chosen_info.get("fruits_order", []))
        prev_fruits_collected = set(state["fruits_collected"])

        for step in range(args.explore_steps):
            action = random_action(prev_action, args.sticky_prob)
            obs, reward, done, truncated, info = env.step(action)
            total_steps += 1
            explore_steps += 1
            prev_action = action

            state = read_state(env)
            new_actions.append(list(action))

            # Track any new fruit collection.
            new_fruits = state["fruits_collected"] - prev_fruits_collected
            if new_fruits:
                for f in sorted(new_fruits):
                    fruits_order.append(f)
                prev_fruits_collected = set(state["fruits_collected"])

            # Track bonus stall for death detection
            if state["bonus"] == prev_bonus:
                stall_count += 1
            else:
                stall_count = 0
                prev_bonus = state["bonus"]

            # Check death
            if is_dead(state, stall_count, prev_lives):
                break
            prev_lives = state["lives"]

            # Record cell — only save state if it's a new or improved cell
            # Skip death animation states (Y < 30 is above the game area)
            if state["y_pos"] < 30:
                continue
            cell = make_cell(
                state["x_pos"],
                state["y_pos"],
                state["fruits_collected"],
            )
            existing = archive.cells.get(cell)
            should_save = (
                existing is None
                or state["score"] > existing["score"]
                or (
                    state["score"] == existing["score"]
                    and (len(base_trajectory) + len(new_actions)) < existing["steps"]
                )
            )
            if should_save:
                full_trajectory = base_trajectory + new_actions
                cell_state = env.save_state()
                is_new = archive.add_or_update(
                    cell,
                    cell_state,
                    state["score"],
                    len(full_trajectory),
                    full_trajectory.copy(),
                    fruits_order,
                )
                if is_new:
                    new_cells_this_iter += 1
                    archive.reset_since_new()

            explore_score = state["score"]

            # Track best
            floor = cell[0]
            if floor > best_floor:
                best_floor = floor
                print(f"  NEW FLOOR: {best_floor} at step {total_steps}", flush=True)
            if explore_score > best_score:
                best_score = explore_score
                best_trajectory = (base_trajectory + new_actions).copy()

            if total_steps >= args.steps:
                break

        # Log
        elapsed = time.time() - start_time
        log_writer.writerow(
            [
                iteration,
                total_steps,
                len(archive.cells),
                new_cells_this_iter,
                str(chosen_key),
                explore_score,
                explore_steps,
                best_score,
                best_floor,
                f"{elapsed:.1f}",
            ]
        )
        log_file.flush()

        if iteration % 10 == 0:
            fps = total_steps / elapsed if elapsed > 0 else 0
            print(
                f"  iter={iteration} steps={total_steps}/{args.steps} "
                f"cells={len(archive.cells)} best_score={best_score} "
                f"best_floor={best_floor} fps={fps:.0f}",
                flush=True,
            )

    log_file.close()

    # Save results
    print("\n=== Go-Explore Phase 1 Complete ===")
    print(f"  Total steps: {total_steps}")
    print(f"  Total iterations: {iteration}")
    print(f"  Cells discovered: {len(archive.cells)}")
    print(f"  Best score: {best_score}")
    print(f"  Best floor: {best_floor}")
    print(f"  Wall time: {time.time() - start_time:.0f}s")

    # Save archive summary
    summary = {
        "total_steps": total_steps,
        "iterations": iteration,
        "cells_found": len(archive.cells),
        "best_score": best_score,
        "best_floor": best_floor,
        "wall_time": time.time() - start_time,
        "cells_by_floor": {},
        "cells_by_fruits_collected": {},
    }
    for cell_key in archive.cells:
        floor = str(cell_key[0])
        summary["cells_by_floor"][floor] = summary["cells_by_floor"].get(floor, 0) + 1
        # cell_key[2] is a frozenset of collected floor numbers.
        fc = str(len(cell_key[2]))
        summary["cells_by_fruits_collected"][fc] = (
            summary["cells_by_fruits_collected"].get(fc, 0) + 1
        )

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save best trajectory
    if best_trajectory:
        with open(os.path.join(args.output, "best_trajectory.json"), "w") as f:
            json.dump(
                {
                    "score": best_score,
                    "floor": best_floor,
                    "actions": [
                        a if isinstance(a, list) else list(a) for a in best_trajectory
                    ],
                },
                f,
            )

    # Visualize cells on game background
    visualize_archive(env, archive, args.output)

    # Save archive for Phase 2 (save states + trajectories)
    import pickle

    archive_path = os.path.join(args.output, "archive.pkl")
    archive_data = {}
    for cell_key, info in archive.cells.items():
        archive_data[cell_key] = {
            "state": bytes(info["state"]),
            "trajectory": info["trajectory"],
            "fruits_order": list(info.get("fruits_order", [])),
            "score": info["score"],
            "steps": info["steps"],
        }
    with open(archive_path, "wb") as f:
        pickle.dump(archive_data, f)
    print(f"  Saved archive ({len(archive_data)} cells) to {archive_path}", flush=True)

    env.reset()  # cleanup (no close method on BaseEnv)


def visualize_archive(env, archive, output_dir):
    """Draw discovered cells as a heatmap on the game background.

    Color intensity = number of unique cells discovered in each region.
    Cold (blue) = barely explored, hot (red/yellow) = heavily explored.
    """
    env.reset()
    # Run a few steps to get an actual gameplay frame
    for _ in range(10):
        env.step([0, 0, 0])
    bg = env._last_raw_obs
    if bg is None:
        return

    img = Image.fromarray(bg.copy())
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Floor Y ranges (pixel coordinates)
    floor_y_ranges = {
        0: (170, 200),  # floor 1 - bottom
        1: (140, 170),  # floor 2
        2: (110, 140),  # floor 3
        3: (80, 110),  # floor 4
        4: (40, 80),  # top area / princess
    }

    x_bucket_px = 320 // 8  # 40px per bucket

    # Count cells per region
    region_counts = {}
    for cell_key in archive.cells:
        y_bucket, x_bucket, _score = cell_key
        region = (y_bucket, x_bucket)
        region_counts[region] = region_counts.get(region, 0) + 1

    if not region_counts:
        return

    max_count = max(region_counts.values())

    for (y_bucket, x_bucket), count in region_counts.items():
        y_min, y_max = floor_y_ranges.get(y_bucket, (0, 200))
        x_min = x_bucket * x_bucket_px
        x_max = x_min + x_bucket_px

        # Heatmap: blue (cold) -> green -> yellow -> red (hot)
        t = count / max_count
        if t < 0.33:
            r, g, b = 0, int(255 * t * 3), 255
        elif t < 0.66:
            r, g, b = int(255 * (t - 0.33) * 3), 255, int(255 * (1 - (t - 0.33) * 3))
        else:
            r, g, b = 255, int(255 * (1 - (t - 0.66) * 3)), 0

        draw.rectangle([x_min, y_min, x_max, y_max], fill=(r, g, b, 90))
        # Show count in the cell
        cx = (x_min + x_max) // 2 - 4
        cy = (y_min + y_max) // 2 - 4
        draw.text((cx, cy), str(count), fill=(255, 255, 255, 200))

    # Grid lines
    for y_bucket, (y_min, y_max) in floor_y_ranges.items():
        draw.line([(0, y_min), (320, y_min)], fill=(255, 255, 255, 40), width=1)
    for i in range(9):
        x = i * x_bucket_px
        draw.line([(x, 40), (x, 200)], fill=(255, 255, 255, 40), width=1)

    # Composite
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Legend
    draw2 = ImageDraw.Draw(img)
    draw2.text(
        (5, 2),
        f"{len(region_counts)} regions / {len(archive.cells)} cells",
        fill=(255, 255, 255),
    )
    draw2.text(
        (5, 13),
        f"best score: {max(c['score'] for c in archive.cells.values())}",
        fill=(255, 255, 255),
    )

    img.save(os.path.join(output_dir, "cells.png"))
    print(f"  Saved cells visualization to {output_dir}/cells.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Go-Explore Phase 1")
    parser.add_argument("--profile", default="yeti", help="Game profile name")
    parser.add_argument(
        "--steps", type=int, default=500000, help="Total exploration steps"
    )
    parser.add_argument(
        "--explore-steps",
        type=int,
        default=100,
        help="Random steps per exploration from a cell",
    )
    parser.add_argument(
        "--sticky-prob",
        type=float,
        default=0.85,
        help="Probability of repeating previous action",
    )
    parser.add_argument(
        "--output", default="output/mo5/yeti/go_explore", help="Output directory"
    )
    parser.add_argument(
        "--seed-archive",
        default=None,
        help="Optional path to a prior archive.pkl; its cells are added to "
        "the in-memory archive as extra starting points.",
    )
    parser.add_argument(
        "--seed-min-cp",
        type=int,
        default=None,
        help="If --seed-archive is set, skip cells whose CP level "
        "(4 - fruits_remaining) is below this value. Useful for "
        "'explore only from CP2+' runs.",
    )
    parser.add_argument(
        "--seed-validate",
        action="store_true",
        default=True,
        help="Validate each seeded state and drop non-viable ones. Default: on.",
    )
    parser.add_argument(
        "--no-seed-validate",
        action="store_false",
        dest="seed_validate",
        help="Disable validation of seeded states.",
    )
    args = parser.parse_args()
    explore(args)


if __name__ == "__main__":
    main()
