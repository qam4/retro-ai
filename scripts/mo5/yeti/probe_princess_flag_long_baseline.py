#!/usr/bin/env python3
"""Long-baseline confidence check for the princess-touch flag at byte
11050.

Two scenarios:

  A. CP4 seed rollout (fruits=0, agent already past last fruit).
     Random policy for 5000 frames per seed across all seeds in the
     pool. The princess is rarely reachable from these seeds, so any
     11050 0->1 here is a real touch and we log score/lives at the
     transition.

  B. Trained policy rollout from CP0 (fresh reset, fruits=4). Many
     fruit pickups, deaths, jumps. We log every frame where 11050
     changes value and confirm it never flips to 1 outside a princess
     touch.

Outputs a single PASS/FAIL summary plus the list of flag transitions
observed.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build" / "ci-linux"))

from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402

X_ADDR = 11090
Y_ADDR = 11089
LIVES_ADDR = 11095
FRUITS_ADDR = 11055
SCORE_HI = 11093
SCORE_LO = 11094
PRINCESS_FLAG_ADDR = 11050


def env_factory():
    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=10_000,
        stall_threshold=10_000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", cfg)
    base = stack.base
    base.reset(seed=0)
    return stack, base


def read(iface):
    return {
        "x": iface.read_ram_byte(X_ADDR),
        "y": iface.read_ram_byte(Y_ADDR),
        "fr": iface.read_ram_byte(FRUITS_ADDR),
        "lv": iface.read_ram_byte(LIVES_ADDR),
        "score": (iface.read_ram_byte(SCORE_HI) << 8) | iface.read_ram_byte(SCORE_LO),
        "flag": iface.read_ram_byte(PRINCESS_FLAG_ADDR),
    }


def scenario_a_cp4_seeds(seed_pool_path: Path, frames_per_seed: int = 5000):
    print(f"\n=== Scenario A: CP4-seed random rollout x {frames_per_seed} frames ===")
    if not seed_pool_path.exists():
        print(f"  skipped: {seed_pool_path} missing")
        return [], 0

    with seed_pool_path.open("rb") as f:
        pool = pickle.load(f)
    cp_buckets = pool["checkpoints"]
    cp4_seeds = cp_buckets[4] if isinstance(cp_buckets, list) else cp_buckets.get(4, [])
    if not cp4_seeds:
        print("  skipped: no CP4 seeds in pool")
        return [], 0

    print(f"  seeds: {len(cp4_seeds)}, frames per seed: {frames_per_seed}")
    transitions = []
    total_frames = 0
    rng = np.random.default_rng(0)

    for sidx, seed_state in enumerate(cp4_seeds):
        stack, base = env_factory()
        gym_env = stack.gym
        iface = base._interface
        iface.load_state(seed_state)
        for _ in range(5):
            gym_env.step([0, 0, 0])

        prev_flag = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        for f in range(frames_per_seed):
            # Random joystick action.
            dx = rng.integers(-1, 2)
            dy = rng.integers(-1, 2)
            fire = rng.integers(0, 2)
            gym_env.step([int(fire), int(dx), int(dy)])
            cur_flag = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            if cur_flag != prev_flag:
                s = read(iface)
                transitions.append(("A", sidx, f, prev_flag, cur_flag, s))
                if cur_flag == 1 and prev_flag == 0:
                    print(
                        f"  seed {sidx} frame {f}: 0->1 transition. "
                        f"x={s['x']} y={s['y']} fr={s['fr']} "
                        f"lv={s['lv']} score={s['score']}"
                    )
                prev_flag = cur_flag
            total_frames += 1
            # If lives reached 0, stop early.
            if iface.read_ram_byte(LIVES_ADDR) == 0:
                break
    print(f"  total frames: {total_frames}, transitions: {len(transitions)}")
    return transitions, total_frames


def scenario_b_random_from_cp0(frames: int = 5000):
    print(f"\n=== Scenario B: random rollout from CP0 x {frames} frames ===")
    stack, base = env_factory()
    gym_env = stack.gym
    iface = base._interface
    rng = np.random.default_rng(1)

    transitions = []
    fruit_pickups = []
    prev_flag = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
    prev_fruits = iface.read_ram_byte(FRUITS_ADDR)
    deaths = 0

    for f in range(frames):
        dx = rng.integers(-1, 2)
        dy = rng.integers(-1, 2)
        fire = rng.integers(0, 2)
        gym_env.step([int(fire), int(dx), int(dy)])
        cur_flag = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        cur_fruits = iface.read_ram_byte(FRUITS_ADDR)
        cur_lives = iface.read_ram_byte(LIVES_ADDR)

        if cur_flag != prev_flag:
            s = read(iface)
            transitions.append(("B", f, prev_flag, cur_flag, s))
            print(f"  flag {prev_flag}->{cur_flag} at frame {f}: {s}")
            prev_flag = cur_flag

        if cur_fruits < prev_fruits:
            s = read(iface)
            fruit_pickups.append((f, prev_fruits, cur_fruits, s["flag"]))
            prev_fruits = cur_fruits

        if cur_lives == 0:
            deaths += 1
            # Reset and continue to keep the rollout going.
            base.reset(seed=deaths)
            iface = base._interface
            prev_flag = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            prev_fruits = iface.read_ram_byte(FRUITS_ADDR)

    print(
        f"  total frames: {frames}, deaths: {deaths}, "
        f"fruit pickups: {len(fruit_pickups)}, transitions: {len(transitions)}"
    )
    if fruit_pickups:
        print(
            "  fruit pickup events (frame, prev_fruits -> cur_fruits, flag at pickup):"
        )
        for fp in fruit_pickups[:20]:
            print(f"    frame {fp[0]}: {fp[1]}->{fp[2]}  flag={fp[3]}")
        if len(fruit_pickups) > 20:
            print(f"    ... and {len(fruit_pickups) - 20} more")
    return transitions, fruit_pickups


def main() -> int:
    seed_path = ROOT / "output/mo5/yeti/seeds/v9_v4_cp4enriched.pkl"

    a_trans, a_frames = scenario_a_cp4_seeds(seed_path, frames_per_seed=5000)
    b_trans, b_picks = scenario_b_random_from_cp0(frames=5000)

    print("\n=== Summary ===")
    print(f"Scenario A: {a_frames} frames, {len(a_trans)} flag transitions")
    print(
        f"Scenario B: 5000 frames, {len(b_trans)} flag transitions, "
        f"{len(b_picks)} fruit pickups"
    )

    # Pass criteria:
    # 1. In scenario B (fruits never 0 most of the time and no
    #    princess reachable by random play), no 0->1 transitions.
    # 2. Fruit pickups never coincide with flag=1.
    a_zero_to_one = [t for t in a_trans if t[3] == 0 and t[4] == 1]
    b_zero_to_one = [t for t in b_trans if t[2] == 0 and t[3] == 1]
    pickups_with_flag = [fp for fp in b_picks if fp[3] != 0]

    print(f"\nScenario A 0->1 transitions: {len(a_zero_to_one)}")
    print(f"Scenario B 0->1 transitions: {len(b_zero_to_one)} (expected 0)")
    print(f"Fruit pickups with flag != 0: {len(pickups_with_flag)} (expected 0)")

    if len(b_zero_to_one) == 0 and len(pickups_with_flag) == 0:
        print("\nPASS: byte 11050 is a clean princess-touch signal.")
        return 0
    print("\nFAIL: byte 11050 fired in non-princess-touch contexts.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
