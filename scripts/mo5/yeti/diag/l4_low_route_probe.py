#!/usr/bin/env python3
"""Where does the low route actually break, and what poses do the rope/spring use?

v4's route table says Low1_launch reach 0.54 but Low1 0.00, and Low2_launch holds
100 seeds while Low2 got 1 capture. Two candidate walls, one step apart:

    A.  f11 -> f12   plain JUMP (tiles 60->59)   reached as `Low1`
    B.  f12 -> f13   ROPE       (tiles 54->53)   reached as `Low2`

PART 1 asks the question that cracked `Step`: is the move EXECUTABLE from states we
actually hold? Script it and count. High scripted success => the spot is benign and
the 0.00 is a learning/allocation problem. Low success => a real skill or a
mis-modelled mechanic.

PART 2 uses the fact that the SAME policy already crosses rope 1 (0.90) and the
spring (0.89). Replay it from Rope1_launch / Spring_launch seeds and dump the
pose/x/y trace, so we finally have measured poses for a rope carry and a spring
bounce instead of guessing. That is what L3's escalator needed (pose 13) before its
seeds stopped being doomed.

Geometry (pixels -> x_ram = (px - 8) // 4):
    f11 P12  y78  x 248-320   left edge  248px  -> x_ram 60
    f12 P11  y70  x 184-232   right edge 232px  -> x_ram 56
    f13 P10  y70  x   0-128   right edge 128px  -> x_ram 30
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

# `go_explore` is a sibling of this file's PARENT (scripts/mo5/yeti), not of this file.
# When this script lived in debug/ it was run with that directory already on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from go_explore import make_env  # noqa: E402
from retro_ai.games import yeti  # noqa: E402

NEUTRAL = [0, 0, 0]
RIGHT, LEFT, UP = [0, 1, 0], [0, 2, 0], [1, 0, 0]
JUMP, JUMP_R, JUMP_L = [0, 0, 1], [0, 1, 1], [0, 2, 1]
ACTION_NAME = {
    (0, 0, 0): "-",
    (0, 1, 0): "R",
    (0, 2, 0): "L",
    (1, 0, 0): "U",
    (0, 0, 1): "JMP",
    (0, 1, 1): "JMP-R",
    (0, 2, 1): "JMP-L",
}

F11_Y, F12_Y, F13_Y = 78, 70, 70
F12_XMAX = 58  # x_ram: f12 spans 184-232px -> x_ram 44..56 (+tol)
F13_XMAX = 32  # x_ram: f13 spans 0-128px  -> x_ram 0..30 (+tol)


def dead(ifc):
    return ifc.read_ram_byte(yeti.DEATH_FLAG_ADDR) == yeti.DEATH_FLAG_VALUE


def read(ifc):
    return (
        ifc.read_ram_byte(yeti.X_ADDR),
        ifc.read_ram_byte(yeti.Y_ADDR),
        ifc.read_ram_byte(yeti.POSE_ADDR),
    )


def load_pool(run, name):
    with open(os.path.join(run, "checkpoints.pkl"), "rb") as fh:
        wps = pickle.load(fh)["waypoints"]
    if name not in wps:
        return []
    return wps[name][0]


def on_platform(x, y, y_target, x_max):
    """Landed: at the target platform's standing y, within its x extent."""
    return abs(y - y_target) <= 2 and x <= x_max


def scripted(env, ifc, state, plan, y_target, x_max, budget=120):
    """Run `plan(i, x, y, pose) -> action` and report whether we land."""
    ifc.load_state(state)
    for i in range(budget):
        if dead(ifc):
            return False, "died"
        x, y, pose = read(ifc)
        if on_platform(x, y, y_target, x_max):
            return True, "landed"
        env.step(plan(i, x, y, pose))
    return False, "timeout"


# --- the two scripted plans -------------------------------------------------
# A: walk LEFT to f11's edge (x_ram 60), then JUMP-LEFT. Repeat the jump every
#    16 frames so a mistimed first attempt is retried rather than counted a fail.
def plan_jump_left_from_edge(edge_x):
    def plan(i, x, y, pose):
        if x > edge_x:
            return LEFT
        return JUMP_L if (i % 16) < 3 else NEUTRAL

    return plan


# B: same shape at the rope. The rope oscillates, so the ONLY variable a script can
#    sweep is the phase it jumps on -- hence --phase, swept by the caller.
def plan_rope(edge_x, phase, period=24):
    def plan(i, x, y, pose):
        if x > edge_x:
            return LEFT
        return JUMP_L if (i % period) == phase else NEUTRAL

    return plan


def part1(env, ifc, run, n):
    print("=" * 72)
    print("PART 1 -- is the move executable from states we hold?")
    print("=" * 72)

    for pool_name, label, edge_x, y_target, x_max in [
        ("Low1_launch", "A. f11 -> f12  plain JUMP", 60, F12_Y, F12_XMAX),
        ("Low2_launch", "B. f12 -> f13  ROPE", 46, F13_Y, F13_XMAX),
    ]:
        states = load_pool(run, pool_name)
        if not states:
            print(f"\n{label}\n  pool {pool_name}: EMPTY, cannot probe")
            continue
        sel = states[:n]
        print(
            f"\n{label}   (pool {pool_name}: {len(states)} states, probing {len(sel)})"
        )

        # untimed attempt
        ok = 0
        why = Counter()
        for e in sel:
            good, reason = scripted(
                env, ifc, bytes(e[2]), plan_jump_left_from_edge(edge_x), y_target, x_max
            )
            ok += good
            why[reason] += 1
        print(
            f"  scripted walk-left-then-JUMP-LEFT: {ok}/{len(sel)}"
            f"  ({100 * ok / len(sel):.0f}%)   {dict(why)}"
        )

        # phase sweep: best single fixed phase, and per-seed best over all phases
        best_phase, best_ok = None, -1
        per_seed_any = [False] * len(sel)
        for phase in range(24):
            hits = 0
            for j, e in enumerate(sel):
                good, _ = scripted(
                    env, ifc, bytes(e[2]), plan_rope(edge_x, phase), y_target, x_max
                )
                hits += good
                per_seed_any[j] |= good
            if hits > best_ok:
                best_ok, best_phase = hits, phase
        print(
            f"  best single fixed phase ({best_phase:2d}/24):"
            f" {best_ok}/{len(sel)}  ({100 * best_ok / len(sel):.0f}%)"
        )
        solved = sum(per_seed_any)
        print(
            f"  solvable by SOME phase:            {solved}/{len(sel)}"
            f"  ({100 * solved / len(sel):.0f}%)"
        )

        # NOOP survival, to separate "doomed seed" from "hard move"
        lives = []
        for e in sel:
            ifc.load_state(bytes(e[2]))
            life = 150
            for s in range(150):
                env.step(NEUTRAL)
                if dead(ifc):
                    life = s
                    break
            lives.append(life)
        lives.sort()
        print(
            f"  NOOP lifetime: min {lives[0]}  median {lives[len(lives) // 2]}"
            f"  max {lives[-1]}   survived 150: "
            f"{sum(1 for v in lives if v >= 150)}/{len(lives)}"
        )


def part2(env, ifc, run, model_path, n, trace_len):
    print()
    print("=" * 72)
    print("PART 2 -- measured poses on the crossings the policy ALREADY makes")
    print("=" * 72)
    if not os.path.exists(model_path):
        print(f"  model not found: {model_path}")
        return
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="auto")

    for pool_name, label in [
        ("Rope1_launch", "ROPE 1  f6 -> f7  (policy reach 0.90)"),
        ("Spring_launch", "SPRING  f8 -> f9  (policy reach 0.89)"),
    ]:
        states = load_pool(run, pool_name)
        if not states:
            print(f"\n{label}\n  pool {pool_name}: EMPTY")
            continue
        sel = states[:n]
        print(f"\n{label}   (pool {pool_name}, {len(sel)} seeds)")
        airborne_poses = Counter()
        shown = 0
        for e in sel:
            obs, _ = env.reset()
            ifc.load_state(bytes(e[2]))
            obs = env._get_obs() if hasattr(env, "_get_obs") else obs
            rows = []
            for _ in range(trace_len):
                act, _ = model.predict(obs, deterministic=False)
                a = list(np.atleast_1d(act).astype(int))
                obs, _r, term, trunc, _i = env.step(a)
                x, y, pose = read(ifc)
                rows.append((tuple(a), x, y, pose, dead(ifc)))
                if pose not in yeti.SURFACE_POSES:
                    airborne_poses[pose] += 1
                if term or trunc or dead(ifc):
                    break
            if shown < 2:
                shown += 1
                print(f"    trace {shown}:  act      x    y  pose")
                for a, x, y, pose, dd in rows:
                    tag = " DEAD" if dd else ""
                    surf = "" if pose in yeti.SURFACE_POSES else "  <-airborne"
                    nm = ACTION_NAME.get(a, str(a))
                    print(f"           {nm:>7}  {x:3d}  {y:3d}   {pose:3d}{surf}{tag}")
        print(
            f"  non-surface poses seen: "
            f"{dict(sorted(airborne_poses.items(), key=lambda kv: -kv[1]))}"
        )
        print("  (known: 9,10 jump  11 fall  12 death-anim; L3 escalator ride = 13)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", default="output/mo5/yeti/training/yeti_curriculum_l4_v4_15m"
    )
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--trace-len", type=int, default=40)
    ap.add_argument("--skip-poses", action="store_true")
    args = ap.parse_args()

    env = make_env("yeti_fruit_level4")
    env.reset()
    ifc = env._interface

    part1(env, ifc, args.run, args.n)
    if not args.skip_poses:
        part2(
            env,
            ifc,
            args.run,
            os.path.join(args.run, "final_model.zip"),
            n=6,
            trace_len=args.trace_len,
        )


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (used in part2)

    main()
