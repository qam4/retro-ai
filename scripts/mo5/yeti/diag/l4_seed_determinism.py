#!/usr/bin/env python3
"""Is the outcome from a seed decided by the SEED or by the POLICY?

If a Lclimb3_top / Low1_launch seed's fate is fixed by the kangaroo phase baked into
the saved state, then that pool is a bag of coin flips: no action sequence changes the
result, the agent is rewarded and punished for reasons unrelated to what it did, and
seeding there teaches nothing. If instead the same seed sometimes succeeds and
sometimes fails under stochastic sampling, the policy has agency and the pool is
legitimate practice.

Method: replay each seed R times with a stochastic policy and record the outcome each
time. Then split the total variance:

  BETWEEN-seed  -- differences in per-seed success rate  => the SEED decides
  WITHIN-seed   -- the same seed landing sometimes and not others => the POLICY decides

Reported as: how many seeds are ALWAYS-win, ALWAYS-lose, or MIXED. A pool that is
mostly always-win/always-lose is phase-determined.
"""
from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

from retro_ai.games import yeti
from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

F12_Y, F12_X = 70, (44, 56)
F11_Y, F11_X = 78, (60, 78)


def is_clean_f11(x, y, pose):
    """A seed genuinely STANDING on the launch platform f11.

    The Low1_launch capture box is (60, 78) with jump tolerance 6, so it spans
    y 72..84 and pose 8 is seedable -- 43/100 of the pool is mid-ladder at y82,
    BELOW the platform, where the jump is 0/7 scriptable. Those states cannot
    answer "does the policy have agency here", so they are filterable out.
    """
    return (
        pose in yeti.SURFACE_POSES and abs(y - F11_Y) <= 2 and F11_X[0] <= x <= F11_X[1]
    )


def landed(res):
    for px, py in res.positions:
        x = px // 4
        if abs(py - F12_Y) <= 2 and F12_X[0] <= x <= F12_X[1]:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", default="output/mo5/yeti/training/yeti_curriculum_l4_v4_15m"
    )
    ap.add_argument("--pools", default="Lclimb3_top,Low1_launch")
    ap.add_argument("--n", type=int, default=20, help="seeds per pool")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument(
        "--model",
        default="final",
        help="'final' or a snapshot step (e.g. 14000000). Prefer a snapshot: the "
        "final model is routinely degraded, and a bad policy loses from every seed, "
        "which fakes a phase-determined verdict.",
    )
    ap.add_argument(
        "--clean-only",
        action="store_true",
        help="keep only seeds genuinely standing on f11 (drops the mid-ladder y82 "
        "captures, where the move is not executable at all)",
    )
    args = ap.parse_args()

    with open(os.path.join(args.run, "checkpoints.pkl"), "rb") as fh:
        wps = pickle.load(fh)["waypoints"]

    cfg = EnvConfig(
        profile="yeti_fruit_level4",
        action_mode="joystick",
        max_steps=1500,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level4", cfg)
    ifc = stack.base._interface
    stack.gym.reset()
    model_path = (
        os.path.join(args.run, "final_model.zip")
        if args.model == "final"
        else os.path.join(args.run, "snapshots", f"model_{args.model}_steps.zip")
    )
    if not os.path.exists(model_path):
        raise SystemExit(f"model not found: {model_path}")
    print(f"model: {model_path}")
    model = PPO.load(model_path, device="auto")

    for pool in [p.strip() for p in args.pools.split(",")]:
        if pool not in wps or not wps[pool][0]:
            print(f"\n{pool}: EMPTY")
            continue
        candidates = wps[pool][0]
        if args.clean_only:
            kept = []
            for e in candidates:
                ifc.load_state(bytes(e[2]))
                if is_clean_f11(
                    ifc.read_ram_byte(yeti.X_ADDR),
                    ifc.read_ram_byte(yeti.Y_ADDR),
                    ifc.read_ram_byte(yeti.POSE_ADDR),
                ):
                    kept.append(e)
            print(f"\n{pool}: clean-only filter kept {len(kept)}/{len(candidates)}")
            candidates = kept
        seeds = candidates[: args.n]
        print(
            f"\n{'=' * 70}\n{pool}: {len(seeds)} seeds x {args.repeats} repeats"
            f"\n{'=' * 70}"
        )
        print(f"  {'seed':>5} {'x':>4} {'y':>4} {'pose':>5}  wins/repeats  lifetimes")
        klass = Counter()
        rates = []
        for i, e in enumerate(seeds):
            ifc.load_state(bytes(e[2]))
            sx = ifc.read_ram_byte(yeti.X_ADDR)
            sy = ifc.read_ram_byte(yeti.Y_ADDR)
            sp = ifc.read_ram_byte(yeti.POSE_ADDR)
            wins, lens = 0, []
            for _r in range(args.repeats):
                res = rollout_episode(
                    stack,
                    model,
                    level=4,
                    fruits_total=1,
                    start_state=bytes(e[2]),
                    max_steps=args.max_steps,
                    stall_threshold=10**9,
                    deterministic=False,
                    keep_frames=False,
                    reset_env=False,
                )
                wins += landed(res)
                lens.append(res.length)
            rate = wins / args.repeats
            rates.append(rate)
            k = (
                "ALWAYS-win"
                if wins == args.repeats
                else ("ALWAYS-lose" if wins == 0 else "MIXED")
            )
            klass[k] += 1
            print(
                f"  {i:>5} {sx:>4} {sy:>4} {sp:>5}  {wins:>3}/{args.repeats}"
                f"        {sorted(lens)}"
            )
        n = len(seeds)
        mean = sum(rates) / n
        # within-seed (binomial) vs between-seed variance of the success indicator
        within = sum(p * (1 - p) for p in rates) / n
        between = sum((p - mean) ** 2 for p in rates) / n
        print(f"\n  classes: {dict(klass)}")
        print(f"  mean success {mean:.2f}")
        print(f"  BETWEEN-seed variance (seed decides):   {between:.4f}")
        print(f"  WITHIN-seed  variance (policy decides): {within:.4f}")
        if between + within > 0:
            frac = between / (between + within)
            print(
                f"  -> {100 * frac:.0f}% of the variance is attributable to WHICH SEED"
            )
        print(
            "  reading: mostly ALWAYS-win/ALWAYS-lose + high between-seed variance"
            "\n  => the pool is phase-determined and seeding there teaches little."
        )


if __name__ == "__main__":
    main()
