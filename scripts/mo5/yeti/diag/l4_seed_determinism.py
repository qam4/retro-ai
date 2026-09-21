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
from pathlib import Path

from retro_ai.games import yeti
from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import get_level_map
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


def noop_survival(stack, ifc, state, cap=150):
    """Frames survived from ``state`` while holding NOOP. ``cap`` means 'still alive'.

    A pure safety measure: it removes the agent's agency, so what remains is whether
    the state itself is survivable. A state about to be squashed dies no matter what.
    """
    ifc.load_state(bytes(state))
    stack.preprocessed.notify_state_loaded()
    stack.gym.step([0, 0, 0])
    for i in range(cap):
        stack.gym.step([0, 0, 0])
        if yeti.is_dead(ifc):
            return i
    return cap


def _quantiles(v):
    s = sorted(v)
    n = len(s)
    return s[0], s[n // 4], s[n // 2], s[(3 * n) // 4], s[-1]


def arrival_trend(args, stack, ifc, wps, models):
    """Track ARRIVAL SURVIVABILITY across a run's snapshots.

    THE QUESTION THIS SETTLES. L4 stops at one transition: the agent reaches floor 11
    and is crushed before it can start the floor-12 jump. The jump needs an 8-step
    approach, and what decides the outcome is how many frames of life the agent has WHEN
    IT ARRIVES -- measured by holding NOOP from the arrival state. Cold runs arrive with
    3-5 frames; the runs that got past floor 11 arrived with 12.
    That leaves a fork worth real compute:
        would a cold run get there if we simply ran it LONGER?
    A 15M run costs ~7h, and the warm alternative carries ~60M of cumulative training
    across four chained runs, so the honest comparison is expensive. This is the cheap
    version: if the quantity is CLIMBING across snapshots the skill is being acquired
    and more steps plausibly reach the threshold; if it is FLAT the run is not acquiring
    it and no amount of scaling will.
    Companion to l3_link_over_snapshots.py, which does the same across-snapshots trick
    for a link's reach. The final snapshot is routinely a trough on these runs, so a
    single endpoint cannot answer this.
    """
    from retro_ai.training.targets import build_targets

    tg = {t.id: t for t in build_targets(4)}
    snaps = sorted(
        Path(args.run).glob("snapshots/model_*_steps.zip"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not snaps:
        raise SystemExit(f"no snapshots under {args.run}/snapshots")
    step = max(1, len(snaps) // args.over_snapshots)
    picked = snaps[::step][: args.over_snapshots]
    if snaps[-1] not in picked:
        picked.append(snaps[-1])
    pool = [p.strip() for p in args.pools.split(",")][0]
    floor = tg[pool].floor
    y_t = {p.floor: p.y for p in get_level_map(4).platforms}[floor]
    start_bytes = Path("output/mo5/yeti/level4/level4_start.sav").read_bytes()
    print(f"\n  arrival survivability at {pool} (floor {floor}), from reset")
    print("  NOOP frames of life on arrival; the floor-12 approach needs 8\n")
    print(f"  {'step':>12} {'n':>4} {'min':>4} {'median':>7} {'max':>4}")
    from stable_baselines3 import PPO

    trend = []
    for sp in picked:
        model = PPO.load(str(sp), device="cpu")
        arrivals = []
        for _ep in range(args.vs_reset or 25):
            ifc.load_state(start_bytes)
            stack.preprocessed.notify_state_loaded()
            obs, *_ = stack.gym.step([0, 0, 0])
            for _s in range(900):
                a, _ = model.predict(obs, deterministic=False)
                obs, *_rest = stack.gym.step([int(v) for v in a])
                if (
                    ifc.read_ram_byte(yeti.Y_ADDR) == y_t
                    and ifc.read_ram_byte(yeti.POSE_ADDR) in yeti.SURFACE_POSES
                ):
                    arrivals.append(stack.base.save_state())
                    break
                if yeti.is_dead(ifc):
                    break
        st = int(sp.stem.split("_")[1])
        if not arrivals:
            print(f"  {st:>12,} {0:>4}   -- never reached floor {floor} --")
            continue
        v = [noop_survival(stack, ifc, s, args.noop_cap) for s in arrivals]
        lo, _q1, med, _q3, hi = _quantiles(v)
        trend.append((st, med))
        print(f"  {st:>12,} {len(v):>4} {lo:>4} {med:>7} {hi:>4}", flush=True)
    if len(trend) >= 4:
        # FIRST-vs-LAST IS THE WRONG TEST and printed a misleading verdict once. On a
        # cold run the early snapshots cannot reach the waypoint at all, so the first
        # measurable point is near zero and any plateau above it reads as "rising".
        # v16c went 0 (3.7M) -> 5 (5.5M) -> 3 for the remaining 10M steps: acquired
        # early, then flat for two thirds of the run, nowhere near the threshold.
        # Compare the LAST THIRD against the middle third instead, and say plainly
        # whether the plateau clears the bar.
        half = len(trend) // 3
        early = sorted(m for _s, m in trend[half : 2 * half]) or [trend[0][1]]
        late = sorted(m for _s, m in trend[-half:]) or [trend[-1][1]]
        e, la = early[len(early) // 2], late[len(late) // 2]
        best = max(m for _s, m in trend)
        print(
            f"\n  median over the middle third {e}, over the last third {la}, "
            f"best single snapshot {best}."
        )
        if la > e:
            print(
                "  STILL RISING: the skill is being acquired; a longer run may reach "
                "the threshold."
            )
        else:
            print(
                f"  PLATEAUED at ~{la}: the run stopped improving on this well before "
                "the end, so\n  scaling it cannot reach the threshold. The difference "
                "must come from elsewhere."
            )
    return None


def noop_safety(args, stack, ifc, wps, models):
    """Is the seed pool as dangerous as what the policy actually arrives into?

    CAVEAT ON WHAT THIS MEASURES. Survival under NOOP is PHASE-CONDITIONAL: it asks
    whether a hazard reached THIS state within the cap, which depends on where the
    hazard happened to be. It is NOT structural exposure. A fully exposed position
    scores the cap if the hazard is far away in every state sampled, and a pool holds
    only states that passed `admit_requires_survival`, i.e. a biased sample of phases.
    Read a low number as 'this state was doomed'; do NOT read a high number as 'this
    position is safe'. Raise --noop-cap past the hazard's period before concluding.
    """
    from retro_ai.training.targets import build_targets

    tg = {t.id: t for t in build_targets(4)}
    CAP = args.noop_cap
    for pool in [p.strip() for p in args.pools.split(",")]:
        if pool not in wps or not wps[pool][0]:
            print(f"\n{pool}: EMPTY")
            continue
        seeds = list(wps[pool][0])[: args.n]
        pool_v = [noop_survival(stack, ifc, e[2], CAP) for e in seeds]
        print(f"\n{'=' * 70}\n{pool}: NOOP frames survived (cap {CAP})\n{'=' * 70}")
        lo, q1, med, q3, hi = _quantiles(pool_v)
        died = sum(1 for v in pool_v if v < CAP)
        print(
            f"  POOL       n={len(pool_v):<4} min {lo:>3}  q1 {q1:>3}  median {med:>3}"
            f"  q3 {q3:>3}  max {hi:>3}   died within cap: {died}/{len(pool_v)}"
        )
        if not args.vs_reset:
            continue
        floor = tg[pool].floor if pool in tg else None
        y_t = {p.floor: p.y for p in get_level_map(4).platforms}.get(floor)
        if y_t is None:
            print(f"  (cannot locate floor for {pool}; skipping --vs-reset)")
            continue
        label, model = models[0]
        start_bytes = Path("output/mo5/yeti/level4/level4_start.sav").read_bytes()
        arrivals = []
        for _ep in range(args.vs_reset):
            ifc.load_state(start_bytes)
            stack.preprocessed.notify_state_loaded()
            obs, *_ = stack.gym.step([0, 0, 0])
            for _s in range(900):
                a, _ = model.predict(obs, deterministic=False)
                obs, *_rest = stack.gym.step([int(v) for v in a])
                if (
                    ifc.read_ram_byte(yeti.Y_ADDR) == y_t
                    and ifc.read_ram_byte(yeti.POSE_ADDR) in yeti.SURFACE_POSES
                ):
                    arrivals.append(stack.base.save_state())
                    break
                if yeti.is_dead(ifc):
                    break
        if not arrivals:
            print(
                f"  RESET      no arrival on floor {floor} in {args.vs_reset} episodes"
            )
            continue
        res_v = [noop_survival(stack, ifc, s, CAP) for s in arrivals]
        lo, q1, med, q3, hi = _quantiles(res_v)
        died = sum(1 for v in res_v if v < CAP)
        print(
            f"  RESET      n={len(res_v):<4} min {lo:>3}  q1 {q1:>3}  median {med:>3}"
            f"  q3 {q3:>3}  max {hi:>3}   died within cap: {died}/{len(res_v)}"
            f"   ({label})"
        )
        pm = sorted(pool_v)[len(pool_v) // 2]
        rm = sorted(res_v)[len(res_v) // 2]
        print(
            f"\n  median pool {pm} vs median reset arrival {rm}."
            + (
                "  The POOL IS SAFER than reality: `admit_requires_survival` selected "
                "for\n  benign hazard phases, so practising there under-trains the "
                "situation the\n  policy actually faces."
                if pm > rm
                else "  The pool is not systematically safer than reality."
            )
        )
    return None


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
    ap.add_argument(
        "--noop-safety",
        action="store_true",
        help="measure how many NOOP frames each seed survives, as a proxy for how safe "
        "the state is -- a state about to be hit by a hazard dies whatever the agent "
        "does. Add --vs-reset to test whether the POOL is representative of what the "
        "policy actually meets from reset: `admit_requires_survival` only admits "
        "states the agent survived, so the pool is selected FOR safe hazard phases "
        "and may be systematically easier than reality.",
    )
    ap.add_argument(
        "--over-snapshots",
        type=int,
        default=0,
        metavar="N",
        help="sweep N snapshots evenly across the run and report --vs-reset arrival "
        "survivability for each, i.e. IS THE RUN LEARNING THIS AT ALL. Answers 'would "
        "more steps get there?' without spending them: a rising trend means the skill "
        "is being acquired slowly and a longer run is worth trying; a flat trend means "
        "it is not being acquired and scaling the run cannot help.",
    )
    ap.add_argument(
        "--noop-cap",
        type=int,
        default=150,
        help="frames to hold NOOP before calling a state survivable. Too small and "
        "the measurement is truncated: a bimodal result (a few early deaths, the "
        "rest at exactly the cap) means the cap is shorter than the hazard cycle.",
    )
    ap.add_argument(
        "--vs-reset",
        type=int,
        default=0,
        metavar="N",
        help="roll the policy from reset N times and measure the same noop-survival "
        "for the state at its FIRST grounded arrival on the pool's floor",
    )
    ap.add_argument(
        "--models",
        help="comma-separated .zip paths to compare on the IDENTICAL seed set, e.g. "
        "'<v13>/best/best_model.zip,<v16c>/best/best_model.zip'. Overrides --model. "
        "The seeds come from --run for every policy, which is the point: two runs' "
        "own pools hold different kangaroo phases, so comparing them would confound "
        "the policy with the state it was handed.",
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
    if args.models:
        paths = [p.strip() for p in args.models.split(",") if p.strip()]
    else:
        paths = [
            (
                os.path.join(args.run, "final_model.zip")
                if args.model == "final"
                else os.path.join(
                    args.run, "snapshots", f"model_{args.model}_steps.zip"
                )
            )
        ]
    models = []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"model not found: {p}")
        # label by the run directory, which is what distinguishes them
        label = (
            Path(p).parent.parent.name
            if Path(p).parent.name == "best"
            else Path(p).stem
        )
        print(f"model: {p}")
        models.append((label, PPO.load(p, device="auto")))
    print(f"seeds come from: {args.run}")

    if args.over_snapshots:
        return arrival_trend(args, stack, ifc, wps, models)
    if args.noop_safety:
        return noop_safety(args, stack, ifc, wps, models)

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
        summary = []
        for label, model in models:
            print(
                f"\n{'=' * 70}\n{pool} / {label}: {len(seeds)} seeds x "
                f"{args.repeats} repeats\n{'=' * 70}"
            )
            print(
                f"  {'seed':>5} {'x':>4} {'y':>4} {'pose':>5}  wins/repeats  lifetimes"
            )
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
                    f"  -> {100 * frac:.0f}% of the variance is attributable to "
                    "WHICH SEED"
                )
            summary.append((label, mean, dict(klass), between, within))
        print(
            "\n  reading: mostly ALWAYS-win/ALWAYS-lose + high between-seed variance"
            "\n  => the pool is phase-determined and seeding there teaches little."
        )
        if len(summary) > 1:
            print(
                f"\n  {'=' * 66}\n  SAME {len(seeds)} SEEDS, DIFFERENT POLICIES\n"
                f"  {'=' * 66}"
            )
            print(
                f"  {'policy':>44} {'mean':>6} {'always-win':>11} {'mixed':>6}"
                f" {'always-lose':>12}"
            )
            for label, mean, klass, _b, _w in summary:
                print(
                    f"  {label:>44} {mean:>6.2f} {klass.get('ALWAYS-win', 0):>11}"
                    f" {klass.get('MIXED', 0):>6} {klass.get('ALWAYS-lose', 0):>12}"
                )


if __name__ == "__main__":
    main()
