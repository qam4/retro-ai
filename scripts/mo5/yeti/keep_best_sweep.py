#!/usr/bin/env python3
"""Keep-best capture: eval training snapshots from reset and retain the
best policy (by princess-from-reset rate).

Why this exists (experiment 003, H-U): a single PPO policy on Yeti
oscillates — reach-4/princess swing the full range across snapshots and
the *final* model is usually degraded. The good policy is a transient
peak (v14 hit 58%% princess at 12.75M while the snapshots on either side
read ~0%%). Training metrics can't identify it (most training episodes
don't start from reset; the live signal is a noisy EMA), so we must eval
frozen snapshots from reset and keep the best.

This runs each eval as a SEPARATE PROCESS (scripts/mo5/yeti/eval_from_reset.py).
The Crayon emulator keeps in-process global state, so eval must not share
a process with training; a subprocess is fully isolated. Defaults to CPU
so it is safe to run *alongside* a GPU training job (pass --device gpu to
use the GPU when nothing else is training).

Usage
-----
One-shot (eval every snapshot present, keep the best)::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/mo5/yeti/keep_best_sweep.py \\
        --snapshots-dir output/mo5/yeti/training/<run>/snapshots

Watch a live run (poll for new snapshots, stop after idle)::

    ... keep_best_sweep.py --snapshots-dir <run>/snapshots --watch

The best policy is copied to ``<best-dir>/best_model.zip`` with
``best_meta.json``; per-snapshot results accumulate in
``<best-dir>/sweep_state.json`` so re-runs skip already-eval'd snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter

_STEP_RE = re.compile(r"model_(\d+)_steps\.zip$")


def _snapshots(snap_dir):
    """Return [(step, path)] sorted by step. Empty if the dir does not exist yet.

    NOT AN EXISTENCE CHECK DRESSED UP AS ONE. The trainer creates `snapshots/` lazily,
    when it writes its first snapshot, so launching this alongside a fresh run -- which
    is the documented way to use `--watch` -- races it. `os.listdir` raised
    FileNotFoundError and the evaluator died 10 s in, leaving a 6M run training for
    1h40m with no referee and `on_regression` unable to fire. Returning empty lets the
    watch loop do what it is for: wait.
    """
    if not os.path.isdir(snap_dir):
        return []
    out = []
    for name in os.listdir(snap_dir):
        m = _STEP_RE.search(name)
        if m:
            out.append((int(m.group(1)), os.path.join(snap_dir, name)))
    out.sort()
    return out


def _load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"evaluated": {}, "best": None}


def _save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _eval_snapshot(
    model_path,
    episodes,
    device,
    tmp_json,
    *,
    profile,
    fruits_total,
    start_state,
    stall_threshold,
    max_steps,
    level=None,
    resize_mode="nearest",
    reach_mode="sprite",
):
    """Run eval_from_reset.py in a subprocess; return (princess, reach_top, mean_rung,
    n_rungs).

    ``mean_rung`` is the average number of MANDATORY route targets an episode gets
    behind it -- the same quantity training reports as ``reset_reach``. It exists
    because princess and reach_top cannot rank snapshots on a single-fruit level: see
    the scoring note in main().

    Also returns ``rates``: per-route-point, the FRACTION OF EPISODES that reached it.
    A rate is a Bernoulli mean, so its noise is knowable and small where `mean_rung`'s
    is not. Bootstrapped from the 300-episode champion evals:

        metric                      n=12    n=30    n=60    n=100
        mean_rung            sd     0.75    0.47    0.34    0.26
        a reach rate at ~0.75 sd    0.126   0.078   0.057   0.043

    The v16c-vs-v18 difference that mattered was 0.47 RUNGS, so `mean_rung` at 12
    episodes carries more noise than the signal it is asked to resolve -- every champion
    this project ever selected was picked on that. A rate at 30 episodes resolves a
    0.24 drop at three sigma.
    """
    cmd = [
        sys.executable,
        "scripts/mo5/yeti/eval_from_reset.py",
        "--model",
        model_path,
        "--episodes",
        str(episodes),
        "--stochastic",
        "--out",
        tmp_json,
        "--profile",
        profile,
        "--fruits-total",
        str(fruits_total),
        "--stall-threshold",
        str(stall_threshold),
        "--max-steps",
        str(max_steps),
    ]
    if level is not None:
        cmd += ["--level", str(level)]
    if start_state:
        cmd += ["--start-state", start_state]
    # Only passed when non-default, so the command for every existing run is unchanged.
    if resize_mode != "nearest":
        cmd += ["--resize-mode", resize_mode]
    # eval_from_reset's default is "sprite", the trainer's default; only "box" needs
    # saying. Before 2026-10-07 eval ALWAYS used box, whatever the run trained with.
    if reach_mode != "sprite":
        cmd += ["--reach-mode", reach_mode]
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, env=env, capture_output=True, timeout=3600, check=True)
    with open(tmp_json) as f:
        data = json.load(f)
    n = data["episodes"]
    princess = data["princess_touches"] / n
    reach_top = (
        sum(v for k, v in data["max_cp_counts"].items() if int(k) >= fruits_total) / n
    )
    hits: Counter = Counter()
    for row in data.get("rows") or ():
        for pt in row.get("reached_points") or ():
            hits[pt] += 1
    rates = {k: v / n for k, v in hits.items()} if n else {}
    # Progress-ladder rates: "rungK" = fraction of episodes that got AT LEAST K rungs.
    # The frontier for a level with no route order (see _frontier_route). Added for
    # every level; a level WITH a route order never looks these keys up.
    if n:
        rungs = {int(k): v for k, v in (data.get("rung_counts") or {}).items()}
        for k in range(1, int(data.get("n_rungs") or 0) + 1):
            rates[f"rung{k}"] = sum(v for r, v in rungs.items() if r >= k) / n
    return (
        princess,
        reach_top,
        data.get("mean_rung", 0.0),
        data.get("n_rungs", 0),
        rates,
    )


def _eval_snapshot_registry(
    model_path, episodes, device, tmp_json, config, level, pool=None
):
    """``_eval_snapshot`` for --config mode: every setting from the run config and the
    registry (eval_from_reset.py --config), none typed here.

    Returns ``_eval_snapshot``'s five values plus ``(princess_touches, episodes,
    native_sha256)``, which the champion comparison needs as counts, together with the
    emulator build the numbers came from.
    """
    cmd = [
        sys.executable,
        "scripts/mo5/yeti/eval_from_reset.py",
        "--model",
        model_path,
        "--config",
        config,
        "--game-level",
        str(level),
        "--episodes",
        str(episodes),
        "--out",
        tmp_json,
    ]
    if pool:
        cmd += ["--start-pool", pool, "--pool-split", "heldout"]
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, env=env, capture_output=True, timeout=3600, check=True)
    with open(tmp_json) as f:
        data = json.load(f)
    n = data["episodes"]
    fruits_total = int(data["fruits_total"])
    princess = data["princess_touches"] / n
    reach_top = (
        sum(v for k, v in data["max_cp_counts"].items() if int(k) >= fruits_total) / n
    )
    hits: Counter = Counter()
    for row in data.get("rows") or ():
        for pt in row.get("reached_points") or ():
            hits[pt] += 1
    rates = {k: v / n for k, v in hits.items()} if n else {}
    if n:
        rungs = {int(k): v for k, v in (data.get("rung_counts") or {}).items()}
        for k in range(1, int(data.get("n_rungs") or 0) + 1):
            rates[f"rung{k}"] = sum(v for r, v in rungs.items() if r >= k) / n
    return (
        princess,
        reach_top,
        data.get("mean_rung", 0.0),
        data.get("n_rungs", 0),
        rates,
        int(data["princess_touches"]),
        int(n),
        data.get("native_sha256"),
    )


def _score_level(princess, reach_top, mean_rung, n_rungs, rates, route):
    """One level's ranking score and the parts it is made of (see the note in main)."""
    depth = (mean_rung / n_rungs) if n_rungs else 0.0
    fwid, frate = _frontier(rates, route)
    fdepth = (route.index(fwid) + 1) / len(route) if (fwid and route) else 0.0
    score = (
        princess
        + 1e-2 * fdepth  # how far along the route the frontier sits
        + 1e-4 * frate  # how reliably it gets there
        + 1e-6 * depth
    )
    return {
        "princess": princess,
        "reach_top": reach_top,
        "mean_rung": mean_rung,
        "n_rungs": n_rungs,
        "depth": depth,
        "frontier": fwid,
        "frontier_rate": frate,
        "score": score,
        "rates": rates,
    }


def _frontier_route(level_route, n_rungs):
    """The ordered list the frontier walks: the level's route order, or its ladder.

    A level map's ``route_order`` names waypoints in travel order, and L3/L4 define one.
    L1 and L2 do not -- their progress ladder is just the fruits -- so the frontier used
    to come back empty there, and their snapshots were ranked on princess and mean rung
    alone. With no route order, the route is the ladder itself: ``rung1 .. rungN``,
    where ``rungK``'s rate (from _eval_snapshot) is the fraction of episodes that got at
    least K rungs. That is order-free, so it does not assume the fruits are collected in
    any particular sequence.

    NOT fixed by giving L1 a route order: the trainer's start gate refuses any pool
    absent from ``route_order``, so a fruits-only route would lock out every ladder
    waypoint pool on the level.
    """
    if level_route:
        return list(level_route)
    return [f"rung{k}" for k in range(1, int(n_rungs or 0) + 1)]


def _frontier(rates, route_order, floor=0.05):
    """The DEEPEST route point this policy still reaches, and its rate.

    "Deepest reached" rather than "deepest defined": a point nothing ever reaches has
    rate 0 and carries no signal, so scoring on it cannot distinguish two policies. The
    frontier moves outward on its own as the agent improves, which is what makes this
    level-agnostic -- nothing here names a waypoint.
    """
    best = (None, 0.0)
    for wid in route_order or ():
        r = rates.get(wid, 0.0)
        if r >= floor:
            best = (wid, r)
    return best


def _rate_ci(rate, n, z=1.96):
    """95% Wilson interval half-width for a rate seen over ``n`` episodes.

    WHY EVERY EVAL NEEDS THIS PRINTED NEXT TO IT. A frontier rate is a proportion over
    independent from-reset episodes, so how well it is known depends entirely on ``n``,
    and nothing in this project's logs ever said so. At the n=30 this tool defaults to,
    a rate near 0.7 is known to about +-0.16 -- wider than most differences that have
    been argued over, including the 0.125 between v24 and v25 that cost three 6M runs.
    At n=300 the same rate is +-0.05.

    Wilson rather than the textbook sqrt(p(1-p)/n): our rates sit AT 0 and 1 regularly
    (a collapsed frontier reads 1.00 on a shallow point), where the textbook form gives
    a nonsensical +-0.00 and would report a one-episode certainty.

    Returns the half-width only -- the interval is clipped to [0, 1] by construction, so
    a single number is the honest summary for a log line.
    """
    if not n:
        return 0.0
    p = min(max(float(rate), 0.0), 1.0)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (hi - lo) / 2.0


# Flags whose value --config takes from the run config and the registry.
_LEVEL_FLAGS = (
    "profile",
    "fruits_total",
    "level",
    "start_state",
    "stall_threshold",
    "max_steps",
    "resize_mode",
    "reach_mode",
    "level_route",
)


def _registry_plan(config_path):
    """``(registry, cfg, [(game_level, settings, route_order)])`` for --config mode.

    Every level the run trains on (its start groups, else its own level), each with the
    settings it is scored under and its route order. Refuses a run that trained under
    settings other than a level's (Registry.mismatches): its referee scores would be
    taken under conditions it never trained in, after the hours of training, so it is
    checked before -- run_train_and_score.sh calls --check first.
    """
    from retro_ai.training.root_pool import read_manifest
    from retro_ai.training.run_config import RunConfig
    from retro_ai.training.yeti_map import get_level_map
    from retro_ai.training.yeti_registry import REPO, Registry

    registry = Registry.load()
    cfg = RunConfig.from_yaml(config_path)
    try:
        levels = registry.levels_of(cfg)
    except ValueError as exc:
        raise SystemExit(f"{config_path}: {exc}") from None

    def _same(a, b):
        return (REPO / a).resolve() == (REPO / b).resolve()

    # The pool the run TRAINS on (curriculum.root_pool), and its level.
    train_pool = getattr(cfg.curriculum, "root_pool", None)
    train_level = int(read_manifest(train_pool)["game_level"]) if train_pool else None
    if train_pool and train_level not in levels:
        raise SystemExit(
            f"root pool {train_pool} is for game level {train_level}, not one of the "
            f"run's levels {levels}"
        )
    plan = []
    for n in levels:
        clash = registry.mismatches(cfg, n)
        if clash:
            raise SystemExit(
                f"{config_path} trains under settings other than game level {n}'s: "
                + "; ".join(clash)
            )
        s = registry.config_eval_settings(cfg, n)
        route = list(getattr(get_level_map(s.layout), "route_order", None) or [])
        # WHERE A LEVEL IS SCORED. A level with a start pool (registry.yaml) is scored
        # on that pool's HELD-OUT hazard sequences, whether or not the run trains on
        # the pool: a level start other than a game reset is one draw among many (the
        # hazards a level opens with depend on when the previous level was finished),
        # so scoring on it would reward memorising one pattern. The level's own start
        # is kept beside it only when it is the GAME RESET (L1), which every real game
        # does play first, and then for information only (_informational).
        pool = registry.level(n).start_pool
        if train_pool and train_level == n:
            if pool and not _same(pool, train_pool):
                raise SystemExit(
                    f"{config_path} trains on root pool {train_pool}, but game level "
                    f"{n} is scored on its start pool {pool} (registry.yaml). Their "
                    "held-out sequences are not each other's, so training on one can "
                    "train on the other's held-out starts. Train on the level's pool, "
                    "or register the new one."
                )
            pool = pool or train_pool
        if pool is None:
            plan.append((n, s, route, None))
            continue
        if int(read_manifest(pool)["game_level"]) != n:
            raise SystemExit(f"start pool {pool} is not for game level {n}")
        if s.start_state is None:
            plan.append((n, s, route, None))
        plan.append((n, s, route, pool))
    return registry, cfg, plan


def _informational(plan) -> set:
    """Entry keys reported for information only: a level's own start when the level
    is also scored on a start pool. They count toward no score and no champion rule."""
    pooled = {n for n, _s, _r, p in plan if p is not None}
    return {_entry_key(n, p) for n, _s, _r, p in plan if p is None and n in pooled}


def _score_levels(level_results, plan, registry, champion_ref):
    """Per-entry records, the levels a snapshot fell below a champion on, and the
    snapshot's score: the MEAN over the entries that decide (not _informational).

    ``level_results`` are ``(game_level, route, pool, eval_result)`` in plan order.
    A deciding entry with a registered champion is compared with that champion's rate
    MEASURED THE SAME WAY -- from the level start, or on the same pool's held-out
    sequences -- on the same emulator build (regressed_against, 3 sigma). A champion
    with no such result has nothing to fall below.
    """
    from retro_ai.training.yeti_registry import regressed_against

    info = _informational(plan)
    levels_out, regressed = {}, []
    for n, lroute, lpool, res in level_results:
        lp, lr, lm, ln, lrates, lk, leps, lnative = res
        rec = _score_level(lp, lr, lm, ln, lrates, _frontier_route(lroute, ln))
        rec.update(princess_touches=lk, episodes=leps, native_sha256=lnative)
        key = _entry_key(n, lpool)
        if key in info:
            rec["informational"] = True
        champ = None if key in info else registry.champions.get(n)
        if champ:
            ref = (n, lpool, lnative)
            if ref not in champion_ref:
                champion_ref[ref] = registry.rate(
                    champ, n, lnative, lpool, "heldout" if lpool else None
                )
            ck, cn = champion_ref[ref]
            bad, z = regressed_against(lk, leps, ck, cn)
            rec.update(champion=champ, champion_k=ck, champion_n=cn, z=z)
            rec["champion_regressed"] = bad
            if bad:
                regressed.append(n)
        levels_out[key] = rec
    deciding = [r for k, r in levels_out.items() if k not in info]
    score = sum(r["score"] for r in deciding) / len(deciding)
    return levels_out, regressed, score


def _entry_key(n, pool) -> str:
    return f"{n}" if pool is None else f"{n}-heldout"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshots-dir", required=True)
    p.add_argument(
        "--config",
        default=None,
        help="the run config that produced the snapshots. Takes every level setting "
        "from it and from experiments/003-yeti/registry.yaml (passing --profile, "
        "--level, --start-state, --fruits-total, --stall-threshold, --max-steps, "
        "--resize-mode, --reach-mode or --level-route as well is refused), scores "
        "EVERY level the run trains on (its curriculum.start_groups), and with more "
        "than one level only lets a snapshot become best if no level with a "
        "registered champion scores below that champion's measured rate by more than "
        "the two sample sizes explain.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="with --config: resolve and print what would be evaluated, then exit "
        "(non-zero if the config cannot be scored)",
    )
    p.add_argument(
        "--best-dir",
        default=None,
        help="where to keep best_model.zip (default: " "<snapshots-dir>/../best)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="episodes per eval (cheap trigger; re-eval the "
        "winner with more for a precise number)",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu (safe alongside training) or gpu",
    )
    p.add_argument(
        "--watch", action="store_true", help="poll for new snapshots until idle"
    )
    p.add_argument("--poll-sec", type=int, default=120)
    p.add_argument("--max-idle-min", type=float, default=30.0)
    # Level awareness (defaults = level 1). For level 2 pass:
    #   --profile yeti_fruit_level2 --fruits-total 2 --stall-threshold 40
    #   --start-state output/mo5/yeti/level2/level2_start.sav
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--fruits-total", type=int, default=4)
    p.add_argument(
        "--level",
        type=int,
        default=None,
        help="level geometry; REQUIRED for route-depth scoring to be meaningful",
    )
    p.add_argument("--start-state", default=None)
    p.add_argument("--stall-threshold", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument(
        "--resize-mode",
        default="nearest",
        choices=["nearest", "max"],
        help="forwarded to eval_from_reset. MUST match the run's env.resize_mode: a "
        "mismatch scores every snapshot on a picture it was never trained on, and "
        "nothing errors (the L4 v29 champion reads mean rung 8.17 vs 0.02).",
    )
    p.add_argument(
        "--reach-mode",
        default="sprite",
        choices=["box", "sprite"],
        help="forwarded to eval_from_reset; MUST match the run's "
        "curriculum.waypoint_reach_mode. The frontier and its rate are computed from "
        "the reach test, so a mismatch moves both. 'box' reproduces referee scores "
        "made before 2026-10-07.",
    )
    p.add_argument(
        "--level-route",
        type=int,
        default=None,
        help="level whose route order defines the frontier. Defaults to --level. The "
        "frontier is the DEEPEST route point still reached, discovered per snapshot, "
        "so nothing here names a waypoint and it works on any level.",
    )
    p.add_argument(
        "--regress-margin",
        type=float,
        default=None,
        help="how far the frontier rate may fall before it counts as a regression. "
        "Default: 3x the binomial sd at --episodes, so the flag means 'more than the "
        "measurement can explain' rather than a hand-picked number.",
    )
    return p


def main() -> None:
    argv = sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)

    # --config: every level setting from the run config and the registry.
    registry = plan = None
    if args.config:
        from retro_ai.training.yeti_registry import hand_settings_given

        clash = hand_settings_given(build_parser(), argv, _LEVEL_FLAGS)
        if clash:
            raise SystemExit(
                f"--config takes {', '.join(clash)} from the run config and the "
                f"registry; do not pass {'it' if len(clash) == 1 else 'them'} as well"
            )
        registry, _cfg, plan = _registry_plan(args.config)
        info = _informational(plan)
        for n, s, route, pool in plan:
            key = _entry_key(n, pool)
            champ = None if key in info else registry.champions.get(n)
            start = s.start_state or "reset"
            if pool is not None:
                start = f"held-out starts of {pool}"
            print(
                f"[keep-best] level {key}: layout {s.layout}, start "
                f"{start}, profile {s.env.profile}, stop "
                f"{s.env.max_steps}/{s.env.stall_threshold}, resize "
                f"{s.env.resize_mode}, reach {s.reach_mode}, route "
                f"{len(route) or 'ladder'}, "
                + (
                    "for information only"
                    if key in info
                    else f"champion {champ or '-'}"
                ),
                flush=True,
            )
        if args.check:
            return
    elif args.check:
        raise SystemExit("--check needs --config")
    # Per-level scoring (_score_levels) whenever there is more than one entry, or the
    # one entry is a start pool's held-out starts (its champion bar is measured there).
    multi = bool(plan) and (len(plan) > 1 or plan[0][3] is not None)
    # (level, pool, emulator sha) -> (k, n) of the level's champion, measured the same
    # way; filled on first use, because the emulator sha comes from the eval output.
    champion_ref: dict = {}

    # Travel order for the frontier. From the level map, so no waypoint is named here.
    route_order: list = []
    _lvl = args.level_route if args.level_route is not None else args.level
    if _lvl and not plan:
        try:
            from retro_ai.training.yeti_map import get_level_map

            route_order = list(get_level_map(int(_lvl)).route_order or [])
        except Exception as exc:  # pragma: no cover - level without a route
            print(f"[keep-best] no route order for level {_lvl}: {exc}", flush=True)

    snap_dir = args.snapshots_dir
    best_dir = args.best_dir or os.path.join(os.path.dirname(snap_dir), "best")
    os.makedirs(best_dir, exist_ok=True)
    state_path = os.path.join(best_dir, "sweep_state.json")
    tmp_json = os.path.join(best_dir, "_eval.json")
    state = _load_state(state_path)

    def best_score():
        return state["best"]["score"] if state["best"] else -1.0

    last_new = time.time()
    while True:
        snaps = _snapshots(snap_dir)
        new = [
            (s, path)
            for s, path in snaps
            if os.path.basename(path) not in state["evaluated"]
        ]
        for step, path in new:
            # STAMP PER SNAPSHOT, NOT PER BATCH. `last_new` means "when did we last have
            # work", and the idle check below is `now - last_new`. Stamping it once
            # before the batch made that difference measure time spent WORKING: an eval
            # costs ~56 s, so a backlog bigger than `max_idle_min` minutes of work made
            # the loop quit after one pass while snapshots were still queued.
            #
            # Measured on v26 (2026-09-26): launched against a run already 1h40m in, it
            # evaluated 36 of 60 snapshots, then printed "idle 32 min, stopping" after
            # exactly 32 minutes of continuous evaluation and exited with 24 snapshots
            # unevaluated -- including the window where the CONTROL run v24 had its best
            # snapshot (3.8M), so the comparison the run existed for was unavailable.
            last_new = time.time()
            name = os.path.basename(path)
            try:
                if plan:
                    level_results = [
                        (
                            n,
                            route,
                            pool,
                            _eval_snapshot_registry(
                                path,
                                args.episodes,
                                args.device,
                                tmp_json,
                                args.config,
                                n,
                                pool,
                            ),
                        )
                        for n, _s, route, pool in plan
                    ]
                    # The top-level numbers are the first DECIDING entry's (an
                    # informational one, L1's game reset beside its pool, never is).
                    info = _informational(plan)
                    primary = next(
                        r for r in level_results if _entry_key(r[0], r[2]) not in info
                    )
                    princess, reach_top, mean_rung, n_rungs, rates = primary[3][:5]
                else:
                    princess, reach_top, mean_rung, n_rungs, rates = _eval_snapshot(
                        path,
                        args.episodes,
                        args.device,
                        tmp_json,
                        profile=args.profile,
                        fruits_total=args.fruits_total,
                        start_state=args.start_state,
                        stall_threshold=args.stall_threshold,
                        max_steps=args.max_steps,
                        level=args.level,
                        resize_mode=args.resize_mode,
                        reach_mode=args.reach_mode,
                    )
            except Exception as e:
                print(f"[keep-best] {name}: eval FAILED ({e})", flush=True)
                continue
            # SCORING, in strict priority: princess, then ROUTE DEPTH, then fruit.
            #
            # It used to be `princess + 1e-3 * reach_top`, which cannot rank snapshots
            # on a single-fruit level: princess is uniformly 0 and reach_top just means
            # "collected the fruit", which saturates at 1.0. Ties then went to whichever
            # snapshot was seen FIRST, so every L4 champion ever picked was arbitrary --
            # v4 kept its 100k snapshot, v5 its 1M, v6 its 200k, while the run's actual
            # depth peak was elsewhere entirely (v6's was ~8.76M).
            #
            # `mean_rung / n_rungs` is the fraction of mandatory route targets an
            # episode gets behind it on average -- the same quantity the training route
            # table reports as reset_reach. Weighted below princess so a single princess
            # touch still outranks any amount of depth, and above reach_top so the fruit
            # only breaks ties between equally deep policies.
            depth = (mean_rung / n_rungs) if n_rungs else 0.0
            # FRONTIER RATE is the ranking term, not `mean_rung`. It is a Bernoulli
            # mean, so its noise is known and small (sd 0.078 at n=30 for a rate near
            # 0.75) where `mean_rung`'s sd at n=12 is 0.75 rungs against a 0.47-rung
            # signal. `mean_rung` stays in the record and as the last tie-break, since
            # it is comparable with the training route table.
            # The level's route order, or -- on a level without one (L1, L2) -- its
            # progress ladder. Identical to before wherever a route order exists, so
            # L3/L4 scores stay comparable.
            route = _frontier_route(primary[1] if plan else route_order, n_rungs)
            # (plan entries are (game level, settings, route, root pool or None))
            fwid, frate = _frontier(rates, route)
            fdepth = (route.index(fwid) + 1) / len(route) if (fwid and route) else 0.0
            score = (
                princess
                + 1e-2 * fdepth  # how far along the route the frontier sits
                + 1e-4 * frate  # how reliably it gets there
                + 1e-6 * depth
            )
            # SEVERAL LEVELS (--config with start groups), or a level scored on its
            # start pool. Every deciding entry is scored the same way and the
            # snapshot's score is their MEAN, so no level is weighted by a chosen
            # number; an informational entry (L1's game reset beside its pool) counts
            # toward nothing. A level with a registered champion is also compared
            # with that champion's measured rate on the same emulator build
            # (Registry.rate, pooled over its result files): a snapshot below it by more
            # than the two sample sizes explain (regressed_against, 3 sigma, the same
            # convention as the within-run margin below) is REGRESSED on that level and
            # cannot become best, however much it gains elsewhere. That is the rule
            # "training on L5 must not cost L1". A level with no champion (L5 today)
            # has nothing to fall below.
            levels_out = {}
            champion_regressed = []
            if multi:
                levels_out, champion_regressed, score = _score_levels(
                    level_results, plan, registry, champion_ref
                )
            eligible = not champion_regressed
            state["evaluated"][name] = {
                "step": step,
                "princess": princess,
                "reach_top": reach_top,
                "mean_rung": mean_rung,
                "n_rungs": n_rungs,
                "depth": depth,
                "frontier": fwid,
                "frontier_rate": frate,
                "score": score,
                "n_eval": args.episodes,
                "rates": rates,
            }
            # REGRESSION, against the best frontier seen so far. Reported, never acted
            # on here: this process only observes. `--regress-margin` defaults to 3x the
            # binomial sd at this episode count, so the flag means "bigger than the
            # measurement can explain" rather than a number picked by hand.
            sd = (max(frate, 1e-9) * (1 - min(frate, 1 - 1e-9)) / args.episodes) ** 0.5
            margin = (
                args.regress_margin if args.regress_margin is not None else 3.0 * sd
            )
            bf = (state.get("best") or {}).get("frontier")
            bfr = (state.get("best") or {}).get("frontier_rate") or 0.0
            regressed = bool(
                bf
                and route
                and fwid
                # A best recorded under the other naming (a state file from before this
                # fallback existed) cannot be compared; do not crash on it.
                and bf in route
                and (
                    route.index(fwid) < route.index(bf)
                    or (fwid == bf and frate < bfr - margin)
                )
            )
            if plan:
                state["evaluated"][name]["levels"] = levels_out
            if multi:
                state["evaluated"][name]["champion_regressed"] = champion_regressed
                state["evaluated"][name]["eligible"] = eligible
                # A champion regression counts as a regression for anything acting on
                # eval_status.json (RegressionStopCallback's patience counter).
                regressed = regressed or bool(champion_regressed)
            state["evaluated"][name]["regressed"] = regressed
            state["evaluated"][name]["regress_margin"] = margin
            # How well is that rate known? A proportion over `episodes` independent
            # from-reset episodes, so print the interval beside it -- at n=30 a rate
            # near 0.7 is +-0.16, which is wider than most differences this project has
            # argued about.
            fci = _rate_ci(frate, args.episodes)
            msg = (
                f"[keep-best] step {step}: princess={princess:.3f} "
                f"frontier={fwid or '-'}@{frate:.2f}+-{fci:.2f} "
                f"rung={mean_rung:.2f}/{n_rungs} "
                f"(best={best_score():.4f}{' REGRESSED' if regressed else ''})"
            )
            if multi:
                msg += " | " + "  ".join(
                    f"L{n}{' (info)' if r.get('informational') else ''}: "
                    f"princess={r['princess']:.3f} "
                    f"frontier={r['frontier'] or '-'}@{r['frontier_rate']:.2f}"
                    + (
                        f" vs {r['champion']} {r['champion_k']}/{r['champion_n']} "
                        f"z={r['z']:+.1f}"
                        + (" CHAMPION-REGRESSED" if r["champion_regressed"] else "")
                        if "champion" in r
                        else ""
                    )
                    for n, r in levels_out.items()
                )
            if score > best_score() and eligible:
                shutil.copyfile(path, os.path.join(best_dir, "best_model.zip"))
                state["best"] = {
                    "model": name,
                    "step": step,
                    "score": score,
                    "princess": princess,
                    "reach_top": reach_top,
                    "mean_rung": mean_rung,
                    "n_rungs": n_rungs,
                    "frontier": fwid,
                    "frontier_rate": frate,
                    "n_eval": args.episodes,
                }
                if multi:
                    state["best"]["levels"] = {
                        n: {
                            k: r[k]
                            for k in ("princess", "frontier", "frontier_rate", "score")
                        }
                        for n, r in levels_out.items()
                    }
                with open(os.path.join(best_dir, "best_meta.json"), "w") as f:
                    json.dump(state["best"], f, indent=2)
                msg += "  -> NEW BEST (saved)"
            print(msg, flush=True)
            _save_state(state_path, state)
            # A small, stable file for anything that wants to ACT on this -- a human
            # deciding whether to kill a run, or a future callback reverting weights.
            # Written every eval so a reader never has to parse the sweep state or the
            # log. `consecutive_regressions` is the patience counter such a caller
            # needs: one dip is noise, three in a row is not.
            _run = state.get("consecutive_regressions", 0)
            state["consecutive_regressions"] = (_run + 1) if regressed else 0
            with open(os.path.join(best_dir, "eval_status.json"), "w") as f:
                json.dump(
                    {
                        "step": step,
                        "frontier": fwid,
                        "frontier_rate": frate,
                        # 95% half-width on the rate above, from `n_eval` episodes.
                        # Any consumer comparing two of these MUST check that the
                        # difference exceeds the two half-widths before calling it real.
                        "frontier_rate_ci95": fci,
                        "princess": princess,
                        "princess_ci95": _rate_ci(princess, args.episodes),
                        "mean_rung": mean_rung,
                        "n_eval": args.episodes,
                        "regressed": regressed,
                        "regress_margin": margin,
                        "consecutive_regressions": state["consecutive_regressions"],
                        "best_step": (state.get("best") or {}).get("step"),
                        "best_frontier": (state.get("best") or {}).get("frontier"),
                        "best_frontier_rate": (state.get("best") or {}).get(
                            "frontier_rate"
                        ),
                        "best_model": os.path.join(best_dir, "best_model.zip"),
                        # Per entry, with --config (start groups, start pools). The
                        # top-level fields above are the first DECIDING entry's.
                        **(
                            {
                                "levels": {
                                    n: {
                                        k: r.get(k)
                                        for k in (
                                            "princess",
                                            "princess_touches",
                                            "episodes",
                                            "frontier",
                                            "frontier_rate",
                                            "champion",
                                            "champion_k",
                                            "champion_n",
                                            "z",
                                            "champion_regressed",
                                        )
                                    }
                                    for n, r in levels_out.items()
                                },
                                "champion_regressed": champion_regressed,
                            }
                            if multi
                            else {}
                        ),
                    },
                    f,
                    indent=1,
                )
            # Stamp again now the work is DONE, so "idle" means time with nothing to do
            # rather than time since this snapshot's eval started. Without this a single
            # eval longer than `max_idle_min` trips the idle check on its own.
            last_new = time.time()

        if not args.watch:
            break
        idle_min = (time.time() - last_new) / 60.0
        if idle_min >= args.max_idle_min:
            print(f"[keep-best] idle {idle_min:.0f} min, stopping.", flush=True)
            break
        time.sleep(args.poll_sec)

    b = state["best"]
    if b:
        print(
            f"\nBest: {b['model']} (step {b['step']}) "
            f"princess={b['princess']:.3f} "
            f"rung={b.get('mean_rung', 0):.2f}/{b.get('n_rungs', 0)} "
            f"reach_top={b.get('reach_top', 0):.3f}  "
            f"-> {os.path.join(best_dir, 'best_model.zip')}"
        )
        print("Re-eval the winner with more episodes for a precise number, e.g.:")
        print(
            f"  python scripts/mo5/yeti/eval_from_reset.py --model "
            f"{os.path.join(best_dir, 'best_model.zip')} --episodes 300 --stochastic"
            + (
                f" --resize-mode {args.resize_mode}"
                if args.resize_mode != "nearest"
                else ""
            )
        )
    elif state["evaluated"]:
        blocked = sorted(
            {
                n
                for v in state["evaluated"].values()
                for n in (v.get("champion_regressed") or ())
            }
        )
        print(
            f"No best: none of {len(state['evaluated'])} evaluated snapshot(s) "
            "qualified"
            + (
                f"; every one fell below a champion on level(s) {blocked}"
                if blocked
                else ""
            )
            + "."
        )
    else:
        print("No snapshots evaluated.")


if __name__ == "__main__":
    main()
