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
    return (
        princess,
        reach_top,
        data.get("mean_rung", 0.0),
        data.get("n_rungs", 0),
        rates,
    )


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshots-dir", required=True)
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
    args = p.parse_args()

    # Travel order for the frontier. From the level map, so no waypoint is named here.
    route_order: list = []
    _lvl = args.level_route if args.level_route is not None else args.level
    if _lvl:
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
            fwid, frate = _frontier(rates, route_order)
            fdepth = (
                (route_order.index(fwid) + 1) / len(route_order)
                if (fwid and route_order)
                else 0.0
            )
            score = (
                princess
                + 1e-2 * fdepth  # how far along the route the frontier sits
                + 1e-4 * frate  # how reliably it gets there
                + 1e-6 * depth
            )
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
                and route_order
                and fwid
                and (
                    route_order.index(fwid) < route_order.index(bf)
                    or (fwid == bf and frate < bfr - margin)
                )
            )
            state["evaluated"][name]["regressed"] = regressed
            state["evaluated"][name]["regress_margin"] = margin
            msg = (
                f"[keep-best] step {step}: princess={princess:.3f} "
                f"frontier={fwid or '-'}@{frate:.2f} "
                f"rung={mean_rung:.2f}/{n_rungs} "
                f"(best={best_score():.4f}{' REGRESSED' if regressed else ''})"
            )
            if score > best_score():
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
                        "princess": princess,
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
        )
    else:
        print("No snapshots evaluated.")


if __name__ == "__main__":
    main()
