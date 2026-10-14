#!/usr/bin/env python3
"""Evaluate a single trained policy from game reset (CP0).

Rolls out N episodes from a clean reset (no save-state loading), under the
SAME episode termination as training (death / stall / max_steps / princess
touch), and reports the distribution of the deepest checkpoint reached and
the princess-touch rate.

This measures the North Star directly: P(reach CP_k | start = reset) for a
single policy. Defaults to a deterministic policy; pass --stochastic to
sample actions instead.

Example::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/mo5/yeti/eval_from_reset.py \\
        --model output/mo5/yeti/warmstart/v2_clean/final_model.zip \\
        --episodes 200

PREFER THE REGISTRY for any number that will be quoted or compared::

    python scripts/mo5/yeti/eval_from_reset.py --policy l1_v17 --game-level 5 \\
        --episodes 300

``--policy`` takes the model, level, start state, profile, stop rules, resize mode and
reach test from ``experiments/003-yeti/registry.yaml`` and the policy's training config,
refuses those flags if they are also passed, checks the model's sha256, samples
stochastically, and writes a result file under
``experiments/003-yeti/data/champion_recheck/<policy>/`` that the registry reads back.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="required unless --policy is given")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--stall-threshold", type=int, default=15)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out", default=None)
    # Level awareness (defaults = level 1). For level 2 pass e.g.:
    #   --profile yeti_fruit_level2 --fruits-total 2 --stall-threshold 40
    #   --start-state output/mo5/yeti/level2/level2_start.sav
    p.add_argument(
        "--fruits-total",
        type=int,
        default=4,
        help="collectible fruits in the level (L1=4, L2=2). Sets CP indexing "
        "and the princess goal index (= fruits_total + 1).",
    )
    p.add_argument(
        "--start-state",
        default=None,
        help="save-state to load on each reset (level 2 starts from a save, "
        "not a game reset). Omit for level 1 (game reset).",
    )
    p.add_argument(
        "--level",
        type=int,
        default=None,
        help="level geometry to use. Previously INFERRED as '2 if --start-state else "
        "1', so every L3/L4 eval silently used level-2 geometry. Defaults to that old "
        "behaviour when omitted so existing invocations are unchanged; pass it "
        "explicitly (and you must, for route-depth scoring to mean anything).",
    )
    p.add_argument(
        "--no-depth",
        action="store_true",
        help="skip route-depth tracking (max_rung). Depth is what lets a sweep rank "
        "snapshots on a level where princess is uniformly 0 and the fruit count "
        "saturates.",
    )
    p.add_argument(
        "--waypoint-tolerance",
        type=int,
        default=2,
        help="ladder tolerance, used by --reach-mode box only",
    )
    p.add_argument(
        "--jump-waypoint-tolerance",
        type=int,
        default=6,
        help="jump-landing tolerance, used by --reach-mode box only",
    )
    p.add_argument(
        "--reach-mode",
        default="sprite",
        choices=["box", "sprite"],
        help="how a route point counts as reached. MUST match the run's "
        "curriculum.waypoint_reach_mode (default sprite since de21939). 'box' is kept "
        "to reproduce evals made before 2026-10-07, which used it whatever the run "
        "trained with -- and so read L4's rope-2 landing as never reached.",
    )
    p.add_argument(
        "--resize-mode",
        default="nearest",
        choices=["nearest", "max"],
        help="MUST match the mode the model was trained with (env.resize_mode). A "
        "mismatch does not error, it just scores the policy on a picture it never saw: "
        "the L4 v29 champion reads mean rung 8.17 under its own mode and 0.02 under "
        "the other.",
    )
    p.add_argument(
        "--policy",
        default=None,
        help="registry policy id (experiments/003-yeti/registry.yaml). Sets the model "
        "and every setting the registry owns; passing any of those as well is refused. "
        "Implies --stochastic and writes a result file the registry reads back.",
    )
    p.add_argument(
        "--game-level",
        type=int,
        default=None,
        help="the game level to evaluate --policy (or --config) on, counted as the "
        "game counts them (L5 is a level of its own, on L1's layout)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="run config that trained --model (an unregistered model, e.g. a "
        "snapshot the referee is scoring). With --game-level, takes the policy's view "
        "from this config and the level settings from the registry, and refuses the "
        "same hand-typed flags --policy does, except --model.",
    )
    p.add_argument(
        "--start-pool",
        default=None,
        help="respawn-harvest manifest (scripts/mo5/yeti/harvest_respawns.py). Each "
        "episode starts from a state drawn from it, uniform over snowball sequences "
        "then states, instead of the level start. Needs --policy or --config with "
        "--game-level equal to the manifest's.",
    )
    p.add_argument(
        "--pool-split",
        default="heldout",
        choices=["heldout", "train", "all"],
        help="which sequences of --start-pool to draw from (default: the held-out "
        "ones, never trained on)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seeds the policy's action sampling. Default: a random seed, recorded in "
        "the output so the run can be repeated.",
    )
    return p


def _from_config(args: argparse.Namespace, argv: list):
    """Fill ``args`` for --model + --config + --game-level (an unregistered model)."""
    from retro_ai.training import yeti_registry as reg
    from retro_ai.training.run_config import RunConfig

    if args.game_level is None or args.model is None:
        raise SystemExit("--config needs --model and --game-level")
    if args.policy is not None:
        raise SystemExit("pass --policy or --config, not both")
    owned = [s for s in reg.HAND_SETTINGS if s != "model"]
    clash = reg.hand_settings_given(build_parser(), argv, owned)
    if clash:
        raise SystemExit(
            f"--config takes {', '.join(clash)} from the run config and the "
            f"registry; do not pass {'it' if len(clash) == 1 else 'them'} as well"
        )
    registry = reg.Registry.load()
    s = registry.config_eval_settings(RunConfig.from_yaml(args.config), args.game_level)
    args.stochastic = True
    args.profile = s.env.profile
    args.max_steps = s.env.max_steps
    args.stall_threshold = s.env.stall_threshold
    args.fruits_total = s.fruits_total
    args.start_state = s.start_state
    args.level = s.layout
    args.resize_mode = s.env.resize_mode
    args.reach_mode = s.reach_mode
    return s, registry


def _from_registry(args: argparse.Namespace, argv: list):
    """Fill ``args`` from the registry for --policy / --game-level."""
    from retro_ai.training import yeti_registry as reg

    if args.config is not None:
        return _from_config(args, argv)
    if args.policy is None or args.game_level is None:
        raise SystemExit("--policy and --game-level must be given together")
    clash = reg.hand_settings_given(build_parser(), argv)
    if clash:
        raise SystemExit(
            f"--policy takes {', '.join(clash)} from the registry; do not pass "
            f"{'it' if len(clash) == 1 else 'them'} as well"
        )
    registry = reg.Registry.load()
    s = registry.eval_settings(args.policy, args.game_level)
    actual = reg.sha256_of(reg.REPO / s.model)
    if actual != s.model_sha256:
        raise SystemExit(
            f"{s.model} has sha256 {actual}, but the registry records "
            f"{s.model_sha256} for {args.policy}: not the registered model"
        )
    # A deterministic policy from a fixed start replays ONE trajectory N times (L3
    # notes, "v15 FINAL"), so only stochastic results are results.
    args.stochastic = True
    args.model = s.model
    args.profile = s.env.profile
    args.max_steps = s.env.max_steps
    args.stall_threshold = s.env.stall_threshold
    args.fruits_total = s.fruits_total
    args.start_state = s.start_state
    args.level = s.layout
    args.resize_mode = s.env.resize_mode
    args.reach_mode = s.reach_mode
    return s, registry


def main() -> None:
    argv = sys.argv[1:]
    p = build_parser()
    args = p.parse_args(argv)
    settings = registry = None
    if any(v is not None for v in (args.policy, args.game_level, args.config)):
        settings, registry = _from_registry(args, argv)
    elif args.model is None:
        p.error("--model is required (or use --policy with --game-level)")

    # --policy: a registered model, result written as registry data. --config: an
    # unregistered model (a referee's snapshot) evaluated under registry level
    # settings and its run's view; its output is the caller's scratch file.
    registered = settings is not None and args.policy is not None

    def _path(rel):
        # Registry paths are repo-relative; hand-typed ones stay as given.
        if rel is None or settings is None:
            return rel
        from retro_ai.training.yeti_registry import REPO

        return str(REPO / rel)

    def _model_path(rel):
        # Under --config the model is the caller's path, not a registry path.
        return _path(rel) if registered else rel

    # THE SEED THAT IS ACTUALLY USED. SB3's PPO.load calls set_random_seed with the seed
    # saved in the model (42 for every Yeti run), re-seeding random, numpy and torch. So
    # every evaluation of one model has always replayed the same action samples: two
    # 300-episode runs of the same model are the same 300 episodes, not 600 (measured:
    # with seeding before the load, seeds 123 and 124 gave identical rows). A seed given
    # here is applied AFTER the load. Without one, hand-typed runs keep the model's
    # seed, as they always had, and record it; registry runs draw a fresh seed, so
    # repeated measurements of one model are independent and can be pooled.
    seed = args.seed
    if seed is None and settings is not None:
        seed = secrets.randbits(31)

    # Respawn starts instead of the level start (see retro_ai.training.root_pool).
    pool = None
    if args.start_pool:
        from retro_ai.training import root_pool

        if settings is None:
            raise SystemExit(
                "--start-pool needs --policy or --config with --game-level"
            )
        pool = root_pool.load(args.start_pool, args.pool_split)
        if pool.game_level != args.game_level or pool.layout != settings.layout:
            raise SystemExit(
                f"{args.start_pool} is for game level {pool.game_level} (layout "
                f"{pool.layout}), not --game-level {args.game_level} (layout "
                f"{settings.layout})"
            )
        print(f"start pool: {root_pool.describe(pool)}", flush=True)

    out_path = args.out
    if registered:
        out = (
            Path(args.out)
            if args.out
            else registry.default_result_path(
                args.policy,
                args.game_level,
                args.episodes,
                seed,
                pool_tag=(
                    f"pool-{Path(args.start_pool).stem}-{args.pool_split}"
                    if pool is not None
                    else None
                ),
            )
        )
        if out.exists():
            raise SystemExit(f"{out} exists; a result file is data, never overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(out)
    elif settings is not None and not args.out:
        raise SystemExit("--config needs --out (its result is not registry data)")
    if settings is not None:
        who = f"policy {args.policy}" if registered else f"config {args.config}"
        print(
            f"{who} on game level {args.game_level}: model "
            f"{args.model}, layout {args.level}, start {args.start_state or 'reset'}, "
            f"profile "
            f"{args.profile}, stop {args.max_steps}/{args.stall_threshold}, resize "
            f"{settings.env.resize_mode}, reach {args.reach_mode}, seed {seed}",
            flush=True,
        )

    deterministic = not args.stochastic
    fruits_total = args.fruits_total
    princess_cp = fruits_total + 1
    start_state_bytes = None
    if args.start_state:
        with open(_path(args.start_state), "rb") as f:
            start_state_bytes = f.read()

    if settings is not None:
        # The policy's own view: resize, resize mode, crop, frame settings, action mode.
        env_cfg = settings.env
    else:
        env_cfg = EnvConfig(
            profile=args.profile,
            action_mode="joystick",
            max_steps=args.max_steps,
            stall_threshold=args.stall_threshold,
            resize=(84, 84),
            resize_mode=args.resize_mode,
        )
    stack = build_training_env(env_cfg.profile, env_cfg)

    model = PPO.load(_model_path(args.model), device="auto")
    if seed is None:
        seed = model.seed  # PPO.load already seeded random, numpy and torch with it
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    level = args.level if args.level is not None else (2 if start_state_bytes else 1)
    track = not args.no_depth
    # Drawn with the seeded `random`, after the seeding above, so a seed repeats the
    # same starts as well as the same actions.
    pool_rng = random.Random(seed)

    rows = []
    max_cp_counts: Counter[int] = Counter()
    rung_counts: Counter[int] = Counter()
    princess_touches = 0
    n_rungs = 0

    for ep in range(args.episodes):
        # Shared rollout harness: identical termination (princess -> death via
        # 0x2AFC -> stall -> env done -> max_steps) and CP tracking for all
        # eval/analysis scripts. Level 2 boots from a save-state.
        ep_start, ep_start_path = start_state_bytes, None
        if pool is not None:
            ep_start_path, ep_start = pool.draw(pool_rng)
        result = rollout_episode(
            stack,
            model,
            level=level,
            fruits_total=fruits_total,
            start_state=ep_start,
            max_steps=args.max_steps,
            stall_threshold=args.stall_threshold,
            deterministic=deterministic,
            track_waypoints=track,
            wp_tol=args.waypoint_tolerance,
            wp_jump_tol=args.jump_waypoint_tolerance,
            reach_mode=args.reach_mode,
        )
        max_cp_counts[result.max_cp] += 1
        rung_counts[result.max_rung] += 1
        n_rungs = result.n_rungs or n_rungs
        if result.princess_touched:
            princess_touches += 1
        rows.append(
            {
                "ep": ep,
                "max_cp": result.max_cp,
                "max_rung": result.max_rung,
                "steps": result.length,
                "end_reason": result.end_reason,
                "final_x": result.final_x,
                "final_y": result.final_y,
                "reached_points": sorted(result.reached_points),
                **({"start": ep_start_path} if pool is not None else {}),
            }
        )
        if (ep + 1) % 25 == 0:
            reach1 = sum(v for k, v in max_cp_counts.items() if k >= 1)
            reach_top = sum(v for k, v in max_cp_counts.items() if k >= fruits_total)
            print(
                f"  {ep + 1}/{args.episodes} episodes "
                f"(reach1={reach1}, reach{fruits_total}={reach_top}, "
                f"princess={princess_touches})",
                flush=True,
            )

    n = args.episodes
    print(f"\n=== from-reset eval: {args.model} ===")
    print(f"episodes={n}  policy={'deterministic' if deterministic else 'stochastic'}")
    print("\nDeepest checkpoint reached (cumulative):")
    cum = 0
    for cp in range(princess_cp, -1, -1):
        cum += max_cp_counts.get(cp, 0)
        label = "princess" if cp == princess_cp else f"{cp} fruits"
        print(
            f"  reached >= {label:>10}: {cum:>4}/{n}  ({100*cum/n:5.1f}%)"
            + (
                f"   [exactly {cp}: {max_cp_counts.get(cp,0)}]"
                if max_cp_counts.get(cp, 0)
                else ""
            )
        )
    print(f"\nprincess touches: {princess_touches}/{n} ({100*princess_touches/n:.1f}%)")

    mean_rung = (
        sum(k * v for k, v in rung_counts.items()) / n if track and rung_counts else 0.0
    )
    if track and n_rungs:
        print(f"\nRoute depth (mandatory rungs, {n_rungs} total) -- this is what")
        print("distinguishes snapshots on a level where princess is uniformly 0:")
        cum = 0
        for r in range(n_rungs, -1, -1):
            cum += rung_counts.get(r, 0)
            if cum:
                print(
                    f"  reached >= rung {r:>2}: {cum:>4}/{n}  ({100 * cum / n:5.1f}%)"
                )
        print(f"  mean rung: {mean_rung:.2f} / {n_rungs}")

    if out_path:
        from retro_ai.training.yeti_registry import current_native_sha256

        provenance = {
            "seed": seed,
            "profile": env_cfg.profile,
            "max_steps": args.max_steps,
            "stall_threshold": args.stall_threshold,
            # Which emulator build produced these numbers (see 2b0a45d in TODO.md).
            "native_sha256": current_native_sha256(),
        }
        if registered:
            provenance.update(
                policy=args.policy,
                game_level=args.game_level,
                model_sha256=settings.model_sha256,
            )
        elif settings is not None:
            provenance.update(game_level=args.game_level, config=args.config)
        if pool is not None:
            # A pooled result is a DIFFERENT measurement from the level start's; the
            # registry keeps the two apart by these keys (Registry.results).
            provenance.update(
                start_pool=args.start_pool,
                start_pool_sha256=pool.manifest_sha256,
                pool_split=args.pool_split,
                pool_sequences=len(pool.sequences),
                pool_states=pool.n_states,
            )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **provenance,
                    "model": args.model,
                    "episodes": n,
                    "deterministic": deterministic,
                    "fruits_total": fruits_total,
                    "start_state": args.start_state,
                    "level": level,
                    "max_cp_counts": dict(max_cp_counts),
                    "princess_touches": princess_touches,
                    "rung_counts": dict(rung_counts),
                    "n_rungs": n_rungs,
                    "mean_rung": mean_rung,
                    # How these numbers were measured. Both silently change every
                    # route figure, and neither was recorded before 2026-10-07.
                    "reach_mode": args.reach_mode,
                    "resize_mode": args.resize_mode,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
