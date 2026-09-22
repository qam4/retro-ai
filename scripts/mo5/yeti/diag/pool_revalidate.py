#!/usr/bin/env python3
"""Drop seeds from a checkpoints.pkl that the CURRENT admission gate would refuse.

WHY THIS EXISTS. Pools are inherited by every warm start, so a seed admitted under an
older, looser criterion is carried forward for the life of the lineage and there is no
mechanism that ever re-examines it. Measured on L4's `Low2_launch` pool, loading each
seed and reading its position as saved:

    v13   33/100 seeds on px 184
    v18   ...
    v21   87/100 seeds on px 184        <- inherited v13's 33% and made it worse

px 184 is floor 12's tile edge. The agent reads GROUNDED there for one frame and then
falls; px 188 survives a NOOP hold 8/8 and px 184 survives 0/8. The trampoline
underneath then bounces the falling agent for a MEDIAN OF 82 STEPS before it dies,
against `min_survival_steps: 30` -- so the survival gate admits every one of them.
Traced 150 steps under hold-left, hold-right and NOOP, no seed at px 184 ever gets
past px 176 while floor 13's edge is px 128, i.e. the fall-bounce loop cannot reach
the landing at all. The pool meant to teach the rope-2 crossing teaches falling
instead.

`admit_requires_grounded` fixes admission going forward -- measured on v21's pool it
rejects exactly the 90 seeds that die and keeps the 10 that live, because at the end
of the window the falling seeds read pose 17 (the trampoline's vertical lift), which
is not in `SEED_POSES`. It does nothing about seeds already in the file. This tool is
that half.

!! THE NOOP CRITERION IS NOT SOUND AS A BLANKET FILTER. Measured: it drops 100/100 of
L4's `Fr1_launch` pool, whose waypoint the agent clears 98% of the time from reset.
Those seeds sit at px 228 and die at EXACTLY step 15 under NOOP, all of them -- a
deterministic hazard -- but in real play the agent jumps to `Fr1` within a few frames
and lives. Safety under NOOP and survivability under play are different properties, and
only the second is what a start state needs. The trainer's own gate gets this right
because it judges the producing episode's pose, not a NOOP hold.

So `--only` is REQUIRED and nothing is written without `--write`. Use this to MEASURE a
named pool you already have edge data for, not to clean a run wholesale. For L4's
`Low2_launch` the position evidence is independent of NOOP: px 184 is 0/8 survivable
walking either direction, px 188-224 is 8/8.

CRITERION, and how it differs from the trainer's. The trainer judges a capture by the
PRODUCING EPISODE's play. Here there is no episode to replay, so each seed is judged by
holding NOOP -- which removes the agent's agency and asks the narrower question a start
state actually has to pass: is this state survivable at all? A seed is kept when it

    1. stays alive for the whole window (`--window`, default 30 =
       min_survival_steps), and
    2. ends the window in SEED_POSES (a surface pose, or the L3 escalator ride).

Runs as its OWN PROCESS on purpose: the Crayon emulator keeps in-process global
state, so this must not share a process with training. That is the same reason
keep_best_sweep.py shells out per snapshot.

Writes a new pkl and never modifies the input. Point a run at it with
`training.resume_pools`, which keeps the change config-reproducible and leaves the
original pools available as the control arm.

Example::

    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms python3 \\
      scripts/mo5/yeti/diag/pool_revalidate.py \\
        --pools output/mo5/yeti/training/<run>/checkpoints.pkl \\
        --out   output/mo5/yeti/pools/<run>_clean.pkl \\
        --level 4 --profile yeti_fruit_level4 --stall-threshold 40
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple

# Mirrors train_checkpoint_curriculum.py. Kept as a literal rather than imported
# because that module builds a trainer on import of its CLI path; if the trainer's
# set ever changes, test_pool_revalidate_matches_trainer_seed_poses fails.
POSE_ESCALATOR_RIDE = 13

STATE_IDX = 2  # pool entries are (source_cp, bonus, state_bytes, dict, frozenset)


def _verdict(stack, ifc, yeti, seed_poses, state: bytes, window: int):
    """(kept, steps_survived, end_pose, px, y) for one seed under NOOP."""
    ifc.load_state(bytes(state))
    stack.preprocessed.notify_state_loaded()
    x0, y0 = yeti.read_pos(ifc)
    px0 = x0 * 4 + 8
    stack.gym.step([0, 0, 0])
    for i in range(window):
        stack.gym.step([0, 0, 0])
        if yeti.is_dead(ifc):
            return False, i, None, px0, y0
    end_pose = int(ifc.read_ram_byte(yeti.POSE_ADDR))
    return end_pose in seed_poses, window, end_pose, px0, y0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pools", required=True, help="input checkpoints.pkl")
    p.add_argument("--out", required=True, help="output pkl (never overwrites --pools)")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--stall-threshold", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument(
        "--window",
        type=int,
        default=30,
        help="NOOP steps a seed must survive; match the run's min_survival_steps",
    )
    p.add_argument(
        "--only",
        default=None,
        required=True,
        help="comma-separated pool names to check. REQUIRED -- see the NOOP caveat in "
        "the module docstring; a blanket sweep guts healthy pools.",
    )
    p.add_argument("--report", default=None, help="write the per-pool tally as JSON")
    p.add_argument(
        "--write",
        action="store_true",
        help="actually write --out. Default is report-only.",
    )
    args = p.parse_args()

    if os.path.abspath(args.pools) == os.path.abspath(args.out):
        raise SystemExit(
            "--out must differ from --pools; this tool never edits in place"
        )

    from retro_ai.games import yeti
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.run_config import EnvConfig

    seed_poses = frozenset(yeti.SURFACE_POSES | {POSE_ESCALATOR_RIDE})

    with open(args.pools, "rb") as fh:
        data = pickle.load(fh)

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.stall_threshold,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    ifc = stack.base._interface

    wanted = set(args.only.split(",")) if args.only else None
    report: Dict[str, Any] = {
        "pools": args.pools,
        "window": args.window,
        "seed_poses": sorted(seed_poses),
        "tally": {},
    }

    def sweep(name: str, entries: List[Tuple]) -> List[Tuple]:
        if not entries or (wanted is not None and name not in wanted):
            return entries
        kept: List[Tuple] = []
        dropped_px: Counter = Counter()
        dropped_pose: Counter = Counter()
        died = 0
        for e in entries:
            ok, steps, end_pose, px, _y = _verdict(
                stack, ifc, yeti, seed_poses, e[STATE_IDX], args.window
            )
            if ok:
                kept.append(e)
            else:
                dropped_px[px] += 1
                if end_pose is None:
                    died += 1
                else:
                    dropped_pose[end_pose] += 1
        n_drop = len(entries) - len(kept)
        report["tally"][name] = {
            "before": len(entries),
            "kept": len(kept),
            "dropped": n_drop,
            "died_in_window": died,
            "dropped_px": dict(dropped_px),
            "dropped_end_pose": dict(dropped_pose),
        }
        flag = "" if n_drop == 0 else "   <-- dropped"
        print(
            f"  {name:<16} {len(entries):>4} -> {len(kept):>4}  "
            f"dropped={n_drop:<4} died_in_window={died:<4}"
            f"{flag}",
            flush=True,
        )
        if n_drop:
            print(
                f"      dropped by px: {dict(dropped_px)}  "
                f"end_pose: {dict(dropped_pose)}",
                flush=True,
            )
        return kept

    print(f"revalidating {args.pools}")
    print(
        f"  window={args.window} NOOP steps, keep if alive AND end pose in "
        f"{sorted(seed_poses)}\n"
    )

    print("waypoint pools:")
    for name in sorted(data.get("waypoints", {})):
        entries, score = data["waypoints"][name]
        data["waypoints"][name] = (sweep(name, list(entries)), score)

    print("\ncheckpoint pools:")
    for i, entries in enumerate(list(data.get("checkpoints", []))):
        data["checkpoints"][i] = sweep(f"cp{i}", list(entries))

    before = sum(t["before"] for t in report["tally"].values())
    kept = sum(t["kept"] for t in report["tally"].values())
    print(f"\nTOTAL {before} -> {kept}  (dropped {before - kept})")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote report {args.report}")

    if not args.write:
        print("report-only (pass --write to produce the pkl)")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(data, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
