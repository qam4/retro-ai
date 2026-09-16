#!/usr/bin/env python3
"""Validate that every positional target can actually be MARKED. Reports, never fixes.

Motivation. On L4 the reward's milestone `Fr1` is anchored at x_ram 60 -- floor 3's
left extremity -- with `waypoint_reward_tol` 2, giving a box of 58..62. The agent only
ever occupies 64..68 on that platform, so the milestone is never marked and the
potential sums distance to it for the whole episode. Nothing flagged this; it took
eight L4 runs and a manual probe to find.

TWO TIERS, because they catch different failures and only the second catches Fr1.

  STATIC (no emulator, cheap enough to gate CI)
      How many positions in the box resolve to the target's own floor, using the
      SAME resolver the reward uses (`agent_floor_from_pixel_xy`)? This is done in
      PIXELS to avoid the conversion trap -- platform extents are tile-pixel
      boundaries while x_ram is the agent's reference point (px = x_ram*4 + 8), and
      mixing the two is wrong by 1-2 units on every platform.
        0 standable positions  -> the box is unmarkable by construction, hard error.
        box at a platform EXTREMITY -> suspicious: the agent may not go there.

  MEASURED (--measure, needs the emulator and a policy)
      Roll the policy and record every grounded position per floor, then check the
      box against what the agent actually occupies. This is what caught Fr1: its box
      IS 3/5 standable, so static passes it; the agent simply never stands there.
      Caveat: a box could be reachable in principle and merely unused by THIS policy,
      so a measured miss is a strong hint, not proof of impossibility.

Usage
    PYTHONPATH=python python3 debug/yeti_validate_targets.py       # static, all levels
    ... --level 4 --measure --models <a.zip>,<b.zip>                    # add occupancy
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

from retro_ai.games import yeti
from retro_ai.training.targets import build_targets
from retro_ai.training.yeti_map import (
    agent_floor_from_pixel_xy,
    get_level_map,
    jump_waypoints,
)

# The tolerances the two consumers actually pass today.
CURRICULUM_LADDER_TOL = 2
CURRICULUM_JUMP_TOL = 6
REWARD_TOL = 2


def standable(x_ram: int, y_px: int, floor: int, level: int) -> bool:
    """Does (x_ram, y_px) resolve to `floor`? Same helper the reward uses."""
    return agent_floor_from_pixel_xy(x_ram * 4 + 8, y_px, level) == floor


def box_positions(pos, tol_x, tol_y):
    wx, wy = pos
    for x in range(wx - tol_x, wx + tol_x + 1):
        for y in range(wy - tol_y, wy + tol_y + 1):
            yield x, y


def static_report(level: int):
    lvl = get_level_map(level)
    jump_ids = set(jump_waypoints(lvl))
    rows = []
    for t in build_targets(level):
        if t.trigger != "position" or t.pos is None or t.floor is None:
            continue
        cur_tol = CURRICULUM_JUMP_TOL if t.id in jump_ids else CURRICULUM_LADDER_TOL
        entry = {"t": t, "cur_tol": cur_tol, "jump": t.id in jump_ids}
        for name, tol in (("reward", REWARD_TOL), ("curric", cur_tol)):
            ok = [
                (x, y)
                for x, y in box_positions(t.pos, tol, tol)
                if standable(x, y, t.floor, level)
            ]
            entry[name] = (len(ok), sorted({x for x, _ in ok}))
        # is the anchor at an extremity of the standable run on its own floor?
        run = [x for x in range(-4, 84) if standable(x, t.pos[1], t.floor, level)]
        entry["run"] = (min(run), max(run)) if run else None
        rows.append(entry)
    return rows


def print_static(level: int, occupancy=None):
    rows = static_report(level)
    print(f"=============== LEVEL {level} ===============")
    hdr = (
        f"{'target':<16}{'kind':<7}{'anchor':<11}{'floor':>5}  "
        f"{'standable run':<15}{'reward box':<14}{'curric box':<14}"
    )
    if occupancy is not None:
        hdr += "occupied / marks?"
    print(hdr)
    errors, warns, suggestions = [], [], []
    for e in rows:
        t = e["t"]
        n_rew, xs_rew = e["reward"]
        n_cur, xs_cur = e["curric"]
        run = e["run"]
        kind = "JUMP" if e["jump"] else "ladder"
        line = (
            f"{t.id:<16}{kind:<7}{str(t.pos):<11}{t.floor:>5}  "
            f"{str(run):<15}{f'{n_rew} pos':<14}{f'{n_cur} pos':<14}"
        )
        note = ""
        if n_rew == 0:
            note = "REWARD BOX UNMARKABLE (0 standable)"
            errors.append((t.id, note))
        # NOTE: an "anchor sits at a platform extremity" warning was tried here and
        # removed. It fires for all 24 jump waypoints on L4 because that is what
        # `jump_waypoints` DOES -- it places every arrival and launch pad on an edge.
        # So it cannot separate Fr1 (never marks) from Spring (marks 18x): both boxes
        # overlap their platform only at the extreme end, and the difference between
        # them is where the agent goes, which is not visible in the geometry. 24
        # identical warnings is noise. Use --measure for a per-target verdict.
        if occupancy is not None:
            occ = occupancy.get(t.floor, Counter())
            if not occ:
                line += f"{'floor unvisited':<20}"
            else:
                marks_rew = any(
                    (x, y) in occ
                    for x, y in box_positions(t.pos, REWARD_TOL, REWARD_TOL)
                )
                marks_cur = any(
                    (x, y) in occ
                    for x, y in box_positions(t.pos, e["cur_tol"], e["cur_tol"])
                )
                r_ok = "Y" if marks_rew else "N"
                c_ok = "Y" if marks_cur else "N"
                line += f"reward={r_ok} curric={c_ok}  "
                if not marks_rew:
                    note = (
                        note + "; " if note else ""
                    ) + "MEASURED: reward never marks"
                    errors.append((t.id, "MEASURED: reward never marks"))
                    # PROPOSE a replacement anchor from what the agent actually does:
                    # the busiest grounded position on this floor, i.e. the one a
                    # trajectory is most likely to pass through. The proposal is then
                    # scored against the SAME occupancy, so it is known to mark before
                    # anyone edits a map. NOTE this is one policy's habit -- see the
                    # cross-policy spread recorded in level4_notes.md.
                    counts = occ
                    if counts:
                        cand, n_hits = counts.most_common(1)[0]
                        inbox = set(box_positions(cand, REWARD_TOL, REWARD_TOL))
                        covered = sum(c for p, c in counts.items() if p in inbox)
                        total = sum(counts.values())
                        suggestions.append(
                            (
                                t.id,
                                t.pos,
                                cand,
                                f"{covered}/{total} grounded steps on f{t.floor} "
                                f"({100 * covered / total:.0f}%) fall in a tol-2 box "
                                f"there; busiest position seen {n_hits}x",
                            )
                        )
        print(line + ("   <-- " + note if note else ""))
    print()
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for i, n in errors:
            print(f"    {i}: {n}")
    if warns:
        print(f"  WARNINGS ({len(warns)}):")
        for i, n in warns:
            print(f"    {i}: {n}")
    if suggestions:
        print(f"  PROPOSED ANCHORS ({len(suggestions)}) -- verified against the same")
        print(
            "  measured occupancy, so each is known to mark before any map is edited:"
        )
        for i, old, new, why in suggestions:
            print(f"    {i:<16} {old} -> {new}   {why}")
    print()
    return errors


def measure(level, profile, start_state, models, episodes, max_steps):
    """Grounded (x_ram, y) actually occupied, per floor."""
    import numpy as np
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.run_config import EnvConfig
    from stable_baselines3 import PPO

    cfg = EnvConfig(
        profile=profile,
        action_mode="joystick",
        max_steps=max_steps,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env(profile, cfg)
    ifc = stack.base._interface
    start = open(start_state, "rb").read() if start_state else None
    occ = defaultdict(Counter)
    for mp in models:
        if not os.path.exists(mp):
            print(f"  MISSING {mp}")
            continue
        model = PPO.load(mp, device="auto")
        for _ep in range(episodes):
            obs, _ = stack.gym.reset()
            if start:
                ifc.load_state(start)
                stack.preprocessed.notify_state_loaded()
                obs, *_ = stack.gym.step([0, 0, 0])
            for _s in range(max_steps):
                a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
                obs, _r, term, trunc, _i = stack.gym.step([int(v) for v in a])
                x, y = yeti.read_pos(ifc)
                if yeti.read_pose(ifc) in yeti.SURFACE_POSES:
                    f = agent_floor_from_pixel_xy(x * 4 + 8, y, level)
                    if f is not None:
                        occ[f][(x, y)] += 1
                if yeti.is_dead(ifc) or term or trunc:
                    break
        print(f"  measured with {os.path.basename(mp)}", flush=True)
    return occ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, action="append", help="default: 1 2 3 4")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    args = ap.parse_args()

    levels = args.level or [1, 2, 3, 4]
    all_errors = []
    for lvl in levels:
        occ = None
        if args.measure:
            profile = (
                args.profile or f"yeti_fruit_level{lvl}" if lvl > 1 else "yeti_fruit"
            )
            ss = args.start_state
            if ss is None and lvl > 1:
                ss = f"output/mo5/yeti/level{lvl}/level{lvl}_start.sav"
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            occ = measure(lvl, profile, ss, models, args.episodes, args.max_steps)
        all_errors += [(lvl, i, n) for i, n in print_static(lvl, occ)]
    if all_errors:
        print(f"TOTAL ERRORS: {len(all_errors)}")
        for lvl, i, n in all_errors:
            print(f"  L{lvl} {i}: {n}")
    else:
        print("no unmarkable targets found")


if __name__ == "__main__":
    main()
