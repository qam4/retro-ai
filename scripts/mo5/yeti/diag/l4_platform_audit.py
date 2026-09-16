"""Audit the level-4 nav-map platform bounds against measured standable spans.

The finding that motivates this
------------------------------
``yeti_map`` declares ``Platform(floor=12, y=70, x_min=184, x_max=232)`` but the
agent cannot stand on px 184: stepping there reads pose 11 (falling) on arrival,
and 20/20 pool seeds captured there die, at a median of step 82, via the
fall-bounce loop on the spring below. Measured standable span is px 188..224.

Because the PBRS path-progress potential uses nav distance along the map, px 184
is the LOWEST-distance position on floor 12 (56 to ``J12_13_b`` versus 60 at
px 188). So the shaping pays the agent +0.12 to step onto the one pixel that
kills it, and pays 0.000 for anything else on that pad.

If the map's bounds are wrong on other floors, the same lure exists elsewhere.
This script measures the true span of every floor we hold pool seeds for, by
loading a seed and walking to each edge until the agent stops being grounded at
its original y, then diffs that against the map.

Read-only.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import pickle
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from retro_ai.games import yeti  # noqa: E402
from retro_ai.training import yeti_map as ym  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import RunConfig  # noqa: E402

NOOP = [0, 0, 0]
LEFT = [0, 2, 0]
RIGHT = [0, 1, 0]


def _px(iface):
    return yeti.read_pos(iface)[0] * 4 + 8


def _grounded(iface):
    return yeti.read_pose(iface) in yeti.SURFACE_POSES


def _holds(env, iface, y0, hold):
    """Does the CURRENT state still stand at y0 after ``hold`` idle steps?

    A one-frame grounded read is not standing. Walking onto floor 12's px 184
    reads a surface pose on arrival and then falls, which is exactly the blind
    spot that lets doomed px-184 captures through admission. So every candidate
    px must survive an idle hold before it counts as standable.
    """
    probe = env.base.save_state()
    ok = True
    for _ in range(hold):
        env.gym.step(NOOP)
        if yeti.is_dead(iface) or not _grounded(iface):
            ok = False
            break
        if yeti.read_pos(iface)[1] != y0:
            ok = False
            break
    iface.load_state(bytes(probe))
    env.preprocessed.notify_state_loaded()
    return ok


def walk_edge(env, iface, state, direction, limit=80, hold=10):
    """Walk one way; return the last px the agent can actually STAND on."""
    iface.load_state(bytes(state))
    env.preprocessed.notify_state_loaded()
    env.gym.step(NOOP)
    y0 = yeti.read_pos(iface)[1]
    if not _grounded(iface):
        return None
    last = _px(iface)
    for _ in range(limit):
        env.gym.step(direction)
        if yeti.is_dead(iface):
            return last
        x, y = yeti.read_pos(iface)
        if not _grounded(iface) or y != y0:
            return last
        if not _holds(env, iface, y0, hold):
            return last
        last = x * 4 + 8
    return last


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="experiments/003-yeti/configs/yeti_curriculum_l4_v13_anchors_15m.yaml",
    )
    ap.add_argument(
        "--run",
        default="output/mo5/yeti/training/yeti_curriculum_l4_v13_anchors_15m",
    )
    ap.add_argument("--per-wp", type=int, default=3)
    args = ap.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config)
    env_cfg = dataclasses.replace(cfg.env, max_steps=600, stall_threshold=10**9)
    env = build_training_env(env_cfg.profile, env_cfg)
    iface = env.base._interface
    env.gym.reset()

    lvl = ym.get_level_map(4)
    by_floor = {p.floor: p for p in lvl.platforms}

    pool = pickle.load(open(Path(args.run) / "checkpoints.pkl", "rb"))["waypoints"]
    # measured[(y, map_floor)] -> [min_px, max_px]
    measured: dict = collections.defaultdict(lambda: [None, None])
    seen_wp: dict = collections.defaultdict(set)

    for wp, slots in pool.items():
        ents = slots[0][: args.per_wp]
        for e in ents:
            iface.load_state(bytes(e[2]))
            env.preprocessed.notify_state_loaded()
            env.gym.step(NOOP)
            if not _grounded(iface):
                continue
            x, y = yeti.read_pos(iface)
            px = x * 4 + 8
            # identify the floor by y AND px (floors share standing y)
            floor = None
            for f, p in by_floor.items():
                if p.y == y and p.x_min - 8 <= px <= p.x_max + 8:
                    floor = f
                    break
            if floor is None:
                continue
            lo = walk_edge(env, iface, e[2], LEFT)
            hi = walk_edge(env, iface, e[2], RIGHT)
            if lo is None or hi is None:
                continue
            key = (y, floor)
            cur = measured[key]
            cur[0] = lo if cur[0] is None else min(cur[0], lo)
            cur[1] = hi if cur[1] is None else max(cur[1], hi)
            seen_wp[key].add(wp)

    print("level-4 platform bounds: MAP vs MEASURED (walked to each edge)\n")
    print(
        f"{'floor':>5} {'y':>4} {'map x_min':>10} {'map x_max':>10} "
        f"{'meas min':>9} {'meas max':>9} {'dl':>4} {'dr':>4}  waypoints"
    )
    bad = []
    for (y, floor), (lo, hi) in sorted(measured.items(), key=lambda kv: kv[0][1]):
        p = by_floor[floor]
        dl = lo - p.x_min
        dr = hi - p.x_max
        flag = "" if (dl == 0 and dr == 0) else "  <-- MISMATCH"
        if dl or dr:
            bad.append((floor, p.x_min, p.x_max, lo, hi, dl, dr))
        wps = ",".join(sorted(seen_wp[(y, floor)]))
        print(
            f"{floor:>5} {y:>4} {p.x_min:>10} {p.x_max:>10} {lo:>9} {hi:>9} "
            f"{dl:>+4} {dr:>+4}  {wps}{flag}"
        )

    print(
        "\ndl = measured_min - map_x_min   (positive => map claims px the agent "
        "cannot stand on, on the LEFT)"
    )
    print(
        "dr = measured_max - map_x_max   (negative => map claims px the agent "
        "cannot stand on, on the RIGHT)"
    )
    if bad:
        print(f"\n{len(bad)} floor(s) where the map is wider than reality:")
        for floor, xmin, xmax, lo, hi, dl, dr in bad:
            print(f"   floor {floor:>2}: map {xmin}..{xmax}  measured {lo}..{hi}")
        print(
            "\nEvery such over-wide edge is a spot the path-progress potential "
            "rewards\nreaching but the agent cannot stand on."
        )
    else:
        print("\nno mismatches on the floors covered by pool seeds")
    print(
        "\nNOTE: only floors we hold pool seeds for are covered; floors past the "
        "frontier\n(the Hi chain, the princess floor) are untested here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
