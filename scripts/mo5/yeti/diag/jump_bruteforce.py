"""Is a given jump reliably EXECUTABLE at all, by scripted input?

Generalises l3_a1_jump_bruteforce.py, which asked this for exactly one jump (L3
`A1_launch` -> `A1`) with the level, profile, target rows and approach direction all
hardcoded. The question recurs for every hard transition, so the geometry is now derived
from the level map and the jump is named by its endpoints.

WHY IT MATTERS. A policy failing a jump has two very different causes and they need
different fixes:

    the manoeuvre is marginal      no reward or curriculum change helps; the window is
                                   too narrow to land reliably even with perfect input
    the manoeuvre is easy          then the policy simply has not learned it, and
                                   shaping/seeding/practice is the right lever

Measured on L4 floor 11 -> 12 with three champions over identical seeds, landing rates
were 0.14 / 0.07 / 0.03 with not one always-win seed in 60 pairs, and the variance was
mostly WITHIN-seed -- i.e. the policy has agency but nobody has learned it. That is
exactly the situation where this question decides what to do next.

METHOD. From each seed, run a scripted plan and check whether the agent ends up standing
on the destination platform:

    approach A steps  ->  wait W steps  ->  hold the jump for H steps

then sweep the grid. The approach direction and jump direction are inferred from the
geometry (destination left of the seed => walk left, jump left), so a caller names two
waypoints and nothing else. Reports the best plan's landing rate, which is the ceiling a
policy could reach on that transition.

Read-only apart from stdout.

Examples
--------
    # the L4 wall: floor 11 -> floor 12
    jump_bruteforce.py --run <run> --from-pool Lclimb3_top --land-on Low1 --level 4

    # what the L3-specific predecessor did
    jump_bruteforce.py --run <l3run> --from-pool A1_launch --land-on A1 --level 3
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from retro_ai.games import yeti  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402
from retro_ai.training.targets import build_targets  # noqa: E402
from retro_ai.training.yeti_map import get_level_map  # noqa: E402

NOOP = [0, 0, 0]
WALK = {"left": [0, 2, 0], "right": [0, 1, 0]}
JUMP = {"left": [0, 2, 1], "right": [0, 1, 1]}
SURF = set(yeti.SURFACE_POSES)


def destination(level, land_on):
    """(y, x_ram_min, x_ram_max) of the platform the jump must land on."""
    tgt = {t.id: t for t in build_targets(level)}.get(land_on)
    if tgt is None:
        raise SystemExit(f"no target {land_on!r} on level {level}")
    plats = {p.floor: p for p in get_level_map(level).platforms}
    p = plats.get(tgt.floor)
    if p is None:
        raise SystemExit(f"{land_on} claims floor {tgt.floor}, which has no platform")
    # Platform extents are TILE-PIXEL boundaries; x_ram is the agent's reference point
    # (px = x_ram * 4 + 8). Convert once, here, rather than in the hot loop.
    return p.y, (p.x_min - 8) // 4, (p.x_max - 8) // 4, tgt.floor


def attempt(stack, ifc, state, plan, approach, land, max_steps):
    y_t, xlo, xhi, _ = land
    ifc.load_state(bytes(state))
    stack.preprocessed.notify_state_loaded()
    stack.gym.step(NOOP)
    for i in range(max_steps):
        # after the scripted plan runs out, keep walking the approach direction so a
        # landing just short of the platform still resolves rather than hanging
        stack.gym.step(plan[i] if i < len(plan) else WALK[approach])
        x = ifc.read_ram_byte(yeti.X_ADDR)
        y = ifc.read_ram_byte(yeti.Y_ADDR)
        pose = ifc.read_ram_byte(yeti.POSE_ADDR)
        if pose in SURF and abs(y - y_t) <= 2 and xlo <= x <= xhi:
            return True
        if yeti.is_dead(ifc):
            return False
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir holding checkpoints.pkl")
    ap.add_argument("--from-pool", required=True, help="seed pool to launch from")
    ap.add_argument("--land-on", required=True, help="waypoint whose floor must be hit")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--approach", default="", help="left|right; default: from geometry")
    args = ap.parse_args(argv)

    land = destination(args.level, args.land_on)
    y_t, xlo, xhi, floor_t = land

    pool = pickle.load(open(Path(args.run) / "checkpoints.pkl", "rb"))["waypoints"]
    if args.from_pool not in pool or not pool[args.from_pool][0]:
        raise SystemExit(f"pool {args.from_pool!r} missing or empty in {args.run}")
    seeds = list(pool[args.from_pool][0])[: args.n_seeds]

    profile = f"yeti_fruit_level{args.level}" if args.level > 1 else "yeti_fruit"
    cfg = EnvConfig(
        profile=profile,
        action_mode="joystick",
        max_steps=args.max_steps + 20,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env(profile, cfg)
    ifc = stack.base._interface
    stack.base.reset(seed=0)

    # Infer the direction from where the seeds actually are, not from the anchor: the
    # pool is what the run captured, and it is the position a policy would jump from.
    ifc.load_state(bytes(seeds[0][2]))
    x0 = ifc.read_ram_byte(yeti.X_ADDR)
    approach = args.approach or ("left" if (xlo + xhi) / 2 < x0 else "right")
    print(f"  {len(seeds)} {args.from_pool} seeds at x_ram {x0} (px {x0 * 4 + 8})")
    print(
        f"  must land on floor {floor_t}: y {y_t}, x_ram {xlo}..{xhi} "
        f"(px {xlo * 4 + 8}..{xhi * 4 + 8})"
    )
    print(f"  approach/jump direction: {approach}\n")

    results = []
    for a in (0, 2, 4, 6, 8):
        for w in (0, 2, 5):
            for h in (2, 4, 6, 8, 10):
                plan = [WALK[approach]] * a + [NOOP] * w + [JUMP[approach]] * h
                ok = sum(
                    attempt(stack, ifc, e[2], plan, approach, land, args.max_steps)
                    for e in seeds
                )
                results.append((ok, a, w, h))
    results.sort(reverse=True)
    n = len(seeds)
    print(f"  top plans (landed/{n}, approach steps, wait, hold):")
    for ok, a, w, h in results[:10]:
        print(f"    {ok:>3}/{n}   approach={a:<2} wait={w:<2} hold={h}")
    best = results[0][0]
    print(f"\n  BEST {best}/{n} = {100 * best / n:.0f}%  over {len(results)} plans")
    print(
        "  reading: a high best rate means the manoeuvre is executable and the policy\n"
        "  simply has not learned it (shape/seed/practise it). A low best rate means\n"
        "  the window is narrow and no reward change will fix it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
