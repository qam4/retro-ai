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
import collections
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


def attempt(stack, ifc, state, plan, approach, land, max_steps, jump_at):
    """(outcome, launch_px, launch_pose) for one scripted plan.

    ``outcome`` is LANDED, DIED or NO_RESOLVE. ``jump_at`` is the plan index at
    which the JUMP input first appears, so the caller learns WHERE the agent
    actually was when it launched.

    WHY THE LAUNCH PIXEL IS RETURNED, not just a yes/no. A jump's executability is
    a property of the departure pixel, and the APPROACH does not move the agent a
    predictable distance: the walk cycle stalls (dx in {-4, 0}), so on L4 nine left
    steps from px 220 reach px 196 and it takes twelve to reach px 188. A sweep
    over approach lengths that looks like it brackets the launch pixel may never
    touch it. Measured: `--approaches 7,8,9` from `Low1` launched from px
    200/200/196 and scored 0 of 102 plans, which was read as "this jump cannot be
    done from here"; `--approaches 11,12,13` crosses on 26 of 136.

    HOW THAT ZERO HID. An approach that overshoots floor 12's edge does not fail
    loudly: it falls onto the spring at `Platform(24, 142, 168, 200)`, which delays
    death to a measured median of step 82, so the attempt spends most of its budget
    bouncing and then reports DIED like any mistimed jump. Nothing in the old
    output distinguished "the jump missed" from "the agent was never on the pad".
    That is how family C's "0 crossings" in level4_notes.md stood for weeks.

    NO_RESOLVE means alive at the end of ``max_steps`` without ever standing on the
    target. It is reported separately because it is a bad budget rather than a
    failed jump; it did not fire in the L4 grids measured on 2026-09-29, where
    every failure died.
    """
    y_t, xlo, xhi, _ = land
    ifc.load_state(bytes(state))
    stack.preprocessed.notify_state_loaded()
    stack.gym.step(NOOP)
    launch_px = launch_pose = None
    for i in range(max_steps):
        if i == jump_at:
            launch_px = ifc.read_ram_byte(yeti.X_ADDR) * 4 + 8
            launch_pose = ifc.read_ram_byte(yeti.POSE_ADDR)
        # after the scripted plan runs out, keep walking the approach direction so a
        # landing just short of the platform still resolves rather than hanging
        stack.gym.step(plan[i] if i < len(plan) else WALK[approach])
        x = ifc.read_ram_byte(yeti.X_ADDR)
        y = ifc.read_ram_byte(yeti.Y_ADDR)
        pose = ifc.read_ram_byte(yeti.POSE_ADDR)
        if pose in SURF and abs(y - y_t) <= 2 and xlo <= x <= xhi:
            return "LANDED", launch_px, launch_pose
        if yeti.is_dead(ifc):
            return "DIED", launch_px, launch_pose
    return "NO_RESOLVE", launch_px, launch_pose


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir holding checkpoints.pkl")
    ap.add_argument("--from-pool", required=True, help="seed pool to launch from")
    ap.add_argument("--land-on", required=True, help="waypoint whose floor must be hit")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--approach", default="", help="left|right; default: from geometry")
    # The default grid is sized for a PLATFORM-TO-PLATFORM hop. A rope is a different
    # timescale: the agent must wait for the rope to swing within reach and then hold
    # the carry for the whole traverse. L4's rope 2 crosses under `wait 20, hold 70`,
    # which the default grid (wait <= 5, hold <= 10) cannot express -- so it scored
    # 0/12 at px 188/192, a departure pixel independently measured to cross. A zero
    # from a grid that excludes the known-good plan says nothing about the jump.
    ap.add_argument("--approaches", default="0,2,4,6,8")
    ap.add_argument("--waits", default="0,2,5")
    ap.add_argument("--holds", default="2,4,6,8,10")
    ap.add_argument(
        "--seed-px",
        default="",
        help="comma-separated departure px to keep (e.g. 184). Pools are MIXED -- L4's "
        "`Low2_launch` holds 70 seeds on px 184 and 30 on px 188..224 -- and a jump's "
        "executability is a property of the departure pixel, so a pooled rate averages "
        "two different questions. Without this the tool reports the mixture and prints "
        "the departure px only of seeds[0], which reads as if the pool were uniform.",
    )
    args = ap.parse_args(argv)

    land = destination(args.level, args.land_on)
    y_t, xlo, xhi, floor_t = land

    pool = pickle.load(open(Path(args.run) / "checkpoints.pkl", "rb"))["waypoints"]
    if args.from_pool not in pool or not pool[args.from_pool][0]:
        raise SystemExit(f"pool {args.from_pool!r} missing or empty in {args.run}")
    all_seeds = list(pool[args.from_pool][0])

    profile = f"yeti_fruit_level{args.level}" if args.level > 1 else "yeti_fruit"
    cfg = EnvConfig(
        profile=profile,
        action_mode="joystick",
        # NEVER let the env truncate. `attempt` reloads a save-state per plan but
        # does NOT reset the gym, so the env's own step counter accumulates across
        # the whole sweep; a per-plan bound here would cut later attempts short and
        # make a result depend on a plan's POSITION in the sweep. `--max-steps`
        # still bounds each attempt in the loop below, so this only stops the env
        # cutting one short.
        #
        # An earlier version of this comment blamed this for a 40-plan and a
        # 131-plan sweep disagreeing about the rope-1 window. That was wrong: the
        # cause was the report slicing to `results[:10]`, which dropped the lowest
        # winning waits. Re-tested after both fixes -- wait=1 three times in one
        # sweep, including after an intervening 190-step plan, gives 1/1 each time,
        # so there is no order dependence to explain.
        max_steps=10**6,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env(profile, cfg)
    ifc = stack.base._interface
    stack.base.reset(seed=0)

    # Read every seed's departure px, so the selection and the report are both honest
    # about a mixed pool rather than extrapolating from seeds[0].
    px_of = []
    for e in all_seeds:
        ifc.load_state(bytes(e[2]))
        px_of.append(ifc.read_ram_byte(yeti.X_ADDR) * 4 + 8)
    want = {int(v) for v in args.seed_px.split(",") if v.strip()}
    picked = [(e, px) for e, px in zip(all_seeds, px_of) if not want or px in want]
    if not picked:
        raise SystemExit(
            f"no {args.from_pool} seed at px {sorted(want)}; pool has "
            f"{dict(sorted(collections.Counter(px_of).items()))}"
        )
    picked = picked[: args.n_seeds]
    seeds = [e for e, _ in picked]
    hist = collections.Counter(px for _, px in picked)

    # Infer the direction from where the seeds actually are, not from the anchor: the
    # pool is what the run captured, and it is the position a policy would jump from.
    ifc.load_state(bytes(seeds[0][2]))
    x0 = ifc.read_ram_byte(yeti.X_ADDR)
    approach = args.approach or ("left" if (xlo + xhi) / 2 < x0 else "right")
    print(
        f"  pool {args.from_pool}: {len(all_seeds)} seeds, px "
        f"{dict(sorted(collections.Counter(px_of).items()))}"
    )
    print(
        f"  using {len(seeds)} seeds, px {dict(sorted(hist.items()))} "
        f"(x_ram {x0} = px {x0 * 4 + 8})"
    )
    print(
        f"  must land on floor {floor_t}: y {y_t}, x_ram {xlo}..{xhi} "
        f"(px {xlo * 4 + 8}..{xhi * 4 + 8})"
    )
    print(f"  approach/jump direction: {approach}\n")

    def _ints(s):
        return [int(v) for v in s.split(",") if v.strip()]

    approaches, waits, holds = (
        _ints(args.approaches),
        _ints(args.waits),
        _ints(args.holds),
    )
    longest = max(approaches) + max(waits) + max(holds)
    if longest > args.max_steps:
        raise SystemExit(
            f"--max-steps {args.max_steps} is shorter than the longest plan "
            f"({longest} steps); the plan would be cut off mid-jump"
        )

    results = []
    for a in approaches:
        for w in waits:
            for h in holds:
                plan = [WALK[approach]] * a + [NOOP] * w + [JUMP[approach]] * h
                outs = [
                    attempt(
                        stack, ifc, e[2], plan, approach, land, args.max_steps, a + w
                    )
                    for e in seeds
                ]
                ok = sum(1 for o, _, _ in outs if o == "LANDED")
                unresolved = sum(1 for o, _, _ in outs if o == "NO_RESOLVE")
                launched = collections.Counter(px for _, px, _ in outs)
                results.append((ok, a, w, h, unresolved, launched))
    # Sort on the numeric fields only; the Counter is not orderable and two plans
    # can tie on (ok, a, w, h) when a wait or hold is repeated on the command line.
    results.sort(key=lambda r: r[:4], reverse=True)
    n = len(seeds)
    # Print EVERY plan that landed, not the top 10. A wide wait sweep is how a
    # rope's period gets measured, and that needs the full set of winning waits. A
    # 131-plan sweep with 14 winners had its four lowest cut off by a `[:10]`
    # slice, which read as the short and long sweeps disagreeing about the window
    # position, and cost a session hunting a state leak that was not there.
    shown = [r for r in results if r[0]] or results[:10]
    print(f"  plans that landed (of {n} seeds; approach steps, wait, hold):")
    for ok, a, w, h, unres, launched in shown:
        px_note = ",".join(
            f"{k}x{v}" for k, v in sorted(launched.items()) if k is not None
        )
        tail = f"  launch_px {px_note or '?'}"
        if unres:
            tail += f"  never-resolved {unres}"
        print(f"    {ok:>3}/{n}   approach={a:<2} wait={w:<3} hold={h:<3}{tail}")
    if len(shown) > 10:
        print(f"    ({len(shown)} winning plans)")

    best = results[0][0]
    all_launched: collections.Counter = collections.Counter()
    all_unres = 0
    for _ok, _a, _w, _h, unres, launched in results:
        all_launched.update({k: v for k, v in launched.items() if k is not None})
        all_unres += unres
    print(f"\n  BEST {best}/{n} = {100 * best / n:.0f}%  over {len(results)} plans")
    print(
        "  launch pixels this grid actually used: "
        f"{dict(sorted(all_launched.items()))}"
    )
    if all_unres:
        print(
            f"  {all_unres} attempt(s) ended ALIVE at the step budget without ever\n"
            "  standing on the target -- raise --max-steps before reading those as\n"
            "  misses."
        )
    print(
        "  reading: a high best rate means the manoeuvre is executable and the policy\n"
        "  simply has not learned it (shape/seed/practise it). A low best rate means\n"
        "  the window is narrow and no reward change will fix it -- BUT check the\n"
        "  launch pixels above first. A zero from a grid that never launched from\n"
        "  the departure pixel you meant to test says nothing about the jump."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
