"""Per-step reward under mark_airborne false vs true, across climb2->Spring and
Spring->Step, on the SAME trajectory.

Why the same trajectory
-----------------------
Two explanations for v16b's `Spring` collapse have already failed. The next-waypoint
story was wrong (`Lclimb2_top` -> `Spring` -> `Step` are consecutive groups g5, g6, g7,
nothing sits between), and premature marking is measurably zero -- 0 of 114 airborne
`Spring` touches fail to land on floor 9.

The difference left standing is TIMING. `Spring` is touched airborne in 114/150 episodes
and landed in exactly the same 114, so under the fix it is always marked strictly
earlier in the episode, never on a different set of episodes.

To isolate that, one trajectory is recorded from the policy and then REPLAYED through
two reward instances that differ only in `mark_airborne`. Same states, same actions,
same order: any divergence is the lever and nothing else. Two separate rollouts cannot
do this -- the policy is stochastic, so the trajectories would differ.

Why the episode selection is explicit
-------------------------------------
An earlier version of this script broke the rollout the moment it first stood on floor
9, so it never attempted floor 9 -> 10 and could say nothing about `Spring` -> `Step`.
It also mislabelled the arrival it captured. This version runs episodes to completion,
then locates the transitions afterwards:

  climb2 -> Spring : last grounded frame on floor 8  -> first grounded frame on floor 9
  Spring -> Step   : last grounded frame on floor 9  -> first grounded frame on floor 10

Floor 8 (y 94, px 120..168) and floor 9 (y 94, px 200..232) are separated by a gap at
px 168..200 with the trampoline (floor 24, y 142) directly beneath it, so this crossing
is not a plain jump -- per the game's design it goes over the trampoline, which reads as
poses 16/17 and is airborne. The script counts how many f8 -> f9 crossings contain a
trampoline pose rather than assuming it.

Read-only: loads a config and a trained policy, writes nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling diag modules

from l4_pad_reward import PadReward, _load_trainer  # noqa: E402
from retro_ai.games import yeti  # noqa: E402
from retro_ai.training import yeti_map as ym  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.rewards import create, reset_reward  # noqa: E402
from retro_ai.training.run_config import RunConfig  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

NOOP = [0, 0, 0]
TRAMPOLINE_POSES = frozenset({16, 17})

# the three consecutive reward groups this script is about
WATCH = {"Lclimb2_top": 5, "Spring": 6, "Step": 7}
# transitions, as (label, from_floor, to_floor)
TRANSITIONS = [
    ("climb2 -> Spring", 8, 9),
    ("Spring -> Step", 9, 10),
]


def floors():
    return {p.floor: p for p in ym.get_level_map(4).platforms}


def on_floor(px, y, pose, p):
    """Grounded on platform ``p``: surface pose, at its height, within its span."""
    return pose in yeti.SURFACE_POSES and y == p.y and p.x_min <= px <= p.x_max


def find_transition(trace, plat, f_from, f_to):
    """Return (depart, arrive) step indices for the first f_from -> f_to crossing.

    ``depart`` is the last grounded frame on ``f_from`` before the first grounded
    frame on ``f_to`` that follows it. Returns None if the crossing never happens.
    """
    pf, pt = plat[f_from], plat[f_to]
    depart = None
    for t, (px, y, pose) in enumerate(trace):
        if on_floor(px, y, pose, pf):
            depart = t
        elif depart is not None and on_floor(px, y, pose, pt):
            return depart, t
    return None


def aggregate(args, plat, rollout, replay, transitions_of) -> int:
    """Repeat the paired replay over many episodes and summarise the arrival frames.

    A single episode already showed the floor-9 arrival payment going from +3.200 to
    0.000. One episode is how the previous version of this script drew a wrong
    conclusion, so this counts how often it happens rather than assuming it always
    does. For each crossing, records the reward at the ARRIVAL frame under both
    variants and where the destination's own waypoint was marked relative to it.
    """
    stats = {label: [] for label, _, _ in TRANSITIONS}
    tramp_hits = tramp_total = 0
    dest_wp = {"climb2 -> Spring": "Spring", "Spring -> Step": "Step"}
    for ep in range(args.aggregate):
        acts, trace = rollout()
        found = transitions_of(trace)
        if not any(found.values()):
            continue
        t1 = found["climb2 -> Spring"]
        if t1:
            tramp_total += 1
            if any(p in TRAMPOLINE_POSES for _, _, p in trace[t1[0] : t1[1] + 1]):
                tramp_hits += 1
        A, ma = replay(acts, args.control_mark_airborne)
        B, mb = replay(acts, True, args.pay_on_change)
        for label, _, _ in TRANSITIONS:
            if not found[label]:
                continue
            _dep, arr = found[label]
            name = dest_wp[label]
            stats[label].append(
                {
                    "ctrl": A[arr],
                    "fix": B[arr],
                    "mark_ctrl": ma.get(name),
                    "mark_fix": mb.get(name),
                    "arr": arr,
                }
            )
        print(
            f"  ep {ep:>3}: "
            + "  ".join(
                f"{label.split()[-1]} "
                + (
                    f"@{found[label][1]:>4} "
                    f"{A[found[label][1]]:+.3f}->{B[found[label][1]]:+.3f}"
                    if found[label]
                    else "@ --- ------ ------"
                )
                for label, _, _ in TRANSITIONS
            ),
            flush=True,
        )
    print(f"\nepisodes: {args.aggregate}")
    if tramp_total:
        print(
            f"climb2->Spring crossings containing a trampoline pose (16/17): "
            f"{tramp_hits}/{tramp_total}"
        )
    for label, _, to_f in TRANSITIONS:
        rows = stats[label]
        print(f"\n=== {label}  (arrival on f{to_f}, n={len(rows)}) ===")
        if not rows:
            print("  never happened")
            continue
        killed = sum(1 for r in rows if abs(r["fix"]) < 1e-9 and r["ctrl"] > 1e-9)
        mc = statistics.fmean(r["ctrl"] for r in rows)
        mf = statistics.fmean(r["fix"] for r in rows)
        print(f"  arrival-frame reward: control mean {mc:+.3f}, fix mean {mf:+.3f}")
        print(f"  control paid >0 but fix paid exactly 0: {killed}/{len(rows)}")
        name = dest_wp[label]
        before = sum(
            1 for r in rows if r["mark_fix"] is not None and r["mark_fix"] < r["arr"]
        )
        never = sum(1 for r in rows if r["mark_fix"] is None)
        print(
            f"  {name} marked BEFORE the arrival frame under the fix: "
            f"{before}/{len(rows)}   never marked: {never}/{len(rows)}"
        )
        bc = sum(
            1 for r in rows if r["mark_ctrl"] is not None and r["mark_ctrl"] < r["arr"]
        )
        nc = sum(1 for r in rows if r["mark_ctrl"] is None)
        print(
            f"  {name} marked BEFORE the arrival frame under control:  "
            f"{bc}/{len(rows)}   never marked: {nc}/{len(rows)}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=(
            "experiments/003-yeti/configs/"
            "yeti_curriculum_l4_v16a_markground_cold_6m.yaml"
        ),
    )
    ap.add_argument(
        "--run",
        default="output/mo5/yeti/training/yeti_curriculum_l4_v16a_markground_cold_6m",
    )
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=1100)
    ap.add_argument(
        "--aggregate",
        type=int,
        default=0,
        metavar="N",
        help=(
            "instead of one detailed trace, replay every episode among N and "
            "report the arrival-frame delta distribution per transition"
        ),
    )
    ap.add_argument(
        "--pay-on-change",
        action="store_true",
        help=(
            "give the SECOND arm pay_on_target_change=true, i.e. price a frame whose "
            "waypoint list changed against the list it started with (option B)"
        ),
    )
    ap.add_argument(
        "--control-mark-airborne",
        action="store_true",
        help="first arm marks airborne too (isolates pay_on_target_change alone)",
    )
    args = ap.parse_args(argv)

    tcc = _load_trainer()
    cfg = RunConfig.from_yaml(args.config)
    env_cfg = dataclasses.replace(cfg.env, max_steps=args.horizon + 50)
    env = build_training_env(env_cfg.profile, env_cfg)
    iface = env.base._interface
    env.gym.reset()
    model = PPO.load(str(Path(args.run) / "final_model.zip"), device="cpu")
    with open(cfg.curriculum.start_state, "rb") as f:
        start = f.read()
    plat = floors()

    print(f"policy: {args.run}/final_model.zip")
    for label, a, b in TRANSITIONS:
        pa, pb = plat[a], plat[b]
        print(
            f"{label:>17}: f{a} (y {pa.y}, px {pa.x_min}..{pa.x_max}) -> "
            f"f{b} (y {pb.y}, px {pb.x_min}..{pb.x_max})"
        )
    tr = plat[24]
    print(f"{'trampoline':>17}: f24 (y {tr.y}, px {tr.x_min}..{tr.x_max})\n")

    rp = dict(cfg.reward.params)
    rp.setdefault("gamma", cfg.ppo.gamma)
    rp["waypoint_reach_mode"] = cfg.curriculum.waypoint_reach_mode
    fa = {
        int(k): int(v)
        for k, v in (
            getattr(cfg.curriculum, "fruit_presence_addrs", None) or {}
        ).items()
    }

    def rollout():
        """One stochastic episode. Returns its action sequence and (px, y, pose)."""
        env.gym.reset()
        iface.load_state(start)
        env.preprocessed.notify_state_loaded()
        obs, _, _, _, _ = env.gym.step(NOOP)
        acts, tr_ace = [], []
        for _ in range(args.horizon):
            a, _ = model.predict(obs, deterministic=False)
            act = [int(v) for v in a]
            obs, _, te, tru, _ = env.gym.step(act)
            x, y = yeti.read_pos(iface)
            acts.append(act)
            tr_ace.append((x * 4 + 8, y, yeti.read_pose(iface)))
            if yeti.is_dead(iface) or te or tru:
                break
        return acts, tr_ace

    def replay(actions, mark_airborne, pay_on_change=False):
        """Re-run ``actions`` from the start state, scoring with one reward variant."""
        p = dict(rp)
        p["mark_airborne"] = mark_airborne
        p["pay_on_target_change"] = pay_on_change
        fn = create(cfg.reward.name, p)
        pr = PadReward(iface, tcc, fn, fa or None)
        iface.load_state(start)
        env.preprocessed.notify_state_loaded()
        reset_reward(fn)
        env.gym.step(NOOP)
        prev = pr.snapshot()
        rewards, marks = [], {}
        for t, act in enumerate(actions):
            env.gym.step(act)
            r, prev = pr.reward(prev, t + 1, yeti.is_dead(iface))
            rewards.append(r)
            for name, gi in WATCH.items():
                if name not in marks and gi in fn._reached_wp:
                    marks[name] = t
        return rewards, marks

    def transitions_of(tr_ace):
        return {
            label: find_transition(tr_ace, plat, a, b) for label, a, b in TRANSITIONS
        }

    # ---- aggregate mode: how often does the lever delete an arrival payment? -----
    if args.aggregate:
        return aggregate(args, plat, rollout, replay, transitions_of)

    # ---- collect episodes, keep the first that contains BOTH transitions --------
    best = None
    seen = Counter()
    tramp_hits = tramp_total = 0
    for ep in range(args.episodes):
        acts, trace = rollout()
        found = transitions_of(trace)
        for label, v in found.items():
            if v:
                seen[label] += 1
        t1 = found["climb2 -> Spring"]
        if t1:
            tramp_total += 1
            if any(p in TRAMPOLINE_POSES for _, _, p in trace[t1[0] : t1[1] + 1]):
                tramp_hits += 1
        if all(found.values()) and best is None:
            best = (ep, acts, trace, found)
            break
    print(
        f"episodes run: {ep + 1}   with climb2->Spring: "
        f"{seen['climb2 -> Spring']}   with Spring->Step: {seen['Spring -> Step']}"
    )
    if tramp_total:
        print(
            f"climb2->Spring crossings containing a trampoline pose (16/17): "
            f"{tramp_hits}/{tramp_total}"
        )
    if best is None:
        print("\nno single episode contained BOTH transitions; nothing to replay")
        return 1
    ep, actions, trace, found = best
    print(f"\nreplaying episode {ep}, {len(actions)} steps, containing both\n")

    A, ma = replay(actions, args.control_mark_airborne)  # control arm
    B, mb = replay(actions, True, args.pay_on_change)  # treatment arm

    print(f"{'group':>13} {'mark(false)':>12} {'mark(true)':>11} {'earlier by':>11}")
    for name in WATCH:
        a, b = ma.get(name), mb.get(name)
        d = "-" if a is None or b is None else f"{a - b}"
        print(f"{name:>13} {str(a):>12} {str(b):>11} {d:>11}")

    for label, _, to_f in TRANSITIONS:
        dep, arr = found[label]
        lo, hi = max(0, dep - 4), min(len(A), arr + 5)
        print(
            f"\n=== {label}: departs f{to_f - 1} at step {dep}, "
            f"grounded on f{to_f} at step {arr} ==="
        )
        print(
            f"{'t':>5} {'px':>5} {'y':>4} {'pose':>5} {'r(false)':>10} "
            f"{'r(true)':>9} {'delta':>9}  note"
        )
        for t in range(lo, hi):
            px, y, pose = trace[t]
            note = []
            if t == dep:
                note.append(f"LAST on f{to_f - 1}")
            if t == arr:
                note.append(f"GROUNDED on f{to_f}")
            if pose in TRAMPOLINE_POSES:
                note.append("trampoline")
            for name in WATCH:
                if ma.get(name) == t:
                    note.append(f"MARK {name}(false)")
                if mb.get(name) == t:
                    note.append(f"MARK {name}(true)")
            print(
                f"{t:>5} {px:>5} {y:>4} {pose:>5} {A[t]:>+10.3f} {B[t]:>+9.3f} "
                f"{B[t] - A[t]:>+9.3f}  {', '.join(note)}"
            )
        print(
            f"  arrival frame {arr}: control {A[arr]:+.3f}, fix {B[arr]:+.3f}, "
            f"delta {B[arr] - A[arr]:+.3f}"
        )

    sa, sb = sum(A), sum(B)
    print(f"\nepisode totals: control {sa:+.3f}   fix {sb:+.3f}   delta {sb - sa:+.3f}")
    diff = [(t, B[t] - A[t]) for t in range(len(A)) if abs(B[t] - A[t]) > 1e-9]
    print(f"steps where the two rewards differ: {len(diff)}/{len(A)}")
    for t, d in diff:
        px, y, pose = trace[t]
        tag = ""
        for label, _, to_f in TRANSITIONS:
            if found[label][1] == t:
                tag = f"  <- f{to_f} arrival"
        print(f"   t={t:>5} px {px:>4} y {y:>3} pose {pose:>3} delta {d:+8.3f}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
