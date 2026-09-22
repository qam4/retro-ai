"""Measure the REAL shaped reward on the Low2_launch pad (rope-2 launch).

Why this exists
---------------
v13 arrives at ``Low2_launch`` reliably (reach 0.48-0.72) and its ``prog``
from that pool is 0.00: it never crosses rope 2. A behavioural probe of the
v13 policy showed the agent presses LEFT+FIRE in 34/34 episodes but departs
the pad at wait 0-3 in 33/34, never in the 17..23 window that the scripted
feasibility sweep proved works (40 crossings / 3312 trials, 6/6 seeds).

So the input is right and the timing is wrong. The open question is whether
the reward actively pushes the agent off the edge immediately (a gradient
problem) or is simply flat while it stands there (an exploration problem).

An earlier attempt to answer this with ``build_training_env(...).gym.step``
returned 0.00 for every plan. That harness is wrong: the trainer does NOT use
the gym's reward. It builds its own ``RewardContext`` per step and calls
``reward_fn(ctx)`` (train_checkpoint_curriculum.py, ~line 1647). This script
reproduces that construction exactly, including the ``restore_reached_waypoints``
call that stops already-reached waypoints from being re-summed as pending
targets -- without it the reward would aim at an early waypoint, not Low2.

Read-only: loads pool seeds and a config, writes only a figure.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import pickle
import statistics
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from retro_ai.games import yeti  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.rewards import (  # noqa: E402
    RewardContext,
    create,
    reset_reward,
    restore_reached_waypoints,
)
from retro_ai.training.run_config import RunConfig  # noqa: E402

NOOP = [0, 0, 0]
LEFT = [0, 2, 0]
JUMP_L = [0, 2, 1]
JUMP_U = [0, 0, 1]

PLANS = {
    "HOLD (noop)": NOOP,
    "walk LEFT": LEFT,
    "jump-LEFT held": JUMP_L,
}


def _load_trainer():
    """Import the trainer module for its RAM address constants."""
    path = REPO / "scripts/mo5/yeti/train_checkpoint_curriculum.py"
    spec = importlib.util.spec_from_file_location("tcc", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tcc"] = mod
    spec.loader.exec_module(mod)
    return mod


class PadReward:
    """Replays the trainer's per-step RewardContext construction."""

    def __init__(self, iface, tcc, reward_fn, fruit_addrs=None):
        self.i = iface
        self.t = tcc
        self.fn = reward_fn
        # MUST come from the curriculum config, exactly as the trainer does
        # (train_checkpoint_curriculum.py ~1274). Defaulting to the level-1
        # dict makes the reward read the wrong RAM and chase a stale fruit.
        self.fruit_addrs = dict(fruit_addrs or tcc.FRUIT_PRESENCE_ADDRS)
        self.fruit_ids = sorted(self.fruit_addrs)

    def _b(self, addr):
        return self.i.read_ram_byte(addr)

    def _bonus(self):
        return self._b(self.t.BONUS_HI) * 256 + self._b(self.t.BONUS_LO)

    def _score(self):
        return self._b(self.t.SCORE_HI) * 256 + self._b(self.t.SCORE_LO)

    def snapshot(self):
        return {
            "fruits": self._b(self.t.FRUITS_ADDR),
            "lives": self._b(self.t.LIVES_ADDR),
            "bonus": self._bonus(),
            "score": self._score(),
            "princess": self._b(self.t.PRINCESS_FLAG_ADDR),
        }

    def reward(self, prev, step_count, died):
        cur = self.snapshot()
        ctx = RewardContext(
            prev_fruits=prev["fruits"],
            curr_fruits=cur["fruits"],
            prev_bonus=prev["bonus"],
            curr_bonus=cur["bonus"],
            prev_score=prev["score"],
            curr_score=cur["score"],
            prev_lives=prev["lives"],
            curr_lives=cur["lives"],
            step_count=step_count,
            curr_y=self._b(self.t.Y_POS),
            curr_x=self._b(self.t.X_POS),
            fruits_present=tuple(
                self._b(self.fruit_addrs[i]) != 0 for i in self.fruit_ids
            ),
            princess_touched=cur["princess"] == 1 and prev["princess"] == 0,
            pose=self._b(self.t.POSE_ADDR),
            died=died,
        )
        return float(self.fn(ctx)), cur


def _load_record():
    """record.py's plan vocabulary, imported rather than retyped.

    The two scripts must agree on what "JUMP_LEFT:40" means or a reward trace cannot be
    compared with the clip it came from -- which is the whole point of measuring the
    scripted crossing.
    """
    spec = importlib.util.spec_from_file_location(
        "_yeti_record", Path(__file__).resolve().parent / "record.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def roll(env, iface, pr, state, reached, plan, n):
    """Load ``state`` and apply ``plan`` for ``n`` steps, logging reward."""
    iface.load_state(bytes(state))
    env.preprocessed.notify_state_loaded()
    reset_reward(pr.fn)
    restore_reached_waypoints(pr.fn, frozenset(reached))
    obs, _, _, _, _ = env.gym.step(NOOP)
    prev = pr.snapshot()
    rows = []
    # `plan` is either ONE action held for n steps, or a per-step sequence (from
    # --plan). A sequence is what the rope-2 crossing needs: the working departure is a
    # wait of 18-22 steps and then a held jump-left, and holding either alone never
    # crosses.
    seq = isinstance(plan[0], (list, tuple))
    for t in range(n):
        act = plan[min(t, len(plan) - 1)] if seq else plan
        obs, _, te, tr, _ = env.gym.step(act)
        died = yeti.is_dead(iface)
        r, prev = pr.reward(prev, t + 1, died)
        x, y = yeti.read_pos(iface)
        rows.append(
            {
                "t": t,
                "r": r,
                "px": x * 4 + 8,
                "y": y,
                "pose": yeti.read_pose(iface),
                "died": died,
            }
        )
        if died or te or tr:
            break
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="experiments/003-yeti/configs/yeti_curriculum_l4_v13_anchors_15m.yaml",
    )
    ap.add_argument(
        "--run",
        default=("output/mo5/yeti/training/yeti_curriculum_l4_v13_anchors_15m"),
    )
    ap.add_argument("--waypoint", default="Low2_launch")
    ap.add_argument(
        "--plan",
        default=None,
        help='record.py plan syntax, e.g. "NOOP:20,JUMP_LEFT:60". Replaces the three '
        "held-input plans. Use it to price a trajectory you have a clip of.",
    )
    ap.add_argument("--steps", type=int, default=26)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="debug/l4_rope2_geom/pad_reward.png")
    args = ap.parse_args(argv)

    plans = dict(PLANS)
    if args.plan:
        plans = {args.plan: _load_record().parse_plan(args.plan)}
        args.steps = max(args.steps, len(plans[args.plan]))

    tcc = _load_trainer()
    cfg = RunConfig.from_yaml(args.config)
    env_cfg = dataclasses.replace(cfg.env, max_steps=400, stall_threshold=10**9)
    env = build_training_env(env_cfg.profile, env_cfg)
    iface = env.base._interface
    env.gym.reset()

    reward_params = dict(cfg.reward.params)
    reward_params.setdefault("gamma", cfg.ppo.gamma)
    reward_params["waypoint_reach_mode"] = cfg.curriculum.waypoint_reach_mode
    reward_fn = create(cfg.reward.name, reward_params)
    fruit_addrs = {
        int(k): int(v)
        for k, v in (
            getattr(cfg.curriculum, "fruit_presence_addrs", None) or {}
        ).items()
    }
    pr = PadReward(iface, tcc, reward_fn, fruit_addrs or None)
    print(f"fruit_presence_addrs={pr.fruit_addrs}")

    pool = pickle.load(open(Path(args.run) / "checkpoints.pkl", "rb"))
    entries = pool["waypoints"][args.waypoint][0]

    safe = []
    for e in entries:
        iface.load_state(bytes(e[2]))
        if yeti.read_pos(iface)[0] * 4 + 8 >= 188:
            safe.append(e)
    print(f"reward: {cfg.reward.name}  params={dict(cfg.reward.params)}")
    print(f"{len(safe)}/{len(entries)} {args.waypoint} seeds on safe ground (px>=188)")
    print(f"measuring {args.steps} steps under {len(plans)} plan(s)\n")

    totals = {k: [] for k in plans}
    firsts = {k: [] for k in plans}
    curves = {k: [] for k in plans}
    for e in safe[: args.seeds]:
        for name, plan in plans.items():
            rows = roll(env, iface, pr, e[2], e[4], plan, args.steps)
            totals[name].append(sum(r["r"] for r in rows))
            firsts[name].append(sum(r["r"] for r in rows[:6]))
            cum, acc = [], 0.0
            for r in rows:
                acc += r["r"]
                cum.append(acc)
            curves[name].append(cum)

    n = len(safe[: args.seeds])
    print(f"{'plan':>16} {'mean total':>11} {'median':>9} {'mean 1st 6':>11}")
    for name in plans:
        print(
            f"{name:>16} {statistics.fmean(totals[name]):>11.3f} "
            f"{statistics.median(totals[name]):>9.3f} "
            f"{statistics.fmean(firsts[name]):>11.3f}"
        )
    print(f"\n(n={n} seeds, {args.steps} steps each)")

    print("\nper-step reward, seed 0:")
    for name, plan in plans.items():
        rows = roll(env, iface, pr, safe[0][2], safe[0][4], plan, 12)
        s = " ".join(f"{r['r']:+.3f}" for r in rows)
        print(f"  {name:>16}: {s}")
        px = " ".join(f"{r['px']:>6}" for r in rows)
        print(f"  {'':>16}  px:{px}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colours = {
            "HOLD (noop)": "tab:blue",
            "walk LEFT": "tab:orange",
            "jump-LEFT held": "tab:red",
        }
        for name in plans:
            for i, c in enumerate(curves[name]):
                ax.plot(
                    range(len(c)),
                    c,
                    color=colours[name],
                    alpha=0.28,
                    lw=1,
                    label=name if i == 0 else None,
                )
        ax.axvspan(
            17, 23, color="green", alpha=0.12, label="proven departure window 17..23"
        )
        ax.set_xlabel("step held on the pad")
        ax.set_ylabel("cumulative shaped reward")
        ax.set_title(
            f"Reward on the {args.waypoint} pad ({n} pool seeds, " f"{cfg.reward.name})"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nfigure: {out}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(no figure: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Wait sweep: does a SUCCESSFUL rope-2 crossing pay anything?
#
# The pad probe above shows the only positive shaped reward on the pad is a
# single +0.12 for stepping 188 -> 184, and that 184 is the pixel the scripted
# feasibility sweep found launches 0/8. This sweep attaches the real reward to
# the plan that DOES work -- hold for W frames, then hold jump-left -- so the
# crossing's payout can be compared against that +0.12 trap.
# ---------------------------------------------------------------------------
