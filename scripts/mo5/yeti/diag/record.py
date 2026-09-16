"""Record gameplay video. Replaces 13 near-identical one-off video scripts.

WHY THIS EXISTS
---------------
Thirteen scripts in debug/ recorded video, and every one of them was the same four
choices with the plumbing retyped around them:

    source     from reset, or from a seed in a run's checkpoints.pkl pool
    actor      a trained policy, or a fixed scripted input plan
    selection  keep every episode, or only ones where something happened
    overlay    burn px / y / pose (and reward, when a reward is being traced) in

They diverged only in which waypoint, which pose, which floor. That is argument
territory, not new-file territory -- and each new copy re-derived the frame grab and
the mp4 write, which is where the bugs were.

THE BUG THIS FILE PRESERVES THE FIX FOR. `stamp()` below keeps a comment that cost
real time: folding the caption strip into `resize()` stretches the PLAYFIELD into it
instead of adding space underneath. Every clip produced before that was found was
vertically stretched by ~5%, which is actively misleading when you are judging whether
a sprite is on a ledge. Scale first, then paste onto a taller canvas.

WHAT IT DELIBERATELY DOES NOT DO. It does not print reward traces or measure anything.
Those belong in the diagnostic that is asking the question (see l4_spring_trace.py for
the reward-replay pattern). This writes video and tells you which episodes matched.

Examples
--------
    # the policy from reset, keep the 3 episodes that used the rope carry
    record.py --model <run>/best/best_model.zip --episodes 200 \
        --want pose=14,15 --keep 3

    # a scripted launch off the rope-2 pad, from pool seeds
    record.py --from pool:Low2_launch --run <run> --plan "NOOP:18,JUMP_LEFT:40" \
        --episodes 8

    # everything the policy does from a Spring seed, no filtering
    record.py --from pool:Spring --run <run> --model <m> --episodes 4
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from retro_ai.games import yeti  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import RunConfig  # noqa: E402
from retro_ai.training.targets import build_targets, reaches  # noqa: E402

SCALE = 3
DEFAULT_CONFIG = (
    "experiments/003-yeti/configs/yeti_curriculum_l4_v16c_payonchange_cold_6m.yaml"
)

# Action = [vertical, horizontal, fire]; horizontal 1 = right, 2 = left. Taken from the
# constants the debug scripts agreed on, not from the action-space declaration.
# DOWN is INFERRED by symmetry with LEFT and is untested -- no script ever used it.
VOCAB = {
    "NOOP": [0, 0, 0],
    "UP": [1, 0, 0],
    "DOWN": [2, 0, 0],
    "RIGHT": [0, 1, 0],
    "LEFT": [0, 2, 0],
    "FIRE": [0, 0, 1],
    "JUMP_UP": [0, 0, 1],
    "JUMP_RIGHT": [0, 1, 1],
    "JUMP_LEFT": [0, 2, 1],
}


def stamp(raw, px, y, pose, note="", extra=""):
    """Scale a raw frame and caption it below the playfield."""
    img = Image.fromarray(np.asarray(raw, np.uint8)).convert("RGB")
    # Folding the 30-px caption strip into resize() STRETCHES the playfield into it
    # rather than adding space below -- a 5% vertical stretch for a 200-row frame at
    # SCALE 3. Harmless for the caption, but it silently misrepresents every position
    # you would use the video to judge. Scale by SCALE only, then paste onto a taller
    # canvas.
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + 30), (0, 0, 0))
    canvas.paste(img, (0, 0))
    img = canvas
    d = ImageDraw.Draw(img)
    d.text(
        (5, img.height - 26), f"px {px} y {y} pose {pose} {note}", fill=(255, 255, 0)
    )
    if extra:
        d.text((5, img.height - 14), extra, fill=(0, 255, 255))
    return np.asarray(img)


def parse_plan(spec):
    """ "LEFT:4,JUMP_LEFT:40" -> a flat list of actions."""
    out = []
    for chunk in spec.split(","):
        name, _, n = chunk.strip().partition(":")
        name = name.strip().upper()
        if name not in VOCAB:
            raise SystemExit(f"unknown action {name!r}; known: {sorted(VOCAB)}")
        out.extend([VOCAB[name]] * int(n or 1))
    return out


class Want:
    """A predicate over one episode's trace. Absent => every episode matches."""

    def __init__(self, expr, level):
        self.expr, self.target = expr, None
        if expr is None:
            return
        kind, _, val = expr.partition("=")
        self.kind = kind.strip().lower()
        if self.kind == "pose":
            self.poses = {int(v) for v in val.split(",")}
        elif self.kind == "reach":
            hits = [t for t in build_targets(level) if t.id == val.strip()]
            if not hits:
                raise SystemExit(f"no target {val!r} on level {level}")
            self.target = hits[0]
        elif self.kind == "floor":
            self.y = int(val)
        elif self.kind not in ("died", "survived"):
            raise SystemExit(
                "--want must be pose=N[,N] | reach=NAME | floor=Y | died | survived"
            )

    def match(self, trace, died):
        """``trace`` is a list of (t, action, px, y, pose, x_ram)."""
        if self.expr is None:
            return True, 0
        if self.kind == "died":
            return died, len(trace) - 1
        if self.kind == "survived":
            return not died, 0
        for t, _a, px, y, pose, x_ram in trace:
            if self.kind == "pose" and pose in self.poses:
                return True, t
            if self.kind == "floor" and y == self.y and pose in yeti.SURFACE_POSES:
                return True, t
            if self.kind == "reach" and pose not in yeti.NON_TRAVERSAL_POSES:
                if reaches(self.target.pos, x_ram, y, 2, mode="sprite"):
                    return True, t
        return False, 0


def sources(args, cfg, iface):
    """Yield (label, state_bytes) to start episodes from."""
    spec = args.__dict__["from"]
    if spec == "reset":
        yield "reset", Path(cfg.curriculum.start_state).read_bytes()
        return
    kind, _, val = spec.partition(":")
    if kind == "state":
        yield Path(val).stem, Path(val).read_bytes()
        return
    if kind != "pool":
        raise SystemExit("--from must be reset | pool:WAYPOINT | state:PATH")
    if not args.run:
        raise SystemExit("--from pool:... needs --run")
    pool = pickle.load(open(Path(args.run) / "checkpoints.pkl", "rb"))
    if val not in pool["waypoints"]:
        raise SystemExit(
            f"no pool {val!r}; have: {', '.join(sorted(pool['waypoints']))}"
        )
    entries = pool["waypoints"][val][0]
    if not entries:
        raise SystemExit(f"pool {val!r} is empty")
    idx = (
        [args.seed_index]
        if args.seed_index is not None
        else range(min(len(entries), args.episodes))
    )
    for i in idx:
        yield f"{val}_seed{i}", bytes(entries[i][2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--from", default="reset", help="reset | pool:WAYPOINT | state:PATH"
    )
    ap.add_argument("--run", help="run dir holding checkpoints.pkl, for --from pool:")
    ap.add_argument(
        "--seed-index", type=int, help="a single pool seed, not the first N"
    )
    ap.add_argument("--model", help="policy actor; omit to use --plan")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--plan", help='scripted actor, e.g. "NOOP:18,JUMP_LEFT:40"')
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument(
        "--want", help="pose=N[,N] | reach=NAME | floor=Y | died | survived"
    )
    ap.add_argument("--keep", type=int, default=5, help="max clips written")
    ap.add_argument("--pre", type=int, default=30, help="frames kept before the match")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="output/monitor/record")
    args = ap.parse_args(argv)

    if not args.model and not args.plan:
        raise SystemExit("need --model or --plan")

    cfg = RunConfig.from_yaml(args.config)
    env_cfg = dataclasses.replace(cfg.env, max_steps=args.max_steps + 50)
    env = build_training_env(env_cfg.profile, env_cfg)
    iface = env.base._interface
    env.gym.reset()
    level = int(cfg.reward.params.get("level", 4))

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
    plan = parse_plan(args.plan) if args.plan else None
    want = Want(args.want, level)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = yeti.POSE_NAMES

    starts = list(sources(args, cfg, iface))
    print(f"source: {args.__dict__['from']}  ({len(starts)} start state(s))")
    print(f"actor : {'policy ' + args.model if model else 'plan ' + args.plan}")
    print(f"want  : {args.want or 'anything'}   keep up to {args.keep}\n")

    written = matched = 0
    for ep in range(args.episodes):
        label, state = starts[ep % len(starts)]
        env.gym.reset()
        iface.load_state(state)
        env.preprocessed.notify_state_loaded()
        obs, _, _, _, _ = env.gym.step(VOCAB["NOOP"])
        trace, raw, died = [], [], False
        for t in range(args.max_steps):
            if model is not None:
                a, _ = model.predict(obs, deterministic=args.deterministic)
                act = [int(v) for v in (a if hasattr(a, "__iter__") else [a])]
            else:
                if t >= len(plan):
                    break
                act = plan[t]
            obs, _, te, tr, _ = env.gym.step(act)
            x, y = yeti.read_pos(iface)
            trace.append((t, act, x * 4 + 8, y, yeti.read_pose(iface), x))
            raw.append(np.asarray(env.base._last_raw_obs, np.uint8).copy())
            died = yeti.is_dead(iface)
            if died or te or tr:
                break
        ok, at = want.match(trace, died)
        if not ok:
            continue
        matched += 1
        if written >= args.keep:
            continue
        lo = max(0, at - args.pre)
        frames = [
            stamp(
                raw[i],
                trace[i][2],
                trace[i][3],
                trace[i][4],
                names.get(trace[i][4], "UNCATALOGUED"),
            )
            for i in range(lo, len(raw))
        ]
        tag = "_" + args.want.replace("=", "") if args.want else ""
        p = out / f"{label}_ep{ep}{tag}.mp4"
        try:
            import imageio.v2 as imageio

            imageio.mimsave(p, frames, fps=args.fps)
            written += 1
            print(
                f"  ep {ep:>4} {label}: matched at step {at}, {len(frames)} frames"
                f" -> {p}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - encoder is optional
            print(f"  ep {ep:>4}: matched but no video ({exc})")
    print(f"\n{matched}/{args.episodes} episodes matched; {written} clip(s) in {out}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
