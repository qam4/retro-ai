"""Annotated screenshots. Replaces 8 near-identical frame-annotation scripts.

WHY THIS EXISTS
---------------
Eight scripts in debug/ rendered a game frame with things drawn on it. They differed in
WHICH state to reach and WHAT to draw, and each retyped the state loading, the scaling
and the coordinate conversion. The conversion is the part that kept going wrong, so it
now lives in exactly one place -- see COORDINATES below.

What it is NOT for: charts. Eight other debug scripts produced matplotlib figures, and
those are not renderings, they are the ANSWER to a specific measurement (a reward curve
on a pad, a swing period, a landing scatter). A chart belongs with the code that
measured it. This file only draws on real frames.

COORDINATES, the bit that bit repeatedly
----------------------------------------
Two different conventions are in play and mixing them silently shifts everything:

  anchors/markers  a waypoint at (x_ram, y) is drawn at pixel (x_ram * 4 + 8, y + 8) --
                   the convention scripts/mo5/yeti/draw_level_map.py established. The
                   +8 in y is because the anchor names the sprite's TOP, and the marker
                   should sit on the sprite's middle.
  boxes            drawn in TRUE frame coordinates, NOT shifted. A waypoint box is the
                   set of rows the sprite's TOP may occupy, so shifting it by +8 would
                   describe a region the test does not use.

Everything is multiplied by SCALE last, after the frame is scaled, never before.

Examples
--------
    # one frame from a pool seed, with every L4 anchor marked
    figure.py --from pool:Spring --run <run> --annotate anchors --out f.png

    # where 200 Low2_launch seeds actually put the agent's head
    figure.py --from pool:Low2_launch --run <run> --seeds 200 \
        --annotate anchors,heads --out seeds.png

    # the px 184 vs px 188 comparison, side by side
    figure.py --compare pool:Low2_launch,pool:Low2 --run <run> --annotate anchors
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling diag modules

from record import VOCAB, parse_plan  # noqa: E402
from retro_ai.games import yeti  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import RunConfig  # noqa: E402
from retro_ai.training.targets import build_targets  # noqa: E402
from retro_ai.training.yeti_map import get_level_map  # noqa: E402

SCALE = 3
SPRITE_W, SPRITE_H = 14, 18
DEFAULT_CONFIG = (
    "experiments/003-yeti/configs/yeti_curriculum_l4_v16c_payonchange_cold_6m.yaml"
)
KINDS = ("anchors", "boxes", "floors", "heads")


def anchor_px(x_ram, y):
    """Marker centre for an anchor at (x_ram, y). See COORDINATES in the docstring."""
    return int(x_ram) * 4 + 8, int(y) + 8


def draw(img, level, targets, heads, kinds, caption):
    """Scale ``img`` and draw the requested overlays. ``heads`` = [(x_ram, y), ...]."""
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + 34), (0, 0, 0))
    canvas.paste(img, (0, 0))  # caption BELOW, never folded into the resize
    img = canvas
    d = ImageDraw.Draw(img)
    s = SCALE

    if "floors" in kinds:
        for p in get_level_map(level).platforms:
            d.line(
                [p.x_min * s, p.y * s, p.x_max * s, p.y * s],
                fill=(60, 60, 120),
                width=1,
            )
            d.text((p.x_min * s + 2, p.y * s - 10), f"f{p.floor}", fill=(90, 90, 160))

    if "boxes" in kinds:
        # TRUE coordinates: the rows the sprite TOP may occupy. Not +8 shifted.
        for t in targets:
            ax, ay = int(t.pos[0]) * 4 + 8, int(t.pos[1])
            d.rectangle(
                [
                    (ax - SPRITE_W // 2) * s,
                    ay * s,
                    (ax + SPRITE_W // 2 - 1) * s,
                    (ay + SPRITE_H - 1) * s,
                ],
                outline=(120, 60, 60),
            )

    # ANCHORS BEFORE HEADS. Drawn the other way round, a single yellow anchor hides the
    # cyan heads underneath it -- which is the one thing a seed figure exists to show.
    if "anchors" in kinds:
        for i, t in enumerate(targets):
            ax, ay = anchor_px(t.pos[0], t.pos[1])
            r = 3
            d.ellipse(
                [ax * s - r, ay * s - r, ax * s + r, ay * s + r],
                fill=(255, 220, 0),
                outline=(0, 0, 0),
            )
            # Stagger labels: L4 has 32 targets and several sit within a few px of each
            # other (Fr1/Fr1_launch, Hi4/Hi3_launch), so a fixed offset overprints them.
            dy = -5 if i % 2 == 0 else 4
            d.text((ax * s + 5, ay * s + dy), t.id, fill=(255, 220, 0))

    if "heads" in kinds:
        # Seeds coincide heavily -- 100 Low2_launch seeds are all at px 184 -- so a
        # plain scatter renders as ONE dot and reads as "no data". Tally instead, and
        # print the count when a position repeats.
        tally: dict = {}
        for x_ram, y in heads:
            tally[(int(x_ram), int(y))] = tally.get((int(x_ram), int(y)), 0) + 1
        for (x_ram, y), n in sorted(tally.items(), key=lambda kv: -kv[1]):
            hx, hy = anchor_px(x_ram, y)
            r = 2 + min(5, n // 10)
            d.ellipse(
                [hx * s - r, hy * s - r, hx * s + r, hy * s + r],
                fill=(0, 220, 220),
            )
            if n > 1:
                d.text((hx * s + r + 1, hy * s - 4), str(n), fill=(0, 220, 220))

    d.text((5, img.height - 30), caption, fill=(255, 255, 255))
    legend = ["yellow=anchor"]
    if "heads" in kinds:
        legend.append("cyan=agent head (number = seeds at that pixel)")
    if "boxes" in kinds:
        legend.append("red=sprite box")
    if "floors" in kinds:
        legend.append("blue=floor line (spans gaps: it is a logical extent)")
    d.text((5, img.height - 16), "  ".join(legend), fill=(150, 150, 150))
    return img


def load_source(spec, args, cfg, iface, env):
    """Reach a state. Returns (label, [(x_ram, y) heads to plot])."""
    kind, _, val = spec.partition(":")
    heads = []
    if kind == "reset" or spec == "reset":
        iface.load_state(Path(cfg.curriculum.start_state).read_bytes())
        label = "reset"
    elif kind == "state":
        iface.load_state(Path(val).read_bytes())
        label = Path(val).stem
    elif kind == "pool":
        if not args.run:
            raise SystemExit("pool:... needs --run")
        pool = pickle.load(open(Path(args.run) / "checkpoints.pkl", "rb"))
        if val not in pool["waypoints"]:
            raise SystemExit(
                f"no pool {val!r}; have: {', '.join(sorted(pool['waypoints']))}"
            )
        entries = pool["waypoints"][val][0]
        if not entries:
            raise SystemExit(f"pool {val!r} is empty")
        # READ POSITIONS AS SAVED -- do not step the env first. Measured: 15 of 1251
        # seeds report a different position after even one step.
        for e in entries[: args.seeds]:
            iface.load_state(bytes(e[2]))
            heads.append(yeti.read_pos(iface))
        iface.load_state(bytes(entries[args.seed_index or 0][2]))
        label = f"{val} (n={len(heads)} seeds)" if args.seeds > 1 else val
    else:
        raise SystemExit("--from must be reset | pool:WAYPOINT | state:PATH")

    env.preprocessed.notify_state_loaded()
    env.gym.step(VOCAB["NOOP"])
    if args.advance:
        plan = parse_plan(args.plan) if args.plan else [VOCAB["NOOP"]] * args.advance
        for i in range(args.advance):
            env.gym.step(plan[min(i, len(plan) - 1)])
    return label, heads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--from", default="reset", help="reset | pool:WAYPOINT | state:PATH"
    )
    ap.add_argument(
        "--compare", help="two sources, comma separated, drawn side by side"
    )
    ap.add_argument("--run", help="run dir holding checkpoints.pkl, for pool:")
    ap.add_argument(
        "--seed-index", type=int, help="which seed to render the frame from"
    )
    ap.add_argument("--seeds", type=int, default=1, help="how many seed heads to plot")
    ap.add_argument(
        "--advance", type=int, default=0, help="steps to run before shooting"
    )
    ap.add_argument("--plan", help='actions while advancing, e.g. "RIGHT:10"')
    ap.add_argument("--annotate", default="anchors", help=f"comma list of {KINDS}")
    ap.add_argument("--out", default="output/monitor/figure/frame.png")
    args = ap.parse_args(argv)

    kinds = [k.strip() for k in args.annotate.split(",") if k.strip()]
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise SystemExit(f"unknown --annotate {bad}; known: {list(KINDS)}")

    cfg = RunConfig.from_yaml(args.config)
    env_cfg = dataclasses.replace(cfg.env, max_steps=10_000)
    env = build_training_env(env_cfg.profile, env_cfg)
    iface = env.base._interface
    env.gym.reset()
    level = int(cfg.reward.params.get("level", 4))
    targets = build_targets(level)

    specs = (
        [s.strip() for s in args.compare.split(",")]
        if args.compare
        else [args.__dict__["from"]]
    )
    panels = []
    for spec in specs:
        label, heads = load_source(spec, args, cfg, iface, env)
        raw = np.asarray(env.base._last_raw_obs, np.uint8).copy()
        x, y = yeti.read_pos(iface)
        cap = f"{label}   agent px {x * 4 + 8} y {y} pose {yeti.read_pose(iface)}"
        panels.append(
            draw(Image.fromarray(raw).convert("RGB"), level, targets, heads, kinds, cap)
        )
        print(f"  {spec}: px {x * 4 + 8} y {y}, {len(heads)} head(s) plotted")

    if len(panels) == 1:
        img = panels[0]
    else:
        w = sum(p.width for p in panels) + 8 * (len(panels) - 1)
        img = Image.new("RGB", (w, max(p.height for p in panels)), (0, 0, 0))
        xo = 0
        for p in panels:
            img.paste(p, (xo, 0))
            xo += p.width + 8

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"figure: {out}  ({img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
