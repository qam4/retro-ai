#!/usr/bin/env python3
"""Short training run that CHECKS THE CHAIN, not just that nothing crashed.

Motivation: a 40k smoke passed before L3 v14, then the full 15M run had already
destroyed the from-reset chain by its first route table (Lsc1_top 0.99 -> 0.10)
and we spent 6h finding out. The smoke was not too short — it only looked for
exceptions. The chain was visible at 90k in the control run (SN3 still 0.75), so
a couple of extra minutes plus the right assertion is enough.

What it checks: how many route points are reliably reached FROM RESET (the
``route[N]: k/N reached>=0.5 from reset`` scalar, which the manager already
prints). A warm-started run must not lose the chain it inherited.

Usage
  # run a short train and assert the chain survives
  smoke_train.py --config <cfg> --timesteps 100000 --min-chain 6

  # or just check a log a run already produced (no emulator needed)
  smoke_train.py --check-log <run>/output.log --min-chain 6

Exit code is 0 on pass, 1 on regression — so it can gate a long run.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

CHAIN_RE = re.compile(r"route\[(\d+)\]:\s*(\d+)/(\d+)\s+reached>=0\.5")
RESET_REACH_RE = re.compile(r"reset_reach=\[([^\]]*)\]")


def parse_chain(text: str):
    """``(reached, total)`` from the LAST route scalar in ``text``, else None."""
    found = CHAIN_RE.findall(text)
    if not found:
        return None
    _n, reached, total = found[-1]
    return int(reached), int(total)


def parse_reset_reach(text: str):
    """The LAST reset_reach array as floats, else None."""
    found = RESET_REACH_RE.findall(text)
    if not found:
        return None
    return [float(v) for v in found[-1].split(",")]


def deepest_rung(reach, threshold: float = 0.5) -> int:
    """Deepest rung reliably reached from reset (contiguous from the start).

    Contiguity matters: an isolated high value further up would be noise, while
    the chain question is "how far does the route hold TOGETHER".
    """
    depth = 0
    for i, v in enumerate(reach):
        if i == 0:
            continue  # rung 0 is the reset itself, always 1.0
        if v >= threshold:
            depth = i
        else:
            break
    return depth


def report(text: str, min_chain: int, min_depth: int | None) -> int:
    """Print the chain verdict. ``min_depth=None`` leaves rungs informational.

    Route points are the portable signal: they are named map points, stable
    across runs. Rung INDICES are not — the progress ladder re-keyed from fruit
    count to mandatory-target count, so L3 went from 3 rungs to 15 and the same
    depth number means different things on either side of that change. Only
    assert on rungs when the caller knows both logs share a ladder.
    """
    chain = parse_chain(text)
    reach = parse_reset_reach(text)
    ok = True
    if chain is None:
        print("  ! no route scalar found — did the run get far enough?")
        ok = False
    else:
        reached, total = chain
        flag = "OK " if reached >= min_chain else "FAIL"
        print(
            f"  [{flag}] route points reached>=0.5 from reset: {reached}/{total} "
            f"(need >= {min_chain})"
        )
        ok = ok and reached >= min_chain
    if reach is None:
        print("  ! no reset_reach array found")
    else:
        d = deepest_rung(reach)
        if min_depth is None:
            print(f"  [   ] deepest contiguous rung from reset: {d} (informational)")
        else:
            flag = "OK " if d >= min_depth else "FAIL"
            print(
                f"  [{flag}] deepest contiguous rung from reset: {d} "
                f"(need >= {min_depth})"
            )
            ok = ok and d >= min_depth
        print(f"        reset_reach={['%.2f' % v for v in reach]}")
    print(
        "\nPASS: the chain survived."
        if ok
        else "\nFAIL: the from-reset chain regressed — do NOT start a long run."
    )
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="training config to smoke")
    ap.add_argument("--check-log", help="check an existing output.log instead")
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument(
        "--min-chain",
        type=int,
        default=1,
        help="route points that must be reached >=0.5 from reset",
    )
    ap.add_argument(
        "--min-depth",
        type=int,
        default=None,
        help="deepest contiguous progress rung required; omit to "
        "report rungs without asserting (indices are not "
        "comparable across ladder changes)",
    )
    args = ap.parse_args()

    if args.check_log:
        with open(args.check_log, errors="replace") as fh:
            return report(fh.read(), args.min_chain, args.min_depth)

    if not args.config:
        ap.error("one of --config or --check-log is required")

    # Short run into a scratch dir, with the route table forced to print inside
    # the budget so the scalar is available.
    out = tempfile.mkdtemp(prefix="smoke_train_")
    import yaml

    cfg = yaml.safe_load(open(args.config))
    cfg["training"]["timesteps"] = args.timesteps
    cfg["training"]["output"] = out
    cfg["training"]["snapshot_freq_steps"] = max(args.timesteps, 1_000_000)
    tmp_cfg = os.path.join(out, "smoke.yaml")
    os.makedirs(out, exist_ok=True)
    with open(tmp_cfg, "w") as fh:
        yaml.safe_dump(cfg, fh)

    print(f"smoke: {args.config} for {args.timesteps} steps -> {out}", flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "python:build/ci-linux")
    # STREAM the child's output while keeping a copy to parse. capture_output
    # buffers everything until exit, which makes a multi-minute smoke look hung
    # to whatever is watching the log — and being watched is the point of this
    # script.
    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/mo5/yeti/train_checkpoint_curriculum.py",
            "--config",
            tmp_cfg,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    if rc != 0:
        print("\nFAIL: the run itself errored.")
        return 1
    return report("".join(lines), args.min_chain, args.min_depth)


if __name__ == "__main__":
    raise SystemExit(main())
