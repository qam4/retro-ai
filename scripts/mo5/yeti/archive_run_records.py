#!/usr/bin/env python3
"""Copy each training run's small, text RECORD into the repo, where it is tracked.

`output/` is gitignored, and it is 150 GB of snapshots, seed pools and logs -- but the
numbers every note in experiments/ is built on live in a handful of small files per run,
which existed only on one disk. This copies those, and only those:

    run.yaml               the full resolved config the run trained with
    env.json               commit, dirty flag, emulator build hash, library versions
    run_dirty.patch        the uncommitted diff, when the run launched from a dirty tree
    best/sweep_state.json  the referee's score for EVERY snapshot
    best/best_meta.json    which snapshot the referee chose
    best/eval_status.json  the referee's last per-eval status

and every JSON under `output/mo5/yeti/champion_recheck/` (300-episode re-measures).

NOT copied, on purpose: models, seed pools, snapshots, TensorBoard logs (too big), and
anything holding emulator STATE. A save-state or a pool is a RAM image, and the MO5 runs
the game from RAM: 38 of 112 64-byte chunks of the Yeti tape image appear verbatim in
`level4_start.sav`. This repo is public and `roms/` is gitignored for that reason.

Idempotent: re-running overwrites with the current files. A run still in progress
(env.json status RUNNING) is skipped, so a half-written record is never committed.
Refuses to copy any file matching a credential pattern.

PERSONAL INFO IS SCRUBBED ON THE WAY IN, because the repo is public: env.json loses its
`hostname`, its `cwd` becomes `<repo>`, the repo path becomes `<repo>` everywhere, and
any other home directory becomes `~`. A file that still identifies the machine after
that is refused rather than copied.

    python3 scripts/mo5/yeti/archive_run_records.py            # all runs
    python3 scripts/mo5/yeti/archive_run_records.py RUN [...]  # named runs only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "output/mo5/yeti/training"
RECHECK = REPO / "output/mo5/yeti/champion_recheck"
DST = REPO / "experiments/003-yeti/data/runs"
DST_RECHECK = REPO / "experiments/003-yeti/data/champion_recheck"
FILES = [
    "run.yaml",
    "env.json",
    "run_dirty.patch",
    "best/sweep_state.json",
    "best/best_meta.json",
    "best/eval_status.json",
]
# A public repo: never copy something that looks like a credential.
SECRET = re.compile(
    r"(AKIA[0-9A-Z]{16}|aws_secret_access_key|ghp_[0-9A-Za-z]{30,}|"
    r"github_pat_[0-9A-Za-z_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?i:password|passwd|api[_-]?key|secret[_-]?key)\s*[:=]\s*\S+)"
)


LIVE_WINDOW_S = 30 * 60


def _running(run_dir: Path) -> bool:
    """Still training: status RUNNING AND written to within the last 30 minutes.

    The status alone is not enough. A run that was killed never records its end, so
    `RUNNING` sits there forever -- 8 of the first archive's runs were in that state,
    some for months, and skipping them would have lost real records.
    """
    try:
        if json.loads((run_dir / "env.json").read_text()).get("status") != "RUNNING":
            return False
    except (OSError, ValueError):
        return False
    newest = max(
        (p.stat().st_mtime for p in run_dir.iterdir() if p.is_file()), default=0.0
    )
    return time.time() - newest < LIVE_WINDOW_S


HOME = str(Path.home())
# The machine and the person must not reach a public repo. env.json records the
# hostname, the working directory and the emulator's absolute path -- all three carry
# the home directory, and the repo path carries the user's name.
PERSONAL = re.compile(
    re.escape(HOME) + r"|/home/[^/\s\"']+|\bip-\d+-\d+-\d+-\d+\b|\.ec2\.internal"
)


def _scrub(text: str) -> str:
    """Replace machine- and person-identifying strings with neutral placeholders."""
    text = text.replace(str(REPO), "<repo>").replace(HOME, "~")
    return re.sub(r"/home/[^/\s\"']+", "~", text)


def _scrub_env(text: str) -> str:
    """env.json: drop the hostname outright, relativise the paths."""
    try:
        env = json.loads(text)
    except ValueError:
        return _scrub(text)
    env.pop("hostname", None)
    if "cwd" in env:
        env["cwd"] = "<repo>"
    return _scrub(json.dumps(env, indent=1))


def _copy(src: Path, dst: Path) -> int:
    text = src.read_text(errors="replace")
    if SECRET.search(text):
        raise SystemExit(f"REFUSING {src}: matches a credential pattern")
    text = _scrub_env(text) if src.name == "env.json" else _scrub(text)
    left = PERSONAL.search(text)
    if left:
        raise SystemExit(f"REFUSING {src}: still identifies the machine ({left[0]!r})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return dst.stat().st_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="run directory names (default: all)")
    args = ap.parse_args(argv)

    runs = (
        [SRC / r for r in args.runs]
        if args.runs
        else sorted(p for p in SRC.iterdir() if p.is_dir())
    )
    n_runs = n_files = n_bytes = 0
    skipped = []
    for run in runs:
        if not run.is_dir():
            print(f"  missing: {run}", file=sys.stderr)
            continue
        if _running(run):
            skipped.append(run.name)
            continue
        got = 0
        for rel in FILES:
            src = run / rel
            if src.is_file():
                n_bytes += _copy(src, DST / run.name / rel)
                got += 1
        if got:
            n_runs += 1
            n_files += got
    if RECHECK.is_dir() and not args.runs:
        for src in sorted(RECHECK.rglob("*.json")):
            if src.name.startswith("_"):  # the referee's per-eval scratch file
                continue
            n_bytes += _copy(src, DST_RECHECK / src.relative_to(RECHECK))
            n_files += 1
    print(f"archived {n_files} files from {n_runs} runs, {n_bytes / 1e6:.1f} MB")
    if skipped:
        print(f"skipped (still running): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
