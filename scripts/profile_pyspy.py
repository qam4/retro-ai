#!/usr/bin/env python3
"""py-spy profiling helper for retro-ai training sessions.

Usage:
    python scripts/profile_pyspy.py --mode record --game-profile game_profiles/course_automobile_training.yaml
    python scripts/profile_pyspy.py --mode top --game-profile game_profiles/course_automobile_training.yaml
    python scripts/profile_pyspy.py --mode record --timesteps 10000 --rate 200 --output-dir output/profiling

Modes:
    record  — Produce a flame graph SVG (non-interactive, saves to file)
    top     — Live top-like view of Python function hotspots (interactive)

Requirements:
    - py-spy must be installed: pip install py-spy
    - May require root/sudo for process attachment on some systems
"""

import argparse
import os
import shutil
import subprocess
import sys


def build_pyspy_command(
    mode: str,
    output_dir: str,
    game_profile: str,
    timesteps: int,
    rate: int,
) -> list:
    """Build the py-spy command line for the given mode.

    Parameters
    ----------
    mode : str
        "record" for flame graph SVG, "top" for live view.
    output_dir : str
        Directory for output files (record mode).
    game_profile : str
        Path to the training config YAML.
    timesteps : int
        Number of training timesteps to run.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    list of str
        Command suitable for subprocess.run().

    Raises
    ------
    ValueError
        If mode is not "record" or "top".
    """
    if mode not in ("record", "top"):
        raise ValueError(f"Unknown mode: {mode!r}. Use 'record' or 'top'.")

    train_cmd = [
        sys.executable, "-m", "retro_ai.training.cli", "train",
        game_profile,
    ]

    if mode == "record":
        os.makedirs(output_dir, exist_ok=True)
        svg_path = os.path.join(output_dir, "flamegraph.svg")
        return [
            "py-spy", "record",
            "-o", svg_path,
            "--rate", str(rate),
            "--subprocesses",
            "--",
        ] + train_cmd
    else:  # top
        return [
            "py-spy", "top",
            "--rate", str(rate),
            "--",
        ] + train_cmd


def main():
    parser = argparse.ArgumentParser(
        description="Profile retro-ai training with py-spy",
    )
    parser.add_argument(
        "--mode", choices=["record", "top"], default="record",
        help="Profiling mode: 'record' for flame graph, 'top' for live view",
    )
    parser.add_argument(
        "--game-profile",
        default="game_profiles/course_automobile_training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--rate", type=int, default=100, help="Sampling rate in Hz")
    parser.add_argument("--output-dir", default="output/profiling")
    args = parser.parse_args()

    # Check py-spy is available
    if not shutil.which("py-spy"):
        print("Error: py-spy not found. Install with: pip install py-spy", file=sys.stderr)
        sys.exit(1)

    cmd = build_pyspy_command(
        mode=args.mode,
        output_dir=args.output_dir,
        game_profile=args.game_profile,
        timesteps=args.timesteps,
        rate=args.rate,
    )

    print(f"Running: {' '.join(cmd)}", flush=True)

    if args.mode == "top":
        # Interactive — run directly
        os.execvp(cmd[0], cmd)
    else:
        # Record mode — run and wait
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode == 0:
            svg_path = os.path.join(args.output_dir, "flamegraph.svg")
            print(f"Flame graph saved to: {svg_path}")
        else:
            print(f"py-spy exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
