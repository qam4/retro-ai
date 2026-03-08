#!/usr/bin/env python3
"""Capture a baseline performance benchmark for retro-ai training.

Runs bench_training.py with default parameters and saves the result
along with environment metadata to benchmarks/baseline.json.

Usage:
    python scripts/capture_baseline.py
    python scripts/capture_baseline.py --output benchmarks/baseline.json --timesteps 10000
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time


def collect_environment_info() -> dict:
    """Collect hardware/software environment for reproducibility."""
    ram_gb = 0.0
    try:
        ram_gb = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1
        )
    except (ValueError, OSError):
        pass

    compiler = "unknown"
    try:
        result = subprocess.run(
            ["cc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            compiler = result.stdout.split("\n")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    pytorch_version = "not installed"
    try:
        import torch
        pytorch_version = torch.__version__
    except ImportError:
        pass

    return {
        "cpu_model": platform.processor() or platform.machine(),
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "pytorch_version": pytorch_version,
        "compiler": compiler,
        "build_flags": "Release (no PGO, no profiling)",
    }


def capture_baseline(
    output_path: str = "benchmarks/baseline.json",
    game_profile: str = "course_automobile",
    timesteps: int = 10000,
) -> dict:
    """Run bench_training.py with defaults and save baseline + environment info."""
    # Run the benchmark and capture JSON output
    cmd = [
        sys.executable, "scripts/bench_training.py",
        "--game-profile", game_profile,
        "--timesteps", str(timesteps),
        "--num-envs", "1",
        "--frame-skip", "4",
        "--observation-mode", "framebuffer",
        "--resize", "84", "84",
    ]

    print(f"Running baseline benchmark ({timesteps} timesteps)...", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(f"Benchmark failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON from stdout — bench_training.py prints human summary then
    # "--- JSON ---" followed by indented JSON
    stdout = result.stdout
    json_marker = "--- JSON ---"
    idx = stdout.find(json_marker)
    if idx < 0:
        print(f"Could not find JSON marker in benchmark output:\n{stdout}", file=sys.stderr)
        sys.exit(1)

    json_text = stdout[idx + len(json_marker):].strip()
    benchmark = json.loads(json_text)
    benchmark["environment"] = collect_environment_info()
    benchmark["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Baseline saved to: {output_path}")
    print(f"  FPS: {benchmark.get('fps', 'N/A')}")
    print(f"  Wall clock: {benchmark.get('wall_clock_seconds', 'N/A')}s")
    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Capture baseline performance benchmark")
    parser.add_argument("--output", default="benchmarks/baseline.json")
    parser.add_argument(
        "--game-profile",
        default="course_automobile",
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()

    capture_baseline(args.output, args.game_profile, args.timesteps)


if __name__ == "__main__":
    main()
