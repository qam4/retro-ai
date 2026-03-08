#!/usr/bin/env python3
"""Performance regression detection for retro-ai training.

Runs a fixed set of benchmark configurations and compares against stored
reference values. Exits non-zero if any configuration regresses beyond
the tolerance threshold.

Usage:
    # Run regression suite (requires benchmarks/references.json):
    python scripts/bench_regression.py

    # Update reference values from current measurements:
    python scripts/bench_regression.py --update-references

    # Custom tolerance (default 10%):
    python scripts/bench_regression.py --tolerance 0.15

    # Specific game profile and timesteps:
    python scripts/bench_regression.py --game-profile course_automobile --timesteps 5000
"""

import argparse
import json
import os
import subprocess
import sys
import time


# Default regression configurations
REGRESSION_CONFIGS = [
    {"name": "default", "args": {}},
    {"name": "vectorized-4", "args": {"num_envs": 4}},
    {"name": "ram-observation", "args": {"observation_mode": "ram"}},
]

REFERENCES_PATH = "benchmarks/references.json"


def check_regression(
    measured_fps: float,
    reference_fps: float,
    tolerance: float = 0.10,
) -> bool:
    """Return True if measured_fps is within tolerance of reference_fps.

    A regression is detected when measured_fps < reference_fps * (1 - tolerance).
    Returns True (no regression) when reference_fps <= 0.
    """
    if reference_fps <= 0:
        return True
    return measured_fps >= reference_fps * (1.0 - tolerance)


def run_benchmark(game_profile: str, timesteps: int, overrides: dict) -> float:
    """Run bench_training.py and return the measured fps."""
    defaults = {
        "num_envs": 1,
        "frame_skip": 4,
        "observation_mode": "framebuffer",
        "resize": [84, 84],
    }
    config = {**defaults, **overrides}

    cmd = [
        sys.executable, "scripts/bench_training.py",
        "--game-profile", game_profile,
        "--timesteps", str(timesteps),
        "--num-envs", str(config["num_envs"]),
        "--frame-skip", str(config["frame_skip"]),
        "--observation-mode", config["observation_mode"],
        "--resize", str(config["resize"][0]), str(config["resize"][1]),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if result.returncode != 0:
        print(f"  Benchmark failed: {result.stderr.strip()}", file=sys.stderr)
        return 0.0

    stdout = result.stdout
    json_marker = "--- JSON ---"
    idx = stdout.find(json_marker)
    if idx < 0:
        print(f"  Could not parse benchmark output", file=sys.stderr)
        return 0.0

    data = json.loads(stdout[idx + len(json_marker):].strip())
    return data.get("fps", 0.0)


def run_regression_suite(
    references: dict,
    tolerance: float = 0.10,
    configs: list = None,
    game_profile: str = "course_automobile",
    timesteps: int = 10000,
) -> tuple:
    """Run all regression configs and return (results, all_passed)."""
    if configs is None:
        configs = REGRESSION_CONFIGS

    results = []
    all_passed = True

    for config in configs:
        name = config["name"]
        ref_fps = references.get(name, {}).get("fps", 0.0)

        print(f"  [{name}] Running benchmark...", file=sys.stderr, flush=True)
        measured_fps = run_benchmark(game_profile, timesteps, config["args"])
        passed = check_regression(measured_fps, ref_fps, tolerance)

        if not passed:
            all_passed = False

        delta_pct = 0.0
        if ref_fps > 0:
            delta_pct = round((measured_fps - ref_fps) / ref_fps * 100, 1)

        results.append({
            "name": name,
            "reference_fps": ref_fps,
            "measured_fps": round(measured_fps, 1),
            "passed": passed,
            "delta_pct": delta_pct,
        })

        status = "PASS" if passed else "FAIL"
        print(
            f"  [{name}] {status}: {measured_fps:.1f} fps "
            f"(ref: {ref_fps:.1f}, delta: {delta_pct:+.1f}%)",
            file=sys.stderr,
        )

    return results, all_passed


def load_references(path: str = REFERENCES_PATH) -> dict:
    """Load reference values from JSON file."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_references(references: dict, path: str = REFERENCES_PATH) -> None:
    """Save reference values to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(references, f, indent=2)


def update_references(
    game_profile: str,
    timesteps: int,
    configs: list = None,
    path: str = REFERENCES_PATH,
) -> dict:
    """Run all configs and save as new reference values."""
    if configs is None:
        configs = REGRESSION_CONFIGS

    references = {}
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for config in configs:
        name = config["name"]
        print(f"  [{name}] Measuring reference...", file=sys.stderr, flush=True)
        fps = run_benchmark(game_profile, timesteps, config["args"])
        references[name] = {"fps": round(fps, 1), "timestamp": timestamp}
        print(f"  [{name}] {fps:.1f} fps", file=sys.stderr)

    save_references(references, path)
    print(f"\nReferences saved to: {path}", file=sys.stderr)
    return references


def main():
    parser = argparse.ArgumentParser(
        description="Performance regression detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--update-references", action="store_true",
        help="Measure and save new reference values instead of checking",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.10,
        help="Regression tolerance (default: 0.10 = 10%%)",
    )
    parser.add_argument("--game-profile", default="course_automobile")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument(
        "--references", default=REFERENCES_PATH,
        help=f"Path to references JSON (default: {REFERENCES_PATH})",
    )
    args = parser.parse_args()

    if args.update_references:
        print("Updating reference values...", file=sys.stderr)
        refs = update_references(
            args.game_profile, args.timesteps, path=args.references,
        )
        print(json.dumps(refs, indent=2))
        sys.exit(0)

    references = load_references(args.references)
    if not references:
        print(
            f"Error: no references found at {args.references}\n"
            "Run: python scripts/bench_regression.py --update-references",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Running regression suite (tolerance: {args.tolerance*100:.0f}%)...",
        file=sys.stderr,
    )
    results, all_passed = run_regression_suite(
        references, args.tolerance,
        game_profile=args.game_profile,
        timesteps=args.timesteps,
    )

    # Output results as JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tolerance": args.tolerance,
        "all_passed": all_passed,
        "results": results,
    }
    print(json.dumps(output, indent=2))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
