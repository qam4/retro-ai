#!/usr/bin/env python3
"""Per-optimization comparison benchmark.

Runs bench_training.py with a specific optimization lever and compares
the result against the stored baseline in benchmarks/baseline.json.

Usage:
    python scripts/bench_compare.py --optimization vecenv-4
    python scripts/bench_compare.py --optimization ram-obs --timesteps 5000
    python scripts/bench_compare.py --optimization frameskip-8 --output benchmarks/compare_fs8.json

Supported optimizations:
    vecenv-2, vecenv-4, vecenv-8
    frameskip-1, frameskip-2, frameskip-8
    ram-obs
    lowres-42, highres-160
"""

import argparse
import json
import os
import subprocess
import sys
import time


# Maps optimization name → bench_training.py overrides
OPTIMIZATION_CONFIGS = {
    "vecenv-2":    {"num_envs": 2},
    "vecenv-4":    {"num_envs": 4},
    "vecenv-8":    {"num_envs": 8},
    "frameskip-1": {"frame_skip": 1},
    "frameskip-2": {"frame_skip": 2},
    "frameskip-8": {"frame_skip": 8},
    "ram-obs":     {"observation_mode": "ram"},
    "lowres-42":   {"resize": [42, 42]},
    "highres-160": {"resize": [160, 240]},
}


def compute_comparison(
    baseline_fps: float,
    optimized_fps: float,
    optimization_name: str,
) -> dict:
    """Compute comparison metrics between baseline and optimized fps."""
    absolute_delta = optimized_fps - baseline_fps
    if baseline_fps > 0:
        percentage_improvement = absolute_delta / baseline_fps * 100.0
    else:
        percentage_improvement = 0.0

    result = {
        "optimization_name": optimization_name,
        "baseline_fps": round(baseline_fps, 1),
        "optimized_fps": round(optimized_fps, 1),
        "absolute_delta": round(absolute_delta, 1),
        "percentage_improvement": round(percentage_improvement, 1),
    }
    if optimized_fps < baseline_fps:
        result["warning"] = "REGRESSION: optimized fps is lower than baseline"
    return result


def load_baseline(path: str = "benchmarks/baseline.json") -> dict:
    """Load baseline benchmark results."""
    if not os.path.exists(path):
        print(
            f"Error: baseline not found at {path}\n"
            "Run: python scripts/capture_baseline.py",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def run_benchmark(
    game_profile: str,
    timesteps: int,
    overrides: dict,
) -> dict:
    """Run bench_training.py with the given overrides and return parsed JSON."""
    baseline_defaults = {
        "num_envs": 1,
        "frame_skip": 4,
        "observation_mode": "framebuffer",
        "resize": [84, 84],
    }
    config = {**baseline_defaults, **overrides}

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
        print(f"Benchmark failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    stdout = result.stdout
    json_marker = "--- JSON ---"
    idx = stdout.find(json_marker)
    if idx < 0:
        print(f"Could not find JSON in benchmark output:\n{stdout}", file=sys.stderr)
        sys.exit(1)

    return json.loads(stdout[idx + len(json_marker):].strip())


def main():
    parser = argparse.ArgumentParser(
        description="Compare a specific optimization against baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--optimization", required=True,
        choices=list(OPTIMIZATION_CONFIGS.keys()),
        help="Optimization lever to benchmark",
    )
    parser.add_argument(
        "--game-profile", default="course_automobile",
        help="Game profile name (default: course_automobile)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000,
        help="Timesteps for the benchmark (default: 10000)",
    )
    parser.add_argument(
        "--baseline", default="benchmarks/baseline.json",
        help="Path to baseline JSON",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save comparison JSON (default: benchmarks/compare_<name>.json)",
    )
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    baseline_fps = baseline.get("fps", 0.0)
    overrides = OPTIMIZATION_CONFIGS[args.optimization]

    print(
        f"Comparing optimization '{args.optimization}' against baseline "
        f"({baseline_fps} fps)...\n",
        file=sys.stderr,
    )
    print(f"Overrides: {overrides}", file=sys.stderr)

    bench_result = run_benchmark(args.game_profile, args.timesteps, overrides)
    optimized_fps = bench_result.get("fps", 0.0)

    comparison = compute_comparison(baseline_fps, optimized_fps, args.optimization)
    comparison["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    comparison["benchmark_details"] = bench_result

    # Print human-readable summary
    print(f"\n=== Comparison: {args.optimization} ===")
    print(f"  Baseline:    {comparison['baseline_fps']} fps")
    print(f"  Optimized:   {comparison['optimized_fps']} fps")
    print(f"  Delta:       {comparison['absolute_delta']:+.1f} fps "
          f"({comparison['percentage_improvement']:+.1f}%)")
    if "warning" in comparison:
        print(f"  ⚠ {comparison['warning']}")

    # Save JSON
    output_path = args.output or f"benchmarks/compare_{args.optimization}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
