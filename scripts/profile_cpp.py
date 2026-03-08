#!/usr/bin/env python3
"""C++ profiling helper for retro-ai emulator core.

Usage:
    # gprof workflow (requires build with -DPROFILING_GPROF=ON):
    python scripts/profile_cpp.py --mode gprof --build-dir build/ci-linux --output-dir output/profiling/cpp

    # perf workflow (requires build with -DPROFILING_PERF=ON):
    python scripts/profile_cpp.py --mode perf --output-dir output/profiling/cpp

Prerequisites:
    gprof mode: cmake --build build/ci-linux --target retro_ai_native -j4 -DPROFILING_GPROF=ON
    perf mode:  cmake --build build/ci-linux --target retro_ai_native -j4 -DPROFILING_PERF=ON
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def find_native_lib(build_dir: str) -> str:
    """Find the retro_ai_native shared library in the build directory."""
    patterns = [
        os.path.join(build_dir, "retro_ai_native*.so"),
        os.path.join(build_dir, "retro_ai_native*.dylib"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"retro_ai_native not found in {build_dir}. "
        "Did you build with the correct profiling flags?"
    )


def get_workload_cmd(game_profile: str, timesteps: int) -> list:
    """Build the training workload command."""
    return [
        sys.executable, "scripts/pgo_training_workload.py",
        "--game-profile", game_profile,
        "--timesteps", str(timesteps),
    ]


def run_gprof_workflow(
    build_dir: str,
    game_profile: str,
    timesteps: int,
    output_dir: str,
) -> None:
    """Run gprof profiling: execute workload, then gprof analysis."""
    os.makedirs(output_dir, exist_ok=True)
    lib_path = find_native_lib(build_dir)

    print(f"Running workload ({timesteps} steps)...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{build_dir}:python"
    result = subprocess.run(
        get_workload_cmd(game_profile, timesteps),
        env=env,
    )
    if result.returncode != 0:
        print("Workload failed", file=sys.stderr)
        sys.exit(1)

    # gprof produces gmon.out in the current directory
    if not os.path.exists("gmon.out"):
        print(
            "Error: gmon.out not found. Was the build compiled with -DPROFILING_GPROF=ON?",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Running gprof analysis...", flush=True)
    flat_path = os.path.join(output_dir, "gprof_flat.txt")
    with open(flat_path, "w") as f:
        subprocess.run(["gprof", lib_path, "gmon.out"], stdout=f)

    # Clean up
    os.remove("gmon.out")
    print(f"gprof flat profile saved to: {flat_path}")


def run_perf_workflow(
    game_profile: str,
    timesteps: int,
    output_dir: str,
    build_dir: str = "build/ci-linux",
) -> None:
    """Run perf profiling: perf record, then generate report/flame graph."""
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{build_dir}:python"
    workload_cmd = get_workload_cmd(game_profile, timesteps)

    print(f"Running perf record ({timesteps} steps)...", flush=True)
    perf_data = os.path.join(output_dir, "perf.data")
    perf_cmd = [
        "perf", "record", "-g",
        "-o", perf_data,
        "--",
    ] + workload_cmd

    result = subprocess.run(perf_cmd, env=env)
    if result.returncode != 0:
        print("perf record failed", file=sys.stderr)
        sys.exit(1)

    # Generate text report
    report_path = os.path.join(output_dir, "perf_report.txt")
    print("Generating perf report...", flush=True)
    with open(report_path, "w") as f:
        subprocess.run(
            ["perf", "report", "-i", perf_data, "--stdio"],
            stdout=f,
        )
    print(f"perf report saved to: {report_path}")

    # Try to generate flame graph if tools are available
    stackcollapse = shutil.which("stackcollapse-perf.pl")
    flamegraph = shutil.which("flamegraph.pl")
    if stackcollapse and flamegraph:
        svg_path = os.path.join(output_dir, "flamegraph_cpp.svg")
        print("Generating flame graph...", flush=True)
        perf_script = subprocess.run(
            ["perf", "script", "-i", perf_data],
            capture_output=True, text=True,
        )
        collapsed = subprocess.run(
            [stackcollapse],
            input=perf_script.stdout,
            capture_output=True, text=True,
        )
        with open(svg_path, "w") as f:
            subprocess.run([flamegraph], input=collapsed.stdout, stdout=f, text=True)
        print(f"Flame graph saved to: {svg_path}")
    else:
        print("Note: stackcollapse-perf.pl / flamegraph.pl not found, skipping flame graph")
        print("Install from: https://github.com/brendangregg/FlameGraph")


def main():
    parser = argparse.ArgumentParser(
        description="Profile retro-ai C++ emulator core with gprof or perf",
    )
    parser.add_argument(
        "--mode", choices=["gprof", "perf"], required=True,
        help="Profiling tool to use",
    )
    parser.add_argument("--build-dir", default="build/ci-linux")
    parser.add_argument(
        "--game-profile",
        default="game_profiles/course_automobile_training.yaml",
    )
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--output-dir", default="output/profiling/cpp")
    args = parser.parse_args()

    if args.mode == "gprof":
        if not shutil.which("gprof"):
            print("Error: gprof not found", file=sys.stderr)
            sys.exit(1)
        run_gprof_workflow(args.build_dir, args.game_profile, args.timesteps, args.output_dir)
    else:
        if not shutil.which("perf"):
            print("Error: perf not found", file=sys.stderr)
            sys.exit(1)
        run_perf_workflow(args.game_profile, args.timesteps, args.output_dir, args.build_dir)


if __name__ == "__main__":
    main()
