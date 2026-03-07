#!/usr/bin/env python3
"""Per-component profiling of emulator frame timings.

Runs N frames headless and reports a per-component timing breakdown
(CPU, VDC, framebuffer extraction, reward computation) using the
C++ instrumentation exposed via get_last_frame_timings().

Requires the C++ library to be built with -DRETRO_AI_PROFILING=ON
for non-zero timings.

Usage examples:

  # Videopac (requires --bios):
  PYTHONPATH=build/ci-linux:python python scripts/bench_components.py \
      --emulator videopac \
      --bios roms/videopac/bios.bin \
      --rom  roms/videopac/game.bin \
      --frames 500

  # MO5 (no bios needed):
  PYTHONPATH=build/ci-linux:python python scripts/bench_components.py \
      --emulator mo5 \
      --rom roms/mo5/game.k7 \
      --frames 500
"""

import argparse
import json
import sys
import time

import retro_ai_native


def create_interface(emulator: str, rom: str, bios: str | None):
    """Instantiate the correct RLInterface for the given emulator."""
    emu = emulator.lower()
    if emu == "videopac":
        if bios is None:
            sys.exit("Error: --bios is required for the videopac emulator")
        return retro_ai_native.VideopacRLInterface(
            bios, rom, "survival", 0, reward_params={},
        )
    if emu == "mo5":
        return retro_ai_native.MO5RLInterface(
            rom, "survival", reward_params={},
        )
    sys.exit(f"Error: unknown emulator {emulator!r}. Supported: videopac, mo5")


def format_timings(avg_timings: dict) -> str:
    """Return a human-readable summary of per-component timings.

    Parameters
    ----------
    avg_timings : dict
        Keys: cpu_us, vdc_us, framebuffer_us, reward_us, total_us
        (average microseconds per frame).

    Returns
    -------
    str
        Multi-line formatted string with percentages and absolute values.
    """
    total = avg_timings["total_us"]
    components = [
        ("CPU emulation", avg_timings["cpu_us"]),
        ("VDC rendering", avg_timings["vdc_us"]),
        ("Framebuffer extraction", avg_timings["framebuffer_us"]),
        ("Reward computation", avg_timings["reward_us"]),
    ]

    lines = []
    if total <= 0:
        lines.append("WARNING: All timings are zero. Build with -DRETRO_AI_PROFILING=ON to enable instrumentation.")
        lines.append("")
        for label, _ in components:
            lines.append(f"  {label:30s}    0.0 µs  ( 0.0%)")
        lines.append(f"  {'Total':30s}    0.0 µs")
        return "\n".join(lines)

    for label, us in components:
        pct = us / total * 100.0
        lines.append(f"  {label:30s}  {us:8.1f} µs  ({pct:5.1f}%)")
    lines.append(f"  {'Total':30s}  {total:8.1f} µs")
    return "\n".join(lines)


def run_profiling(interface, num_frames: int) -> dict:
    """Step the emulator num_frames times and collect timing data.

    Returns a dict with per-component averages (microseconds) and
    the raw per-frame timing lists.
    """
    cpu_list = []
    vdc_list = []
    fb_list = []
    reward_list = []
    total_list = []

    wall_start = time.perf_counter()
    for _ in range(num_frames):
        interface.step_numpy([0])
        t = interface.get_last_frame_timings()
        cpu_list.append(t.cpu_us)
        vdc_list.append(t.vdc_us)
        fb_list.append(t.framebuffer_us)
        reward_list.append(t.reward_us)
        total_list.append(t.total_us)
    wall_elapsed = time.perf_counter() - wall_start

    n = num_frames
    return {
        "num_frames": n,
        "wall_seconds": wall_elapsed,
        "wall_fps": n / wall_elapsed if wall_elapsed > 0 else 0.0,
        "avg": {
            "cpu_us": sum(cpu_list) / n,
            "vdc_us": sum(vdc_list) / n,
            "framebuffer_us": sum(fb_list) / n,
            "reward_us": sum(reward_list) / n,
            "total_us": sum(total_list) / n,
        },
    }


def build_json_output(emulator: str, rom: str, results: dict) -> dict:
    """Build a machine-readable JSON dict from profiling results."""
    avg = results["avg"]
    total = avg["total_us"]
    return {
        "emulator": emulator,
        "rom": rom,
        "num_frames": results["num_frames"],
        "wall_seconds": round(results["wall_seconds"], 4),
        "wall_fps": round(results["wall_fps"], 1),
        "avg_timings_us": {k: round(v, 2) for k, v in avg.items()},
        "percentages": {
            "cpu": round(avg["cpu_us"] / total * 100, 2) if total > 0 else 0.0,
            "vdc": round(avg["vdc_us"] / total * 100, 2) if total > 0 else 0.0,
            "framebuffer": round(avg["framebuffer_us"] / total * 100, 2) if total > 0 else 0.0,
            "reward": round(avg["reward_us"] / total * 100, 2) if total > 0 else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile per-component emulator frame timings (headless).",
    )
    parser.add_argument("--emulator", required=True, help="Emulator type: videopac, mo5")
    parser.add_argument("--rom", required=True, help="Path to ROM file")
    parser.add_argument("--bios", default=None, help="Path to BIOS file (required for videopac)")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to profile (default: 500)")
    args = parser.parse_args()

    interface = create_interface(args.emulator, args.rom, args.bios)
    interface.reset_numpy(-1)

    # Warmup
    warmup = min(50, args.frames // 5)
    for _ in range(warmup):
        interface.step_numpy([0])

    print(f"Profiling {args.frames} frames on {args.emulator} ({args.rom})...\n")
    results = run_profiling(interface, args.frames)

    # Human-readable output
    print(f"Wall time: {results['wall_seconds']:.3f}s  ({results['wall_fps']:.0f} fps)\n")
    print("Per-component average (per frame):")
    print(format_timings(results["avg"]))

    # JSON output
    json_data = build_json_output(args.emulator, args.rom, results)
    print(f"\n--- JSON ---\n{json.dumps(json_data, indent=2)}")


if __name__ == "__main__":
    main()
