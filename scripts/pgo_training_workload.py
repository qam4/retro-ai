#!/usr/bin/env python3
"""PGO (Profile-Guided Optimization) training workload for data collection.

Runs a representative training workload through the emulator hot path
(step, framebuffer extraction, reward computation) to generate profiling
data that the compiler can use to optimize the C++ emulator core.

PGO Workflow
============

Step 1 — Build with profiling instrumentation:

    cmake -B build/pgo -DPGO_GENERATE=ON
    cmake --build build/pgo --target retro_ai_native -j$(nproc)

Step 2 — Run this workload to generate profile data:

    PYTHONPATH=build/pgo:python python scripts/pgo_training_workload.py \
        --game-profile course_automobile

    For a custom ROM without a game profile:

    PYTHONPATH=build/pgo:python python scripts/pgo_training_workload.py \
        --emulator videopac \
        --rom roms/videopac/game.bin \
        --bios roms/videopac/bios.bin \
        --timesteps 5000

Step 3 — (Clang only) Merge raw profile data:

    llvm-profdata merge \
        -output=build/pgo/pgo-data/default.profdata \
        build/pgo/pgo-data/default.profraw

    GCC generates merged-format data automatically; skip this step for GCC.

Step 4 — Rebuild with profile data applied:

    cmake -B build/pgo -DPGO_USE=ON
    cmake --build build/pgo --target retro_ai_native -j$(nproc)

The resulting ``build/pgo/retro_ai_native*.so`` will have compiler
optimizations guided by the actual hot-path execution profile.

Usage examples:

  # Using a game profile (recommended):
  PYTHONPATH=build/pgo:python python scripts/pgo_training_workload.py \
      --game-profile course_automobile --timesteps 5000

  # Using explicit emulator/ROM args:
  PYTHONPATH=build/pgo:python python scripts/pgo_training_workload.py \
      --emulator videopac \
      --bios roms/videopac/bios.bin \
      --rom roms/videopac/game.bin \
      --timesteps 5000
"""

import argparse
import sys
import time

import retro_ai_native


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a representative training workload for PGO data collection.",
    )
    parser.add_argument(
        "--game-profile",
        default=None,
        help="Game profile name or YAML path (searches game_profiles/ directory)",
    )
    parser.add_argument(
        "--emulator",
        default=None,
        help="Emulator type (videopac, mo5). Overridden by --game-profile.",
    )
    parser.add_argument(
        "--rom",
        default=None,
        help="Path to ROM file. Overridden by --game-profile.",
    )
    parser.add_argument(
        "--bios",
        default=None,
        help="Path to BIOS file (required for videopac). Overridden by --game-profile.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="Number of emulator steps to run (default: 5000)",
    )
    return parser.parse_args()


def _create_interface_from_profile(profile):
    """Create an RLInterface from a loaded GameProfile."""
    emu = profile.emulator_type.lower()
    reward_mode = profile.reward_mode or "survival"
    reward_params = {}
    if profile.reward_params:
        # Flatten reward_params to Dict[str, str] for the C++ interface
        reward_params = {str(k): str(v) for k, v in profile.reward_params.items()}

    if emu == "videopac":
        if not profile.bios_path:
            sys.exit("Error: game profile must specify bios_path for videopac")
        return retro_ai_native.VideopacRLInterface(
            profile.bios_path,
            profile.rom_path,
            reward_mode,
            profile.joystick_index,
            reward_params=reward_params,
        )
    if emu == "mo5":
        return retro_ai_native.MO5RLInterface(
            profile.rom_path,
            reward_mode,
            reward_params=reward_params,
        )
    sys.exit(f"Error: unsupported emulator type {emu!r} in game profile")


def _create_interface_from_args(emulator, rom, bios):
    """Create an RLInterface from explicit CLI arguments."""
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


def run_workload(interface, timesteps: int) -> None:
    """Step the emulator for *timesteps* frames, exercising the hot path.

    This cycles through a small set of actions to produce varied execution
    paths (different VDC rendering, different RAM states) so the profile
    data is representative of real training.
    """
    actions = [0, 1, 2, 3]  # cycle through available actions
    num_actions = len(actions)

    interface.reset_numpy(-1)

    resets = 0
    wall_start = time.perf_counter()

    for i in range(timesteps):
        action = actions[i % num_actions]
        result = interface.step_numpy([action])

        # Reset on episode end, just like training would
        if result.get("done", False) or result.get("truncated", False):
            interface.reset_numpy(-1)
            resets += 1

    wall_elapsed = time.perf_counter() - wall_start
    fps = timesteps / wall_elapsed if wall_elapsed > 0 else 0.0

    print(f"\nWorkload complete:")
    print(f"  Timesteps:  {timesteps}")
    print(f"  Resets:     {resets}")
    print(f"  Wall time:  {wall_elapsed:.3f} s")
    print(f"  Throughput: {fps:.0f} fps")


def main():
    args = _parse_args()

    if args.timesteps <= 0:
        sys.exit(f"Error: --timesteps must be > 0, got {args.timesteps}")

    # Resolve the emulator interface
    interface = None
    profile_name = "N/A"

    if args.game_profile:
        from retro_ai.training.game_profile import GameProfileRegistry

        registry = GameProfileRegistry()
        try:
            profile = registry.load(args.game_profile)
        except Exception as e:
            sys.exit(f"Error loading game profile: {e}")
        interface = _create_interface_from_profile(profile)
        profile_name = profile.name
    elif args.emulator and args.rom:
        interface = _create_interface_from_args(args.emulator, args.rom, args.bios)
        profile_name = f"{args.emulator} ({args.rom})"
    else:
        sys.exit(
            "Error: provide either --game-profile or both --emulator and --rom.\n"
            "Run with --help for usage."
        )

    print(
        f"PGO training workload: {args.timesteps} steps on {profile_name}\n"
        f"Exercising: step → framebuffer extraction → reward computation"
    )

    run_workload(interface, args.timesteps)

    print(
        "\nProfile data has been written by the instrumented binary.\n"
        "Next steps:\n"
        "  - Clang: llvm-profdata merge -output=build/pgo/pgo-data/default.profdata "
        "build/pgo/pgo-data/default.profraw\n"
        "  - GCC:   profile data is ready to use (no merge needed)\n"
        "  - Rebuild: cmake -B build/pgo -DPGO_USE=ON && "
        "cmake --build build/pgo --target retro_ai_native -j$(nproc)"
    )


if __name__ == "__main__":
    main()
