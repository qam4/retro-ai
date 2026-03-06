#!/usr/bin/env python3
"""RAM Watcher: discover score-related RAM addresses by diffing emulator state.

Interactive CLI tool that runs an emulator frame-by-frame and lets you mark
snapshots when the score changes.  Addresses that change monotonically across
multiple marks are likely score candidates.

Usage examples:
    python scripts/ram_watcher.py --profile game_profiles/videopac_course_automobile.yaml
    python scripts/ram_watcher.py --bios roms/bios.bin --rom roms/game.bin

Commands (enter at the prompt):
    <Enter>  Advance N frames (default 60)
    m        Mark current snapshot and diff against previous mark
    f        Show monotonic filter results (addresses that only increased)
    y        Output discovered addresses as YAML
    n <N>    Set frames-per-advance to N
    q        Quit
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Pure-logic helpers (importable for testing)
# ---------------------------------------------------------------------------


def compute_diff(
    old_snapshot: bytes, new_snapshot: bytes
) -> List[Dict[str, object]]:
    """Compare two snapshots byte-by-byte and return changed addresses.

    Each entry in the returned list is a dict with keys:
        address (int), old_value (int), new_value (int)

    Only the overlapping region (min length) is compared.
    """
    length = min(len(old_snapshot), len(new_snapshot))
    changes: List[Dict[str, object]] = []
    for i in range(length):
        old_val = old_snapshot[i]
        new_val = new_snapshot[i]
        if old_val != new_val:
            changes.append(
                {"address": i, "old_value": old_val, "new_value": new_val}
            )
    return changes


def filter_monotonic(
    snapshots: List[bytes],
) -> List[int]:
    """Return addresses whose byte values strictly increased across all consecutive snapshot pairs.

    Requires at least 2 snapshots.  An address qualifies only if its value
    increased (not just changed) in *every* consecutive pair.
    """
    if len(snapshots) < 2:
        return []

    length = min(len(s) for s in snapshots)
    # Start with all addresses as candidates
    candidates = set(range(length))

    for idx in range(1, len(snapshots)):
        prev = snapshots[idx - 1]
        curr = snapshots[idx]
        disqualified = set()
        for addr in candidates:
            if curr[addr] <= prev[addr]:
                disqualified.add(addr)
        candidates -= disqualified
        if not candidates:
            break

    return sorted(candidates)


def format_diff_table(changes: List[Dict[str, object]]) -> str:
    """Format a list of changes into a human-readable table."""
    if not changes:
        return "  (no changes)"
    lines = ["  Address      Old          New"]
    lines.append("  " + "-" * 40)
    for c in changes:
        addr = c["address"]
        old = c["old_value"]
        new = c["new_value"]
        lines.append(
            f"  0x{addr:04X} ({addr:5d})  {old:3d} (0x{old:02X})  ->  {new:3d} (0x{new:02X})"
        )
    return "\n".join(lines)


def addresses_to_yaml(addresses: List[int], is_bcd: bool = True) -> str:
    """Format discovered addresses as YAML compatible with reward_params.score_addresses."""
    entries = []
    for addr in addresses:
        entries.append(
            {"address": f"0x{addr:04X}", "num_bytes": 1, "is_bcd": is_bcd}
        )
    data = {"score_addresses": entries}
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the RAM watcher CLI."""
    parser = argparse.ArgumentParser(
        description="RAM Watcher: discover score-related RAM addresses by diffing emulator state snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to a game profile YAML file.",
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=None,
        help="Path to the ROM file (use with --bios for Videopac).",
    )
    parser.add_argument(
        "--bios",
        type=str,
        default=None,
        help="Path to the BIOS file (required for Videopac).",
    )
    parser.add_argument(
        "--emulator",
        type=str,
        default="videopac",
        choices=["videopac", "mo5"],
        help="Emulator type (default: videopac). Ignored when --profile is used.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to advance per step (default: 60).",
    )
    parser.add_argument(
        "--bcd",
        action="store_true",
        default=True,
        help="Assume discovered addresses are BCD-encoded (default: true).",
    )
    parser.add_argument(
        "--no-bcd",
        action="store_true",
        default=False,
        help="Assume discovered addresses are NOT BCD-encoded.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_bcd:
        args.bcd = False

    # Validate: need either --profile or --rom
    if args.profile is None and args.rom is None:
        parser.error("Either --profile or --rom is required.")

    return args


# ---------------------------------------------------------------------------
# Emulator setup
# ---------------------------------------------------------------------------


def create_emulator(args: argparse.Namespace):
    """Create an emulator instance from CLI args.

    Returns the native RLInterface instance, or exits with an error message
    if the emulator cannot be created or doesn't support RAM inspection.
    """
    try:
        import retro_ai_native  # noqa: F401
    except ImportError:
        print(
            "ERROR: retro_ai_native module not found.\n"
            "Make sure the native module is built and on PYTHONPATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.profile:
        # Load from game profile
        try:
            from retro_ai.training.game_profile import GameProfile

            profile = GameProfile.from_yaml(args.profile)
        except Exception as e:
            print(f"ERROR: Failed to load game profile: {e}", file=sys.stderr)
            sys.exit(1)

        emu_type = profile.emulator_type.lower()
        rom_path = profile.rom_path
        bios_path = profile.bios_path
    else:
        emu_type = args.emulator.lower()
        rom_path = args.rom
        bios_path = args.bios

    # Validate paths
    if not rom_path or not os.path.isfile(rom_path):
        print(f"ERROR: ROM file not found: {rom_path}", file=sys.stderr)
        sys.exit(1)

    if emu_type == "videopac":
        if not bios_path or not os.path.isfile(bios_path):
            print(f"ERROR: BIOS file not found: {bios_path}", file=sys.stderr)
            sys.exit(1)

    # Create the emulator
    try:
        if emu_type == "videopac":
            emu = retro_ai_native.VideopacRLInterface(bios_path, rom_path)
        elif emu_type == "mo5":
            emu = retro_ai_native.MO5RLInterface(rom_path)
        else:
            print(f"ERROR: Unknown emulator type: {emu_type}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to create emulator: {e}", file=sys.stderr)
        sys.exit(1)

    # Check that save_state works (our RAM inspection mechanism)
    try:
        emu.reset()
        state = emu.save_state()
        if not state or len(state) == 0:
            print(
                "ERROR: Emulator does not support memory inspection.\n"
                "save_state() returned empty data. RAM watching requires\n"
                "an emulator that exposes its internal state.",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as e:
        print(
            f"ERROR: Emulator does not support memory inspection.\n"
            f"save_state() failed: {e}\n"
            f"RAM watching requires an emulator that exposes its internal state.",
            file=sys.stderr,
        )
        sys.exit(1)

    return emu


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def run_interactive(emu, frames_per_step: int = 60, is_bcd: bool = True) -> None:
    """Run the interactive RAM watcher loop."""
    print("\n=== RAM Watcher ===")
    print(f"Frames per advance: {frames_per_step}")
    print("Commands: <Enter>=advance  m=mark  f=filter  y=yaml  n <N>=set frames  q=quit\n")

    # Take initial snapshot
    current_snapshot: bytes = emu.save_state()
    marked_snapshots: List[bytes] = [current_snapshot]
    total_frames = 0
    mark_count = 0

    print(f"Initial snapshot captured ({len(current_snapshot)} bytes)")
    print(f"Mark #0 set (baseline)\n")

    while True:
        try:
            cmd = input(f"[frame {total_frames}, marks: {mark_count + 1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd == "q":
            print("Exiting.")
            break

        elif cmd.startswith("n "):
            # Set frames per advance
            try:
                frames_per_step = int(cmd.split()[1])
                print(f"Frames per advance set to {frames_per_step}")
            except (ValueError, IndexError):
                print("Usage: n <number>")

        elif cmd == "" or cmd == "s":
            # Advance frames
            for _ in range(frames_per_step):
                emu.step([0])  # NOOP action
                total_frames += 1
            current_snapshot = emu.save_state()
            print(f"Advanced {frames_per_step} frames (total: {total_frames})")

        elif cmd == "m":
            # Mark and diff
            current_snapshot = emu.save_state()
            mark_count += 1
            prev_snapshot = marked_snapshots[-1]
            changes = compute_diff(prev_snapshot, current_snapshot)
            marked_snapshots.append(current_snapshot)

            print(f"\n--- Mark #{mark_count} (frame {total_frames}) ---")
            print(f"Comparing against mark #{mark_count - 1}")
            print(f"Changed addresses: {len(changes)}")
            print(format_diff_table(changes))
            print()

        elif cmd == "f":
            # Show monotonic filter
            if len(marked_snapshots) < 3:
                print(
                    f"Need at least 3 marks for monotonic filtering "
                    f"(have {len(marked_snapshots)}). "
                    f"Mark more snapshots after score increases."
                )
            else:
                mono_addrs = filter_monotonic(marked_snapshots)
                print(f"\n--- Monotonic filter ({len(marked_snapshots)} marks) ---")
                print(f"Addresses that strictly increased across all marks: {len(mono_addrs)}")
                if mono_addrs:
                    # Show current values
                    latest = marked_snapshots[-1]
                    for addr in mono_addrs:
                        val = latest[addr]
                        print(f"  0x{addr:04X} ({addr:5d})  current value: {val:3d} (0x{val:02X})")
                else:
                    print("  (none found — try marking after clear score increases)")
                print()

        elif cmd == "y":
            # Output YAML
            if len(marked_snapshots) < 3:
                print("Need at least 3 marks for YAML output. Use 'f' to preview first.")
                continue
            mono_addrs = filter_monotonic(marked_snapshots)
            if not mono_addrs:
                print("No monotonically increasing addresses found.")
                continue
            print("\n--- YAML output (paste into game profile) ---")
            print(addresses_to_yaml(mono_addrs, is_bcd=is_bcd))

        else:
            print("Unknown command. Commands: <Enter>=advance  m=mark  f=filter  y=yaml  n <N>=set frames  q=quit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the RAM watcher script."""
    args = parse_args(argv)
    emu = create_emulator(args)
    run_interactive(emu, frames_per_step=args.frames, is_bcd=args.bcd)


if __name__ == "__main__":
    main()
