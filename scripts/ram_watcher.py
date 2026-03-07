#!/usr/bin/env python3
"""RAM Watcher: discover score-related RAM addresses by diffing emulator RAM.

Interactive CLI tool that runs an emulator frame-by-frame and lets you mark
snapshots when the score changes.  Addresses that change monotonically across
multiple marks are likely score candidates.

Videopac RAM layout (192 bytes total):
  Bytes  0-63:  8048 internal RAM
    0-7:   Register bank 0 (R0-R7)
    8-23:  Stack (8 levels × 2 bytes)
    24-31: Register bank 1 (R0'-R7')
    32-63: General purpose RAM (game variables: score, timer, lives, etc.)
  Bytes 64-191: External RAM (128 bytes, VDC-accessible)

Usage examples:
    python scripts/ram_watcher.py --profile game_profiles/videopac_course_automobile.yaml
    python scripts/ram_watcher.py --bios roms/bios.bin --rom roms/game.bin

Commands (enter at the prompt):
    <Enter>  Advance N frames (default 60)
    m        Mark current RAM and diff against previous mark
    f        Show monotonic filter results (addresses that only increased)
    d        Show monotonic-decreasing filter (addresses that only decreased)
    a <N>    Send action N for one frame then advance remaining frames
    hold <N> Hold action N for all N frames of the advance
    n <N>    Set frames-per-advance to N
    y        Output discovered addresses as YAML
    dump     Dump all 192 RAM bytes
    watch <addr> [addr2 ...]  Watch specific addresses each advance
    unwatch  Clear watch list
    q        Quit
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Set

import yaml


# ---------------------------------------------------------------------------
# Pure-logic helpers (importable for testing)
# ---------------------------------------------------------------------------

# Videopac RAM layout constants
INTERNAL_RAM_SIZE = 64
EXTERNAL_RAM_SIZE = 128
TOTAL_RAM_SIZE = INTERNAL_RAM_SIZE + EXTERNAL_RAM_SIZE

# Meaningful internal RAM range (game variables live in 32-63)
GAME_RAM_START = 32
GAME_RAM_END = 64


def compute_diff(
    old_snapshot: bytes, new_snapshot: bytes
) -> List[Dict[str, object]]:
    """Compare two RAM snapshots byte-by-byte and return changed addresses."""
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


def filter_monotonic(snapshots: List[bytes], increasing: bool = True) -> List[int]:
    """Return addresses whose byte values strictly increased (or decreased) across all consecutive pairs."""
    if len(snapshots) < 2:
        return []

    length = min(len(s) for s in snapshots)
    candidates = set(range(length))

    for idx in range(1, len(snapshots)):
        prev = snapshots[idx - 1]
        curr = snapshots[idx]
        disqualified = set()
        for addr in candidates:
            if increasing:
                if curr[addr] <= prev[addr]:
                    disqualified.add(addr)
            else:
                if curr[addr] >= prev[addr]:
                    disqualified.add(addr)
        candidates -= disqualified
        if not candidates:
            break

    return sorted(candidates)


def addr_label(addr: int) -> str:
    """Return a human-readable label for a RAM address."""
    if addr < 8:
        return f"R{addr} (bank0)"
    elif addr < 24:
        level = (addr - 8) // 2
        byte = (addr - 8) % 2
        return f"Stack L{level}.{'hi' if byte else 'lo'}"
    elif addr < 32:
        return f"R{addr - 24}' (bank1)"
    elif addr < INTERNAL_RAM_SIZE:
        return f"IntRAM[{addr}]"
    else:
        ext = addr - INTERNAL_RAM_SIZE
        return f"ExtRAM[0x{ext:02X}]"


def format_diff_table(changes: List[Dict[str, object]]) -> str:
    """Format a list of changes into a human-readable table."""
    if not changes:
        return "  (no changes)"
    lines = ["  Address      Region              Old          New"]
    lines.append("  " + "-" * 60)
    for c in changes:
        addr = c["address"]
        old = c["old_value"]
        new = c["new_value"]
        label = addr_label(addr)
        lines.append(
            f"  0x{addr:04X} ({addr:3d})  {label:<20s} {old:3d} (0x{old:02X})  ->  {new:3d} (0x{new:02X})"
        )
    return "\n".join(lines)


def format_watch(ram: bytes, addresses: List[int]) -> str:
    """Format watched addresses with current values."""
    if not addresses:
        return ""
    lines = ["  Watched:"]
    for addr in addresses:
        if addr < len(ram):
            val = ram[addr]
            label = addr_label(addr)
            lines.append(f"    0x{addr:04X} ({addr:3d}) {label:<20s} = {val:3d} (0x{val:02X})")
    return "\n".join(lines)


def dump_ram(ram: bytes) -> str:
    """Hex dump of all RAM bytes."""
    lines = []
    lines.append("  === 8048 Internal RAM (64 bytes) ===")
    for row_start in range(0, INTERNAL_RAM_SIZE, 16):
        hex_vals = " ".join(f"{ram[row_start + i]:02X}" for i in range(16) if row_start + i < INTERNAL_RAM_SIZE)
        ascii_vals = "".join(
            chr(ram[row_start + i]) if 32 <= ram[row_start + i] < 127 else "."
            for i in range(16) if row_start + i < INTERNAL_RAM_SIZE
        )
        lines.append(f"  0x{row_start:04X}: {hex_vals:<48s} {ascii_vals}")

    lines.append("")
    lines.append("  === External RAM (128 bytes) ===")
    for row_start in range(INTERNAL_RAM_SIZE, TOTAL_RAM_SIZE, 16):
        hex_vals = " ".join(f"{ram[row_start + i]:02X}" for i in range(16) if row_start + i < TOTAL_RAM_SIZE)
        ascii_vals = "".join(
            chr(ram[row_start + i]) if 32 <= ram[row_start + i] < 127 else "."
            for i in range(16) if row_start + i < TOTAL_RAM_SIZE
        )
        ext_addr = row_start - INTERNAL_RAM_SIZE
        lines.append(f"  0x{row_start:04X} (ext 0x{ext_addr:02X}): {hex_vals:<48s} {ascii_vals}")

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
        description="RAM Watcher: discover score/timer RAM addresses by diffing emulator RAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Path to a game profile YAML file.",
    )
    parser.add_argument(
        "--rom", type=str, default=None,
        help="Path to the ROM file (use with --bios for Videopac).",
    )
    parser.add_argument(
        "--bios", type=str, default=None,
        help="Path to the BIOS file (required for Videopac).",
    )
    parser.add_argument(
        "--emulator", type=str, default="videopac",
        choices=["videopac", "mo5"],
        help="Emulator type (default: videopac). Ignored when --profile is used.",
    )
    parser.add_argument(
        "--frames", type=int, default=60,
        help="Number of frames to advance per step (default: 60, ~1 second at 60fps).",
    )
    parser.add_argument(
        "--bcd", action="store_true", default=True,
        help="Assume discovered addresses are BCD-encoded (default: true).",
    )
    parser.add_argument(
        "--no-bcd", action="store_true", default=False,
        help="Assume discovered addresses are NOT BCD-encoded.",
    )
    parser.add_argument(
        "--joystick", type=int, default=None,
        help="Joystick index (0 or 1). Overrides game profile setting.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_bcd:
        args.bcd = False

    if args.profile is None and args.rom is None:
        parser.error("Either --profile or --rom is required.")

    return args


# ---------------------------------------------------------------------------
# Emulator setup
# ---------------------------------------------------------------------------


def create_emulator(args: argparse.Namespace):
    """Create an emulator instance from CLI args."""
    try:
        import retro_ai_native
    except ImportError:
        print(
            "ERROR: retro_ai_native module not found.\n"
            "Make sure the native module is built and on PYTHONPATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    joystick_index = 0
    if args.profile:
        try:
            from retro_ai.training.game_profile import GameProfile
            profile = GameProfile.from_yaml(args.profile)
        except Exception as e:
            print(f"ERROR: Failed to load game profile: {e}", file=sys.stderr)
            sys.exit(1)

        emu_type = profile.emulator_type.lower()
        rom_path = profile.rom_path
        bios_path = profile.bios_path
        joystick_index = getattr(profile, "joystick_index", 0) or 0
    else:
        emu_type = args.emulator.lower()
        rom_path = args.rom
        bios_path = args.bios

    if args.joystick is not None:
        joystick_index = args.joystick

    if not rom_path or not os.path.isfile(rom_path):
        print(f"ERROR: ROM file not found: {rom_path}", file=sys.stderr)
        sys.exit(1)

    if emu_type == "videopac":
        if not bios_path or not os.path.isfile(bios_path):
            print(f"ERROR: BIOS file not found: {bios_path}", file=sys.stderr)
            sys.exit(1)

    try:
        if emu_type == "videopac":
            emu = retro_ai_native.VideopacRLInterface(
                bios_path, rom_path, "survival", joystick_index
            )
        elif emu_type == "mo5":
            emu = retro_ai_native.MO5RLInterface(rom_path)
        else:
            print(f"ERROR: Unknown emulator type: {emu_type}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to create emulator: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify read_ram works
    try:
        emu.reset()
        ram = emu.read_ram()
        if not ram or len(ram) == 0:
            print(
                "ERROR: Emulator does not support RAM inspection.\n"
                "read_ram() returned empty data.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"RAM size: {len(ram)} bytes")
    except Exception as e:
        print(f"ERROR: read_ram() failed: {e}", file=sys.stderr)
        sys.exit(1)

    return emu


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: "NOOP", 1: "Up", 2: "Down", 3: "Left", 4: "Right",
    5: "Fire", 6: "Up+Fire", 7: "Down+Fire", 8: "Left+Fire", 9: "Right+Fire",
    10: "Key0", 11: "Key1", 12: "Key2", 13: "Key3",
    14: "Key4", 15: "Key5", 16: "Key6", 17: "Key7",
}


def run_interactive(emu, frames_per_step: int = 60, is_bcd: bool = True) -> None:
    """Run the interactive RAM watcher loop."""
    print("\n=== RAM Watcher ===")
    print(f"Frames per advance: {frames_per_step}")
    print("Actions: 0=NOOP 1=Up 2=Down 3=Left 4=Right 5=Fire 10-17=Key0-7")
    print("Commands: <Enter>=advance  m=mark  f=filter  d=decreasing  a <N>=action")
    print("          hold <N>=hold action  n <N>=set frames  y=yaml  dump=hex dump")
    print("          watch <addr>=watch  unwatch=clear  q=quit\n")

    current_ram: bytes = emu.read_ram()
    marked_snapshots: List[bytes] = [current_ram]
    total_frames = 0
    mark_count = 0
    watch_addrs: List[int] = []

    print(f"RAM snapshot: {len(current_ram)} bytes (internal: {min(INTERNAL_RAM_SIZE, len(current_ram))}, external: {max(0, len(current_ram) - INTERNAL_RAM_SIZE)})")
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
            try:
                frames_per_step = int(cmd.split()[1])
                print(f"Frames per advance set to {frames_per_step}")
            except (ValueError, IndexError):
                print("Usage: n <number>")

        elif cmd.startswith("watch "):
            try:
                parts = cmd.split()[1:]
                watch_addrs = []
                for p in parts:
                    addr = int(p, 16) if p.startswith("0x") else int(p)
                    watch_addrs.append(addr)
                print(f"Watching {len(watch_addrs)} address(es): {[f'0x{a:04X}' for a in watch_addrs]}")
            except (ValueError, IndexError):
                print("Usage: watch <addr> [addr2 ...] (decimal or 0xHEX)")

        elif cmd == "unwatch":
            watch_addrs = []
            print("Watch list cleared.")

        elif cmd == "" or cmd == "s":
            # Advance frames with NOOP
            for _ in range(frames_per_step):
                emu.step([0])
                total_frames += 1
            current_ram = emu.read_ram()
            print(f"Advanced {frames_per_step} frames (total: {total_frames})")
            if watch_addrs:
                print(format_watch(current_ram, watch_addrs))

        elif cmd.startswith("a "):
            # Send action for 1 frame, then NOOP for remaining
            try:
                action = int(cmd.split()[1])
                action_name = ACTION_NAMES.get(action, f"action_{action}")
                emu.step([action])
                total_frames += 1
                for _ in range(frames_per_step - 1):
                    emu.step([0])
                    total_frames += 1
                current_ram = emu.read_ram()
                print(f"Sent {action_name} + {frames_per_step - 1} NOOP (total: {total_frames})")
                if watch_addrs:
                    print(format_watch(current_ram, watch_addrs))
            except (ValueError, IndexError):
                print("Usage: a <action_number>")

        elif cmd.startswith("hold "):
            # Hold action for all frames
            try:
                action = int(cmd.split()[1])
                action_name = ACTION_NAMES.get(action, f"action_{action}")
                for _ in range(frames_per_step):
                    emu.step([action])
                    total_frames += 1
                current_ram = emu.read_ram()
                print(f"Held {action_name} for {frames_per_step} frames (total: {total_frames})")
                if watch_addrs:
                    print(format_watch(current_ram, watch_addrs))
            except (ValueError, IndexError):
                print("Usage: hold <action_number>")

        elif cmd == "m":
            current_ram = emu.read_ram()
            mark_count += 1
            prev_snapshot = marked_snapshots[-1]
            changes = compute_diff(prev_snapshot, current_ram)
            marked_snapshots.append(current_ram)

            print(f"\n--- Mark #{mark_count} (frame {total_frames}) ---")
            print(f"Comparing against mark #{mark_count - 1}")
            print(f"Changed addresses: {len(changes)}")
            print(format_diff_table(changes))
            print()

        elif cmd == "f":
            if len(marked_snapshots) < 3:
                print(
                    f"Need at least 3 marks for monotonic filtering "
                    f"(have {len(marked_snapshots)}). "
                    f"Mark more snapshots after score increases."
                )
            else:
                mono_addrs = filter_monotonic(marked_snapshots, increasing=True)
                print(f"\n--- Monotonic INCREASING filter ({len(marked_snapshots)} marks) ---")
                print(f"Addresses that strictly increased across all marks: {len(mono_addrs)}")
                if mono_addrs:
                    latest = marked_snapshots[-1]
                    for addr in mono_addrs:
                        val = latest[addr]
                        label = addr_label(addr)
                        print(f"  0x{addr:04X} ({addr:3d}) {label:<20s} = {val:3d} (0x{val:02X})")
                else:
                    print("  (none found — try marking after clear score increases)")
                print()

        elif cmd == "d":
            if len(marked_snapshots) < 3:
                print(
                    f"Need at least 3 marks for monotonic filtering "
                    f"(have {len(marked_snapshots)}). "
                    f"Mark more snapshots after timer decreases."
                )
            else:
                mono_addrs = filter_monotonic(marked_snapshots, increasing=False)
                print(f"\n--- Monotonic DECREASING filter ({len(marked_snapshots)} marks) ---")
                print(f"Addresses that strictly decreased across all marks: {len(mono_addrs)}")
                if mono_addrs:
                    latest = marked_snapshots[-1]
                    for addr in mono_addrs:
                        val = latest[addr]
                        label = addr_label(addr)
                        print(f"  0x{addr:04X} ({addr:3d}) {label:<20s} = {val:3d} (0x{val:02X})")
                else:
                    print("  (none found — try marking after timer/counter decreases)")
                print()

        elif cmd == "y":
            if len(marked_snapshots) < 3:
                print("Need at least 3 marks for YAML output. Use 'f' to preview first.")
                continue
            mono_addrs = filter_monotonic(marked_snapshots, increasing=True)
            if not mono_addrs:
                print("No monotonically increasing addresses found.")
                continue
            print("\n--- YAML output (paste into game profile reward_params) ---")
            print(addresses_to_yaml(mono_addrs, is_bcd=is_bcd))

        elif cmd == "dump":
            current_ram = emu.read_ram()
            print(dump_ram(current_ram))

        else:
            print("Unknown command. Type q to quit, or press Enter to advance.")


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
