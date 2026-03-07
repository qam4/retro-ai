#!/usr/bin/env python3
"""Automated RAM scanner for retro-ai game profiles.

Plays a game using a specified action, samples RAM periodically, and reports
addresses that consistently increase (score candidates) or decrease (timer
candidates). This is the first step when adding a new game to retro-ai.

Usage:
    # Using a game profile (recommended):
    python scripts/scan_ram.py --profile game_profiles/videopac_course_automobile.yaml

    # With explicit ROM paths:
    python scripts/scan_ram.py --bios path/to/bios.bin --rom path/to/rom.bin

    # Custom action and duration:
    python scripts/scan_ram.py --profile ... --action 1 --seconds 20 --interval 1.0

    # Watch specific addresses in detail:
    python scripts/scan_ram.py --profile ... --detail 54,55,65,66

Workflow for a new game:
    1. Run this script with --action set to the main gameplay action (usually Up=1)
    2. Look at INCREASING addresses → score candidates
    3. Look at DECREASING addresses → timer candidates
    4. Re-run with --detail on the candidate addresses to confirm
    5. Update the game profile YAML with the discovered addresses
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build', 'ci-linux'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'python'))


# ---------------------------------------------------------------------------
# Address labeling (Videopac 8048 layout, extensible per emulator)
# ---------------------------------------------------------------------------

def videopac_label(addr: int) -> str:
    """Human-readable label for a Videopac RAM address."""
    if addr < 8:
        return f"R{addr} (bank0)"
    elif addr < 24:
        return "Stack"
    elif addr < 32:
        return f"R{addr - 24}' (bank1)"
    elif addr < 64:
        return f"IntRAM[{addr}]"
    else:
        return f"ExtRAM[0x{addr - 64:02X}]"


def generic_label(addr: int) -> str:
    return f"RAM[0x{addr:04X}]"


LABEL_FUNCS = {
    "videopac": videopac_label,
}


# ---------------------------------------------------------------------------
# Emulator creation (reuses GameProfile when available)
# ---------------------------------------------------------------------------

def create_emulator(args: argparse.Namespace) -> Tuple:
    """Create emulator and return (emu, label_func, startup_actions)."""
    import retro_ai_native

    startup_actions: List[Tuple[int, int]] = []  # (action, frames)
    label_func = generic_label
    joystick_index = args.joystick

    if args.profile:
        from retro_ai.training.game_profile import GameProfile
        profile = GameProfile.from_yaml(args.profile)
        emu_type = profile.emulator_type.lower()
        rom_path = profile.rom_path
        bios_path = profile.bios_path
        if joystick_index is None:
            joystick_index = profile.joystick_index or 0
        if profile.startup_sequence:
            for sa in profile.startup_sequence.actions:
                startup_actions.append((sa.action, sa.frames))
    else:
        emu_type = args.emulator
        rom_path = args.rom
        bios_path = args.bios
        if joystick_index is None:
            joystick_index = 0

    label_func = LABEL_FUNCS.get(emu_type, generic_label)

    if emu_type == "videopac":
        if not bios_path:
            print("ERROR: --bios required for Videopac", file=sys.stderr)
            sys.exit(1)
        emu = retro_ai_native.VideopacRLInterface(
            bios_path, rom_path, "survival", joystick_index)
    elif emu_type == "mo5":
        emu = retro_ai_native.MO5RLInterface(rom_path)
    else:
        print(f"ERROR: Unknown emulator type: {emu_type}", file=sys.stderr)
        sys.exit(1)

    return emu, label_func, startup_actions


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def run_scan(emu, label_func, startup_actions, action, seconds, interval,
             detail_addrs=None):
    """Run the automated RAM scan."""
    fps = 60
    interval_frames = int(interval * fps)
    num_samples = int(seconds / interval)

    # --- Phase 1: Reset and run startup sequence ---
    print("Phase 1: Reset + startup sequence...")
    emu.reset()
    for act, frames in startup_actions:
        for _ in range(frames):
            emu.step([act])
    # Post-startup settle
    for _ in range(120):
        emu.step([0])

    # --- Phase 2: Start gameplay and take baseline ---
    print(f"Phase 2: Playing action {action} for {seconds}s, "
          f"sampling every {interval}s...")
    for _ in range(fps):  # 1 second of gameplay action to start things
        emu.step([action])

    baseline = emu.read_ram()
    ram_size = len(baseline)
    print(f"RAM size: {ram_size} bytes\n")

    # --- Phase 3: Collect samples ---
    samples = [baseline]
    for i in range(num_samples):
        for _ in range(interval_frames):
            emu.step([action])
        samples.append(emu.read_ram())

    num_pairs = len(samples) - 1
    print(f"Collected {len(samples)} samples over ~{seconds}s\n")

    # --- Phase 4: Analyze ---
    inc_count = [0] * ram_size
    dec_count = [0] * ram_size
    same_count = [0] * ram_size

    for i in range(1, len(samples)):
        prev, curr = samples[i - 1], samples[i]
        for addr in range(ram_size):
            if curr[addr] > prev[addr]:
                inc_count[addr] += 1
            elif curr[addr] < prev[addr]:
                dec_count[addr] += 1
            else:
                same_count[addr] += 1

    # --- Report ---
    threshold = 0.6

    print("=" * 70)
    print("INCREASING addresses (score candidates)")
    print("=" * 70)
    hdr = (f"{'Addr':<14} {'Region':<18} {'Inc':>4}/{num_pairs}  "
           f"{'Dec':>4}  {'Same':>4}  Values")
    print(hdr)
    print("-" * 70)
    found_inc = False
    for addr in range(ram_size):
        if inc_count[addr] >= num_pairs * threshold:
            found_inc = True
            vals = [s[addr] for s in samples]
            val_str = " -> ".join(f"0x{v:02X}" for v in vals[::max(1, len(vals)//6)])
            print(f"0x{addr:04X} ({addr:3d})  {label_func(addr):<18} "
                  f"{inc_count[addr]:4d}     {dec_count[addr]:4d}  "
                  f"{same_count[addr]:4d}  {val_str}")
    if not found_inc:
        print("  (none found)")

    print()
    print("=" * 70)
    print("DECREASING addresses (timer candidates)")
    print("=" * 70)
    print(hdr.replace("Inc", "Dec"))
    print("-" * 70)
    found_dec = False
    for addr in range(ram_size):
        if dec_count[addr] >= num_pairs * threshold:
            found_dec = True
            vals = [s[addr] for s in samples]
            val_str = " -> ".join(f"0x{v:02X}" for v in vals[::max(1, len(vals)//6)])
            print(f"0x{addr:04X} ({addr:3d})  {label_func(addr):<18} "
                  f"{dec_count[addr]:4d}     {inc_count[addr]:4d}  "
                  f"{same_count[addr]:4d}  {val_str}")
    if not found_dec:
        print("  (none found)")

    # --- Top candidates detail ---
    print()
    print("=" * 70)
    print("DETAILED: Top candidates (all sample values)")
    print("=" * 70)

    candidates = []
    for addr in range(ram_size):
        score = max(inc_count[addr], dec_count[addr])
        if score >= num_pairs * 0.5:
            direction = "INC" if inc_count[addr] > dec_count[addr] else "DEC"
            candidates.append((score, addr, direction))

    candidates.sort(reverse=True)
    for score, addr, direction in candidates[:10]:
        vals = [s[addr] for s in samples]
        print(f"\n0x{addr:04X} ({addr:3d}) {label_func(addr):<18} [{direction}] "
              f"{inc_count[addr]} inc / {dec_count[addr]} dec / {same_count[addr]} same")
        for row_start in range(0, len(vals), 10):
            chunk = vals[row_start:row_start + 10]
            print(f"  [{row_start:2d}-{row_start+len(chunk)-1:2d}]: "
                  f"{' '.join(f'{v:3d}' for v in chunk)}")
            print(f"          {' '.join(f' {v:02X}' for v in chunk)}")

    # --- Detail mode: watch specific addresses ---
    if detail_addrs:
        print()
        print("=" * 70)
        print(f"DETAIL: Addresses {detail_addrs}")
        print("=" * 70)
        header = f"{'Sample':>6}"
        for a in detail_addrs:
            header += f"  {label_func(a):>18}"
        print(header)
        for i, s in enumerate(samples):
            row = f"{i:6d}"
            for a in detail_addrs:
                row += f"  {s[a]:18d}"
            hex_vals = ' '.join(f'{s[a]:02X}' for a in detail_addrs)
            print(f"{row}   (hex: {hex_vals})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan emulator RAM to discover score/timer addresses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Path to a game profile YAML file (recommended).",
    )
    parser.add_argument("--rom", type=str, default=None, help="ROM file path.")
    parser.add_argument("--bios", type=str, default=None, help="BIOS file path.")
    parser.add_argument(
        "--emulator", type=str, default="videopac",
        choices=["videopac", "mo5"],
        help="Emulator type (default: videopac). Ignored with --profile.",
    )
    parser.add_argument(
        "--joystick", type=int, default=None,
        help="Joystick index (overrides profile).",
    )
    parser.add_argument(
        "--action", type=int, default=1,
        help="Action to hold during scan (default: 1 = Up).",
    )
    parser.add_argument(
        "--seconds", type=float, default=15,
        help="Duration of gameplay to scan (default: 15).",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between RAM samples (default: 1.0).",
    )
    parser.add_argument(
        "--detail", type=str, default=None,
        help="Comma-separated addresses to watch in detail (e.g. 54,55,65,66).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.profile is None and args.rom is None:
        parser.error("Either --profile or --rom is required.")

    detail_addrs = None
    if args.detail:
        detail_addrs = [int(x.strip()) for x in args.detail.split(",")]

    emu, label_func, startup_actions = create_emulator(args)
    run_scan(emu, label_func, startup_actions, args.action, args.seconds,
             args.interval, detail_addrs)


if __name__ == "__main__":
    main()
