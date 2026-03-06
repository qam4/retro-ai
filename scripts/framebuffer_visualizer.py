#!/usr/bin/env python3
"""Framebuffer Visualizer: identify score screen regions on the emulator display.

Headless-friendly tool that saves the emulator framebuffer as a scaled PNG,
lets you enter coordinates via CLI, overlays a verification rectangle, and
outputs YAML-compatible screen_region coordinates.

Usage examples:
    python scripts/framebuffer_visualizer.py --profile game_profiles/videopac_course_automobile.yaml
    python scripts/framebuffer_visualizer.py --bios roms/bios.bin --rom roms/game.bin
    python scripts/framebuffer_visualizer.py --profile game_profiles/mo5_game.yaml --scale 4

Workflow:
    1. Script saves the current framebuffer as a scaled PNG
    2. Examine the PNG to identify the score region
    3. Enter coordinates (x, y, width, height) in native emulator resolution
    4. Script overlays a rectangle on the image and saves a verification PNG
    5. Confirm or adjust coordinates
    6. Script outputs YAML

Commands (enter at the prompt):
    <Enter>     Advance N frames (default 60)
    n <N>       Set frames-per-advance to N
    save        Save current framebuffer as PNG
    select      Enter region coordinates interactively
    coords X Y W H   Set region directly (native resolution)
    verify      Overlay current region on framebuffer and save verification PNG
    yaml        Output current region as YAML
    info        Show emulator framebuffer dimensions and current region
    q           Quit
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Pure-logic helpers (importable for testing)
# ---------------------------------------------------------------------------


def scaled_to_native(px: int, py: int, scale: int) -> Tuple[int, int]:
    """Convert scaled display coordinates to native emulator resolution.

    Args:
        px: x coordinate in scaled space
        py: y coordinate in scaled space
        scale: scale factor (positive integer)

    Returns:
        (native_x, native_y) tuple
    """
    return (px // scale, py // scale)


def native_to_scaled(nx: int, ny: int, scale: int) -> Tuple[int, int]:
    """Convert native emulator coordinates to scaled display coordinates.

    Args:
        nx: x coordinate in native resolution
        ny: y coordinate in native resolution
        scale: scale factor (positive integer)

    Returns:
        (scaled_x, scaled_y) tuple
    """
    return (nx * scale, ny * scale)


def clamp_region(
    x: int, y: int, width: int, height: int, fb_width: int, fb_height: int
) -> Tuple[int, int, int, int]:
    """Clamp a region to fit within framebuffer bounds.

    Returns:
        (x, y, width, height) clamped to [0, fb_width) x [0, fb_height)
    """
    x = max(0, min(x, fb_width - 1))
    y = max(0, min(y, fb_height - 1))
    width = max(1, min(width, fb_width - x))
    height = max(1, min(height, fb_height - y))
    return (x, y, width, height)


def region_to_yaml(x: int, y: int, width: int, height: int) -> str:
    """Format a screen region as YAML compatible with reward_params.screen_region."""
    data = {"screen_region": {"x": x, "y": y, "width": width, "height": height}}
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Image helpers (require Pillow)
# ---------------------------------------------------------------------------


def _get_pillow():
    """Import and return PIL.Image, or exit with install instructions."""
    try:
        from PIL import Image, ImageDraw
        return Image, ImageDraw
    except ImportError:
        print(
            "ERROR: Pillow is required for the framebuffer visualizer.\n"
            "Install it with: pip install Pillow",
            file=sys.stderr,
        )
        sys.exit(1)


def framebuffer_to_image(obs_data, width: int, height: int, channels: int):
    """Convert raw observation data (numpy array or bytes) to a PIL Image.

    Args:
        obs_data: numpy array of shape (H, W, C) or flat bytes of length H*W*C
        width: framebuffer width
        height: framebuffer height
        channels: number of channels (3 for RGB)

    Returns:
        PIL.Image in RGB mode
    """
    Image, _ = _get_pillow()

    try:
        import numpy as np
        if isinstance(obs_data, np.ndarray):
            if obs_data.ndim == 3:
                # Shape is (H, W, C) — use directly
                return Image.fromarray(obs_data[:, :, :3], "RGB")
            else:
                # Flat array, reshape
                arr = obs_data.reshape((height, width, channels))
                return Image.fromarray(arr[:, :, :3], "RGB")
    except (ImportError, Exception):
        pass

    # Fallback: treat as raw bytes
    if isinstance(obs_data, (bytes, bytearray)):
        raw = bytes(obs_data)
    else:
        raw = bytes(obs_data)

    if channels >= 3:
        # Extract RGB from potentially RGBA data
        rgb_bytes = bytearray()
        for i in range(0, len(raw), channels):
            rgb_bytes.extend(raw[i:i + 3])
        return Image.frombytes("RGB", (width, height), bytes(rgb_bytes))
    else:
        return Image.frombytes("L", (width, height), raw)


def save_framebuffer_png(
    obs_data, width: int, height: int, channels: int,
    scale: int, output_path: str
) -> str:
    """Save the framebuffer as a scaled PNG.

    Returns:
        The path to the saved file.
    """
    Image, _ = _get_pillow()
    img = framebuffer_to_image(obs_data, width, height, channels)
    scaled = img.resize((width * scale, height * scale), Image.NEAREST)
    scaled.save(output_path)
    return output_path


def save_verification_png(
    obs_data, width: int, height: int, channels: int,
    scale: int, region: Tuple[int, int, int, int], output_path: str
) -> str:
    """Save the framebuffer with a rectangle overlay showing the selected region.

    Args:
        region: (x, y, w, h) in native emulator resolution

    Returns:
        The path to the saved file.
    """
    Image, ImageDraw = _get_pillow()
    img = framebuffer_to_image(obs_data, width, height, channels)
    scaled = img.resize((width * scale, height * scale), Image.NEAREST)

    draw = ImageDraw.Draw(scaled)
    rx, ry, rw, rh = region
    # Draw rectangle in scaled coordinates
    sx, sy = native_to_scaled(rx, ry, scale)
    ex, ey = native_to_scaled(rx + rw, ry + rh, scale)
    # Draw a 2px red rectangle
    draw.rectangle([sx, sy, ex - 1, ey - 1], outline="red", width=2)
    # Add coordinate label
    label = f"({rx},{ry}) {rw}x{rh}"
    draw.text((sx, max(0, sy - 12)), label, fill="red")

    scaled.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the framebuffer visualizer CLI."""
    parser = argparse.ArgumentParser(
        description="Framebuffer Visualizer: identify score screen regions on the emulator display.",
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
        "--scale",
        type=int,
        default=3,
        help="Scale factor for saved PNG images (default: 3).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to advance per step (default: 60).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for saved PNG files (default: current directory).",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate: need either --profile or --rom
    if args.profile is None and args.rom is None:
        parser.error("Either --profile or --rom is required.")

    if args.scale < 1:
        parser.error("--scale must be at least 1.")

    return args


# ---------------------------------------------------------------------------
# Emulator setup
# ---------------------------------------------------------------------------


def create_emulator(args: argparse.Namespace):
    """Create an emulator instance from CLI args.

    Returns:
        (emulator, obs_space) tuple where obs_space has width, height, channels.
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

    emu.reset()
    obs_space = emu.observation_space()

    return emu, obs_space


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def get_current_observation(emu):
    """Get the current framebuffer observation as a numpy array.

    Returns the observation from a step with NOOP action (no frame advance
    side-effect beyond one frame).
    """
    result = emu.step_numpy([0])
    return result["observation"]


def run_interactive(
    emu, obs_space, scale: int = 3, frames_per_step: int = 60,
    output_dir: str = "."
) -> None:
    """Run the interactive framebuffer visualizer loop."""
    fb_width = obs_space.width
    fb_height = obs_space.height
    channels = obs_space.channels

    print("\n=== Framebuffer Visualizer ===")
    print(f"Emulator resolution: {fb_width}x{fb_height} ({channels}ch)")
    print(f"Scale factor: {scale}x  (saved images: {fb_width * scale}x{fb_height * scale})")
    print(f"Frames per advance: {frames_per_step}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print()
    print("Commands: save  select  coords X Y W H  verify  yaml  info  n <N>  <Enter>=advance  q=quit")
    print()

    os.makedirs(output_dir, exist_ok=True)

    current_region: Optional[Tuple[int, int, int, int]] = None
    total_frames = 0
    save_count = 0

    # Save initial framebuffer
    obs = get_current_observation(emu)
    initial_path = os.path.join(output_dir, "framebuffer_000.png")
    save_framebuffer_png(obs, fb_width, fb_height, channels, scale, initial_path)
    print(f"Initial framebuffer saved to: {initial_path}")
    print()

    while True:
        region_str = ""
        if current_region:
            rx, ry, rw, rh = current_region
            region_str = f" region=({rx},{ry},{rw},{rh})"

        try:
            cmd = input(f"[frame {total_frames}{region_str}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if cmd == "q":
            print("Exiting.")
            break

        elif cmd == "" or cmd == "s":
            # Advance frames
            for _ in range(frames_per_step):
                emu.step([0])
                total_frames += 1
            obs = get_current_observation(emu)
            total_frames += 1
            print(f"Advanced {frames_per_step} frames (total: {total_frames})")

        elif cmd.startswith("n "):
            try:
                frames_per_step = int(cmd.split()[1])
                print(f"Frames per advance set to {frames_per_step}")
            except (ValueError, IndexError):
                print("Usage: n <number>")

        elif cmd == "save":
            obs = get_current_observation(emu)
            total_frames += 1
            save_count += 1
            path = os.path.join(output_dir, f"framebuffer_{save_count:03d}.png")
            save_framebuffer_png(obs, fb_width, fb_height, channels, scale, path)
            print(f"Framebuffer saved to: {path}")

        elif cmd == "select":
            # Interactive coordinate entry
            print(f"\nEnter region coordinates in native resolution (0-{fb_width - 1} x 0-{fb_height - 1}):")
            try:
                x_str = input("  x (left edge): ").strip()
                y_str = input("  y (top edge):  ").strip()
                w_str = input("  width:         ").strip()
                h_str = input("  height:        ").strip()
                x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)
                x, y, w, h = clamp_region(x, y, w, h, fb_width, fb_height)
                current_region = (x, y, w, h)
                print(f"Region set to: x={x}, y={y}, width={w}, height={h}")
                print("Use 'verify' to overlay on framebuffer, 'yaml' to output.")
            except (ValueError, EOFError):
                print("Invalid input. Enter integer values.")

        elif cmd.startswith("coords "):
            # Direct coordinate entry: coords X Y W H
            parts = cmd.split()
            if len(parts) != 5:
                print("Usage: coords X Y W H")
                continue
            try:
                x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                x, y, w, h = clamp_region(x, y, w, h, fb_width, fb_height)
                current_region = (x, y, w, h)
                print(f"Region set to: x={x}, y={y}, width={w}, height={h}")
            except ValueError:
                print("Invalid coordinates. Use integers: coords X Y W H")

        elif cmd == "verify":
            if current_region is None:
                print("No region selected. Use 'select' or 'coords X Y W H' first.")
                continue
            obs = get_current_observation(emu)
            total_frames += 1
            save_count += 1
            path = os.path.join(output_dir, f"framebuffer_verify_{save_count:03d}.png")
            save_verification_png(
                obs, fb_width, fb_height, channels, scale, current_region, path
            )
            rx, ry, rw, rh = current_region
            print(f"Verification image saved to: {path}")
            print(f"  Region: x={rx}, y={ry}, width={rw}, height={rh}")
            print("  Check the red rectangle in the image. Use 'yaml' to output or 'select' to adjust.")

        elif cmd == "yaml":
            if current_region is None:
                print("No region selected. Use 'select' or 'coords X Y W H' first.")
                continue
            rx, ry, rw, rh = current_region
            print("\n--- YAML output (paste into game profile under reward_params:) ---")
            print(region_to_yaml(rx, ry, rw, rh))

        elif cmd == "info":
            print(f"\nEmulator resolution: {fb_width}x{fb_height} ({channels} channels)")
            print(f"Scale factor: {scale}x  (saved images: {fb_width * scale}x{fb_height * scale})")
            print(f"Frames per advance: {frames_per_step}")
            print(f"Total frames: {total_frames}")
            if current_region:
                rx, ry, rw, rh = current_region
                print(f"Current region: x={rx}, y={ry}, width={rw}, height={rh}")
            else:
                print("Current region: (none)")
            print()

        else:
            print("Unknown command. Commands: save  select  coords X Y W H  verify  yaml  info  n <N>  <Enter>=advance  q=quit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the framebuffer visualizer script."""
    args = parse_args(argv)
    emu, obs_space = create_emulator(args)
    run_interactive(
        emu, obs_space,
        scale=args.scale,
        frames_per_step=args.frames,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
