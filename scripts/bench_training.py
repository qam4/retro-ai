#!/usr/bin/env python3
"""Training throughput benchmark.

Measures end-to-end training fps and wall-clock time, with per-component
timing breakdown (emulator stepping, reward computation, preprocessing,
model inference).

Outputs both a human-readable summary and machine-readable JSON to stdout.
Defaults to CPU-only inference (no GPU required).

Usage examples:

  # With a game profile:
  PYTHONPATH=build/ci-linux:python python scripts/bench_training.py \
      --game-profile course_automobile \
      --timesteps 5000

  # With explicit parameters (no ROM/profile needed for dry-run):
  PYTHONPATH=build/ci-linux:python python scripts/bench_training.py \
      --game-profile course_automobile \
      --num-envs 4 \
      --frame-skip 4 \
      --observation-mode framebuffer \
      --resize 84 84 \
      --timesteps 10000
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark training throughput with per-component timings.",
    )
    parser.add_argument(
        "--game-profile",
        default=None,
        help="Game profile name or YAML path (uses game_profiles/ directory)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Frame skip value (default: 4)",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["framebuffer", "ram"],
        default="framebuffer",
        help="Observation mode (default: framebuffer)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[84, 84],
        help="Observation resize dimensions H W (default: 84 84)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Total training timesteps to run (default: 10000)",
    )
    return parser.parse_args()


def _build_config(args):
    """Build a TrainingConfig from CLI arguments."""
    from retro_ai.training.config import TrainingConfig, AlgorithmConfig

    return TrainingConfig(
        algorithm=AlgorithmConfig(name="PPO"),
        total_timesteps=args.timesteps,
        game_profile=args.game_profile,
        num_envs=args.num_envs,
        frame_skip=args.frame_skip,
        observation_mode=args.observation_mode,
        resize=tuple(args.resize),
        policy="CnnPolicy" if args.observation_mode == "framebuffer" else "MlpPolicy",
    )


def _resolve_profile(config):
    """Load and merge game profile if configured. Returns (config, profile)."""
    from retro_ai.training.config import merge_config_with_profile
    from retro_ai.training.game_profile import GameProfileRegistry

    profile = None
    if config.game_profile:
        registry = GameProfileRegistry()
        profile = registry.load(config.game_profile)
        config = merge_config_with_profile(config, profile)
    return config, profile


class _TimingTracker:
    """Accumulates per-component timing measurements."""

    def __init__(self):
        self.emulator_step_s = 0.0
        self.reward_s = 0.0
        self.preprocessing_s = 0.0
        self.model_inference_s = 0.0

    def to_ms_dict(self, n_steps: int) -> dict:
        """Return average per-step timings in milliseconds."""
        if n_steps <= 0:
            return {
                "emulator_step_ms": 0.0,
                "reward_ms": 0.0,
                "preprocessing_ms": 0.0,
                "model_inference_ms": 0.0,
            }
        return {
            "emulator_step_ms": round(self.emulator_step_s / n_steps * 1000, 3),
            "reward_ms": round(self.reward_s / n_steps * 1000, 4),
            "preprocessing_ms": round(self.preprocessing_s / n_steps * 1000, 3),
            "model_inference_ms": round(self.model_inference_s / n_steps * 1000, 3),
        }


def format_benchmark_output(
    *,
    game_profile: str,
    num_envs: int,
    frame_skip: int,
    observation_mode: str,
    resize: list,
    total_timesteps: int,
    wall_clock_seconds: float,
    fps: float,
    component_timings: dict,
) -> dict:
    """Build the machine-readable JSON output dict.

    This function is separated for testability (Property 13).
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "game_profile": game_profile or "none",
        "num_envs": num_envs,
        "frame_skip": frame_skip,
        "observation_mode": observation_mode,
        "resize": list(resize) if resize else None,
        "total_timesteps": total_timesteps,
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        "fps": round(fps, 1),
        "component_timings": component_timings,
    }


def format_human_summary(json_output: dict) -> str:
    """Format a human-readable summary from the JSON output dict."""
    lines = [
        "=== Training Benchmark Results ===",
        "",
        f"  Game profile:      {json_output['game_profile']}",
        f"  Num envs:          {json_output['num_envs']}",
        f"  Frame skip:        {json_output['frame_skip']}",
        f"  Observation mode:  {json_output['observation_mode']}",
        f"  Resize:            {json_output['resize']}",
        f"  Total timesteps:   {json_output['total_timesteps']}",
        "",
        f"  Wall-clock time:   {json_output['wall_clock_seconds']:.3f} s",
        f"  Throughput:        {json_output['fps']:.1f} fps",
        "",
        "  Component timings (avg per step):",
    ]
    ct = json_output["component_timings"]
    lines.append(f"    Emulator step:     {ct['emulator_step_ms']:8.3f} ms")
    lines.append(f"    Reward:            {ct['reward_ms']:8.4f} ms")
    lines.append(f"    Preprocessing:     {ct['preprocessing_ms']:8.3f} ms")
    lines.append(f"    Model inference:   {ct['model_inference_ms']:8.3f} ms")
    return "\n".join(lines)


def _run_benchmark(config, profile):
    """Run the training benchmark and return the JSON output dict.

    Builds the full training pipeline (env + model), runs model.learn()
    while measuring wall-clock time, and estimates per-component timings
    from C++ frame timings when available.
    """
    import os
    import tempfile

    import torch

    from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
    from retro_ai.envs.base_env import BaseEnv
    from retro_ai.training.game_profile import StartupSequenceWrapper
    from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper

    # Force CPU-only inference
    device = "cpu"

    # Build environment
    def make_env(rank: int):
        def _init():
            env_config = {}
            if profile and hasattr(profile, "joystick_index"):
                env_config["joystick_index"] = profile.joystick_index
            if profile and profile.reward_params:
                env_config["reward_params"] = profile.reward_params

            base = BaseEnv(
                emulator_type=config.emulator_type,
                rom_path=config.rom_path,
                bios_path=config.bios_path,
                reward_mode=config.reward_mode,
                config=env_config or None,
            )
            pipeline_obj = PreprocessingPipeline(
                grayscale=config.grayscale,
                resize=config.resize,
                frame_stack=config.frame_stack,
                frame_skip=config.frame_skip,
            )
            preprocessed = PreprocessedEnv(base, pipeline_obj)
            env = GymnasiumWrapper(preprocessed)
            if profile and profile.startup_sequence:
                env = StartupSequenceWrapper(env, profile.startup_sequence)
            return env
        return _init

    num_envs = config.num_envs
    if num_envs == 1:
        env = make_env(0)()
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Build model
    from stable_baselines3 import PPO

    policy = config.policy
    if config.observation_mode == "ram" and policy != "MlpPolicy":
        policy = "MlpPolicy"

    kwargs = {
        "policy": policy,
        "env": env,
        "learning_rate": config.algorithm.learning_rate,
        "batch_size": config.algorithm.batch_size,
        "verbose": 0,
        "device": device,
    }
    # Set n_steps so PPO doesn't try to collect more steps than we're benchmarking.
    # PPO needs n_steps * num_envs steps per rollout before an update.
    if num_envs > 1:
        base_n_steps = 2048
        kwargs["n_steps"] = max(1, base_n_steps // num_envs)
    else:
        # For benchmarks, cap n_steps to total_timesteps so small runs don't hang
        kwargs["n_steps"] = min(2048, config.total_timesteps)
    # batch_size can't exceed n_steps * num_envs
    effective_buffer = kwargs["n_steps"] * num_envs
    kwargs["batch_size"] = min(config.algorithm.batch_size, effective_buffer)

    model = PPO(**kwargs)

    # --- Measure training ---
    timings = _TimingTracker()

    # We measure the overall wall-clock for model.learn().
    # For component breakdown, we sample C++ frame timings from the
    # environment interface when available (single-env only).
    interface = None
    if num_envs == 1:
        # Walk the wrapper chain to find the BaseEnv's _interface
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        if hasattr(inner, "_interface"):
            interface = inner._interface

    # Collect C++ frame timings via a sampling approach:
    # Before and after learn(), we can't easily intercept each step.
    # Instead, we measure total wall-clock and estimate component
    # breakdown from the C++ per-frame timings sampled before the run.
    sample_count = min(100, config.total_timesteps // 2)
    if interface is not None:
        t0 = time.perf_counter()
        for _ in range(sample_count):
            interface.step_numpy([0])
            ft = interface.get_last_frame_timings()
            timings.emulator_step_s += (ft.cpu_us + ft.vdc_us) / 1e6
            timings.reward_s += ft.reward_us / 1e6
            timings.preprocessing_s += ft.framebuffer_us / 1e6
        sample_wall = time.perf_counter() - t0
        # Reset after sampling
        interface.reset_numpy(-1)

    # Run training — use a simple callback to print progress so the monitor
    # doesn't think we've hung.
    from stable_baselines3.common.callbacks import BaseCallback

    class _ProgressCallback(BaseCallback):
        def __init__(self, total, interval=1000):
            super().__init__()
            self._total = total
            self._interval = interval
        def _on_step(self) -> bool:
            if self.num_timesteps % self._interval == 0 or self.num_timesteps >= self._total:
                elapsed = time.perf_counter() - wall_start
                fps_so_far = self.num_timesteps / elapsed if elapsed > 0 else 0
                print(f"  [{self.num_timesteps}/{self._total}] {fps_so_far:.1f} fps", flush=True)
            return True

    progress_interval = max(500, config.total_timesteps // 20)
    wall_start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=_ProgressCallback(config.total_timesteps, progress_interval),
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
    wall_elapsed = time.perf_counter() - wall_start

    # Compute fps
    fps = config.total_timesteps / wall_elapsed if wall_elapsed > 0 else 0.0

    # Component timings: use sampled C++ timings scaled to actual timesteps,
    # and estimate model inference as the remainder.
    if interface is not None and sample_count > 0:
        scale = config.total_timesteps / sample_count
        timings.emulator_step_s *= scale
        timings.reward_s *= scale
        timings.preprocessing_s *= scale
        # Model inference = total wall time minus emulator components
        emulator_total = timings.emulator_step_s + timings.reward_s + timings.preprocessing_s
        timings.model_inference_s = max(0.0, wall_elapsed - emulator_total)

    component_timings = timings.to_ms_dict(config.total_timesteps)

    # Cleanup
    try:
        env.close()
    except Exception:
        pass

    return format_benchmark_output(
        game_profile=config.game_profile,
        num_envs=config.num_envs,
        frame_skip=config.frame_skip,
        observation_mode=config.observation_mode,
        resize=list(config.resize) if config.resize else None,
        total_timesteps=config.total_timesteps,
        wall_clock_seconds=wall_elapsed,
        fps=fps,
        component_timings=component_timings,
    )


def main():
    args = _parse_args()

    # Validate parameters early
    if args.frame_skip < 1 or args.frame_skip > 16:
        sys.exit(f"Error: --frame-skip must be in [1, 16], got {args.frame_skip}")
    if args.num_envs < 1:
        sys.exit(f"Error: --num-envs must be >= 1, got {args.num_envs}")
    if args.timesteps <= 0:
        sys.exit(f"Error: --timesteps must be > 0, got {args.timesteps}")

    config = _build_config(args)

    try:
        config, profile = _resolve_profile(config)
    except Exception as e:
        sys.exit(f"Error loading game profile: {e}")

    if config.emulator_type is None:
        sys.exit(
            "Error: no emulator_type configured. "
            "Provide --game-profile or set emulator_type in config."
        )

    print(
        f"Benchmarking {args.timesteps} timesteps "
        f"(num_envs={args.num_envs}, frame_skip={args.frame_skip}, "
        f"obs={args.observation_mode}, resize={args.resize})...\n",
        file=sys.stderr,
    )

    json_output = _run_benchmark(config, profile)

    # Human-readable summary
    print(format_human_summary(json_output))

    # JSON output
    print(f"\n--- JSON ---\n{json.dumps(json_output, indent=2)}")


if __name__ == "__main__":
    main()
