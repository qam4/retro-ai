#!/usr/bin/env python3
"""Experiment 002: Systematic Agent Investigation runner.

Each phase is self-contained and writes results to output/exp002/<phase>/.

Usage:
    python scripts/exp002_runner.py phase0          # random baseline
    python scripts/exp002_runner.py phase1          # vanilla PPO
    python scripts/exp002_runner.py phase1 --smoke  # smoke test only (500 steps)
    python scripts/exp002_runner.py phase1 --eval   # eval only (load existing model)
    python scripts/exp002_runner.py phase2          # PPO + RND
    python scripts/exp002_runner.py phase3          # PPO + DrQ + sticky
    python scripts/exp002_runner.py phase4          # tuned PPO
    python scripts/exp002_runner.py phase5          # DQN + PER
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

GAME_PROFILE = "satellite_attack_memory"
OUTPUT_BASE = "output/exp002"
TIMESTEPS = 100_000
EVAL_EPISODES = 3
EVAL_SEED = 42


def phase0_random_baseline(args):
    """Run 10 episodes with random actions to establish the floor."""
    from retro_ai import BaseEnv
    from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
    from retro_ai.training.game_profile import GameProfileRegistry
    from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper

    out_dir = os.path.join(OUTPUT_BASE, "phase0_random")
    os.makedirs(out_dir, exist_ok=True)

    registry = GameProfileRegistry()
    profile = registry.load(GAME_PROFILE)

    config_dict = {}
    if hasattr(profile, "joystick_index"):
        config_dict["joystick_index"] = profile.joystick_index
    if profile.reward_params:
        config_dict["reward_params"] = profile.reward_params

    base = BaseEnv(
        emulator_type=profile.emulator_type,
        rom_path=profile.rom_path,
        bios_path=profile.bios_path,
        reward_mode="memory",
        config=config_dict or None,
        action_mode="joystick",
    )
    pipeline = PreprocessingPipeline(
        grayscale=True, resize=(84, 84), frame_stack=4, frame_skip=4,
    )
    preprocessed = PreprocessedEnv(base, pipeline)
    env = GymnasiumWrapper(preprocessed)

    from retro_ai.training.game_profile import StartupSequenceWrapper
    if profile.startup_sequence:
        env = StartupSequenceWrapper(env, profile.startup_sequence)

    num_episodes = 10
    results = []
    action_counts = {}

    for ep in range(num_episodes):
        obs, info = env.reset(seed=EVAL_SEED + ep)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            action_key = str(action.tolist() if hasattr(action, "tolist") else action)
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward
            steps += 1
        results.append({"episode": ep, "reward": total_reward, "length": steps})
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, length={steps}")

    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    summary = {
        "phase": "phase0_random_baseline",
        "num_episodes": num_episodes,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "top_actions": sorted(action_counts.items(), key=lambda x: -x[1])[:10],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Phase 0: Random Baseline ===")
    print(f"  Mean reward: {summary['reward_mean']:.1f} ± {summary['reward_std']:.1f}")
    print(f"  Best reward: {summary['reward_max']:.1f}")
    print(f"  Mean length: {summary['length_mean']:.0f}")
    print(f"  Saved to {out_dir}/results.json")


def _make_training_config(phase_name, extra_config=None):
    """Build a TrainingConfig for a given phase."""
    from retro_ai.training.config import AlgorithmConfig, TrainingConfig

    base = dict(
        algorithm=AlgorithmConfig(name="PPO", learning_rate=3e-4, batch_size=64),
        total_timesteps=TIMESTEPS,
        game_profile=GAME_PROFILE,
        reward_mode="memory",
        action_mode="joystick",
        policy="CnnPolicy",
        num_envs=4,
        grayscale=True,
        resize=(84, 84),
        frame_stack=4,
        frame_skip=4,
        device="auto",
        mixed_precision=True,
        vec_env_type="threaded",
        output_dir=os.path.join(OUTPUT_BASE, phase_name),
        tensorboard=False,
        checkpoint_interval=25000,
        max_checkpoints=4,
        log_interval=1000,
        # Disabled by default — phases enable selectively
        sticky_actions=0.0,
        reward_clip=0.0,
        survival_bonus=0.0,
    )
    if extra_config:
        base.update(extra_config)
    return TrainingConfig(**base)


def _run_training(config, smoke_only=False):
    """Run training and return the output dir."""
    from retro_ai.training.pipeline import TrainingPipeline

    if smoke_only:
        from dataclasses import replace
        config = replace(config, total_timesteps=500, checkpoint_interval=1000)

    pipeline = TrainingPipeline(config)
    model_path = pipeline.run()
    print(f"  Model saved to {model_path}")
    return model_path


def _run_eval(model_path, phase_name):
    """Run deterministic eval and return summary."""
    from retro_ai.training.evaluation import EvaluationModule
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profile = registry.load(GAME_PROFILE)
    out_dir = os.path.join(OUTPUT_BASE, phase_name, "eval")

    evaluator = EvaluationModule(
        model_path=str(model_path),
        game_profile=profile,
        num_episodes=EVAL_EPISODES,
        base_seed=EVAL_SEED,
        output_dir=out_dir,
        action_mode="joystick",
    )
    summary = evaluator.run()
    print(f"\n=== {phase_name} Eval ===")
    print(f"  Mean reward: {summary['reward_mean']:.1f} ± {summary['reward_std']:.1f}")
    print(f"  Best reward: {summary['reward_max']:.1f}")
    print(f"  Mean length: {summary['length_mean']:.0f}")
    return summary


def phase1_vanilla_ppo(args):
    """Phase 1: Vanilla PPO, no extras."""
    config = _make_training_config("phase1_vanilla_ppo")
    if args.smoke:
        print("=== Phase 1: Smoke Test (500 steps) ===")
        _run_training(config, smoke_only=True)
        return
    if args.eval:
        model_path = os.path.join(OUTPUT_BASE, "phase1_vanilla_ppo", "final_model.zip")
        _run_eval(model_path, "phase1_vanilla_ppo")
        return
    print("=== Phase 1: Vanilla PPO (100k steps) ===")
    model_path = _run_training(config)
    _run_eval(model_path, "phase1_vanilla_ppo")


def phase2_ppo_rnd(args):
    """Phase 2: PPO + RND with low coefficient."""
    from retro_ai.training.config import IntrinsicRewardConfig
    config = _make_training_config("phase2_ppo_rnd", {
        "intrinsic_reward": IntrinsicRewardConfig(enabled=True, coefficient=0.1),
    })
    if args.smoke:
        print("=== Phase 2: Smoke Test (500 steps) ===")
        _run_training(config, smoke_only=True)
        return
    if args.eval:
        model_path = os.path.join(OUTPUT_BASE, "phase2_ppo_rnd", "final_model.zip")
        _run_eval(model_path, "phase2_ppo_rnd")
        return
    print("=== Phase 2: PPO + RND coeff=0.1 (100k steps) ===")
    model_path = _run_training(config)
    _run_eval(model_path, "phase2_ppo_rnd")


def phase3_ppo_drq_sticky(args):
    """Phase 3: Best of 1/2 + DrQ + sticky actions."""
    config = _make_training_config("phase3_ppo_drq_sticky", {
        "augmentation": True,
        "sticky_actions": 0.25,
    })
    if args.smoke:
        print("=== Phase 3: Smoke Test (500 steps) ===")
        _run_training(config, smoke_only=True)
        return
    if args.eval:
        model_path = os.path.join(OUTPUT_BASE, "phase3_ppo_drq_sticky", "final_model.zip")
        _run_eval(model_path, "phase3_ppo_drq_sticky")
        return
    print("=== Phase 3: PPO + DrQ + Sticky (100k steps) ===")
    model_path = _run_training(config)
    _run_eval(model_path, "phase3_ppo_drq_sticky")


def phase4_tuned_ppo(args):
    """Phase 4: Tuned PPO hyperparameters."""
    from retro_ai.training.config import AlgorithmConfig
    config = _make_training_config("phase4_tuned_ppo", {
        "algorithm": AlgorithmConfig(
            name="PPO", learning_rate=0.001, batch_size=64,
            extra={"ent_coef": 0.05, "n_steps": 512, "n_epochs": 4, "clip_range": 0.2},
        ),
    })
    if args.smoke:
        print("=== Phase 4: Smoke Test (500 steps) ===")
        _run_training(config, smoke_only=True)
        return
    if args.eval:
        model_path = os.path.join(OUTPUT_BASE, "phase4_tuned_ppo", "final_model.zip")
        _run_eval(model_path, "phase4_tuned_ppo")
        return
    print("=== Phase 4: Tuned PPO (100k steps) ===")
    model_path = _run_training(config)
    _run_eval(model_path, "phase4_tuned_ppo")


def phase5_dqn_per(args):
    """Phase 5: DQN + PER."""
    from retro_ai.training.config import AlgorithmConfig
    config = _make_training_config("phase5_dqn_per", {
        "algorithm": AlgorithmConfig(
            name="DQN", learning_rate=1e-4, batch_size=32,
            extra={
                "buffer_size": 50000, "learning_starts": 1000,
                "exploration_fraction": 0.3, "exploration_final_eps": 0.05,
                "target_update_interval": 500, "train_freq": 4,
            },
        ),
        "action_mode": "discrete",
        "num_envs": 1,
    })
    if args.smoke:
        print("=== Phase 5: Smoke Test (500 steps) ===")
        _run_training(config, smoke_only=True)
        return
    if args.eval:
        model_path = os.path.join(OUTPUT_BASE, "phase5_dqn_per", "final_model.zip")
        _run_eval(model_path, "phase5_dqn_per")
        return
    print("=== Phase 5: DQN + PER (100k steps) ===")
    model_path = _run_training(config)
    _run_eval(model_path, "phase5_dqn_per")


PHASES = {
    "phase0": phase0_random_baseline,
    "phase1": phase1_vanilla_ppo,
    "phase2": phase2_ppo_rnd,
    "phase3": phase3_ppo_drq_sticky,
    "phase4": phase4_tuned_ppo,
    "phase5": phase5_dqn_per,
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 002 runner")
    parser.add_argument("phase", choices=list(PHASES.keys()), help="Phase to run")
    parser.add_argument("--smoke", action="store_true", help="Smoke test only (500 steps)")
    parser.add_argument("--eval", action="store_true", help="Eval only (load existing model)")
    args = parser.parse_args()
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
