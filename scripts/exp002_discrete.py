#!/usr/bin/env python3
"""Experiment 002 — Phase 1b: Vanilla PPO with discrete action space."""
import os
from retro_ai.training.config import AlgorithmConfig, TrainingConfig
from retro_ai.training.pipeline import TrainingPipeline
from retro_ai.training.evaluation import EvaluationModule
from retro_ai.training.game_profile import GameProfileRegistry

config = TrainingConfig(
    algorithm=AlgorithmConfig(name="PPO", learning_rate=3e-4, batch_size=64),
    total_timesteps=100_000,
    game_profile="satellite_attack_memory",
    reward_mode="memory",
    action_mode="discrete",  # <-- the key change
    policy="CnnPolicy",
    num_envs=4,
    grayscale=True,
    resize=(84, 84),
    frame_stack=4,
    frame_skip=4,
    device="auto",
    mixed_precision=True,
    vec_env_type="threaded",
    output_dir="output/exp002/phase1b_discrete",
    tensorboard=False,
    checkpoint_interval=25000,
    log_interval=1000,
)

print("=== Phase 1b: Vanilla PPO + Discrete Actions (100k steps) ===")
pipeline = TrainingPipeline(config)
model_path = pipeline.run()
print(f"  Model saved to {model_path}")

# Eval
registry = GameProfileRegistry()
profile = registry.load("satellite_attack_memory")
evaluator = EvaluationModule(
    model_path=str(model_path),
    game_profile=profile,
    num_episodes=3,
    base_seed=42,
    output_dir="output/exp002/phase1b_discrete/eval",
    video_path="output/exp002/phase1b_discrete/eval/replay.mp4",
    action_mode="discrete",
)
summary = evaluator.run()
print(f"\n=== Phase 1b Eval ===")
print(f"  Mean reward: {summary['reward_mean']:.1f} +/- {summary['reward_std']:.1f}")
print(f"  Best reward: {summary['reward_max']:.1f}")
print(f"  Mean length: {summary['length_mean']:.0f}")
