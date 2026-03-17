#!/usr/bin/env python3
"""Experiment 002 — SimPLe phase: 100k real steps with world model."""
import os
from retro_ai.training.config import TrainingConfigParser
from retro_ai.training.simple import SimplePipeline
from retro_ai.training.evaluation import EvaluationModule
from retro_ai.training.game_profile import GameProfileRegistry
from dataclasses import replace

config = TrainingConfigParser.from_yaml("game_profiles/satellite_attack_simple.yaml")
config = replace(
    config,
    output_dir="output/exp002/phase6_simple",
    num_envs=1,  # SimplePipeline collects data sequentially
    device="auto",
    mixed_precision=True,
    # Disable RND — we want clean game score signal
    intrinsic_reward=replace(config.intrinsic_reward, enabled=False),
    augmentation=False,
    sticky_actions=0.0,
)

print("=== Phase 6: SimPLe (100k real steps, 15 rounds) ===")
print(f"  World model: {config.simple.num_rounds} rounds, "
      f"{config.simple.world_model_epochs} epochs, "
      f"synthetic_ratio={config.simple.synthetic_ratio}")

pipeline = SimplePipeline(config)
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
    output_dir="output/exp002/phase6_simple/eval",
    action_mode="joystick",
)
summary = evaluator.run()
print(f"\n=== Phase 6 Eval ===")
print(f"  Mean reward: {summary['reward_mean']:.1f} ± {summary['reward_std']:.1f}")
print(f"  Best reward: {summary['reward_max']:.1f}")
print(f"  Mean length: {summary['length_mean']:.0f}")
