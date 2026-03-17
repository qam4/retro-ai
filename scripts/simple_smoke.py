#!/usr/bin/env python3
"""Quick SimPLe smoke test — 500 steps, 2 rounds."""
from retro_ai.training.config import TrainingConfigParser, SimpleConfig
from retro_ai.training.simple import SimplePipeline
from dataclasses import replace

config = TrainingConfigParser.from_yaml("game_profiles/satellite_attack_simple.yaml")
config = replace(
    config,
    total_timesteps=500,
    simple=SimpleConfig(
        enabled=True, num_rounds=2, world_model_epochs=5,
        world_model_lr=0.001, world_model_batch_size=32,
        synthetic_ratio=2, rollout_horizon=10, quality_threshold=0.1,
    ),
    output_dir="output/exp002/simple_smoke",
    num_envs=1,
)
pipeline = SimplePipeline(config)
path = pipeline.run()
print(f"SimPLe smoke test complete: {path}")
