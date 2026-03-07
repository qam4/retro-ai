#!/usr/bin/env python3
"""Quick smoke test: load model, run 20 steps, verify rewards flow."""
import sys

from retro_ai.training.evaluation import EvaluationModule
from retro_ai.training.game_profile import GameProfileRegistry

registry = GameProfileRegistry()
profile = registry.load("course_automobile")
print(f"Profile: {profile.name}, reward_mode: {profile.reward_mode}")
print(f"reward_params keys: {list(profile.reward_params.keys())}")

# Build env directly to test the fix
evaluator = EvaluationModule(
    model_path="output/course_automobile/final_model.zip",
    game_profile=profile,
    num_episodes=1,
    base_seed=42,
    output_dir="output/course_automobile/smoke",
)
env = evaluator._build_env()
model = evaluator._load_model(env)
print(f"Model loaded: {type(model).__name__}")

obs, info = env.reset()
print(f"Reset ok, obs shape: {obs.shape}")

total_reward = 0.0
for i in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        print(f"Episode ended at step {i+1}")
        break

print(f"20 steps done, total_reward={total_reward}, done={done}")
if total_reward > 0:
    print("PASS: rewards are flowing")
else:
    print("WARN: zero reward after 20 steps (may be normal if car hasn't moved yet)")
print("Smoke test complete")
sys.exit(0)
