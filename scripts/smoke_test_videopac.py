#!/usr/bin/env python3
"""Smoke test: load Satellite Attack on Videopac and run a short episode."""

import time
import numpy as np
from retro_ai.envs.base_env import BaseEnv

BIOS = r"C:\src\videopac\roms\Philips C52 BIOS (19xx)(Philips)(FR).bin"
ROM = r"C:\src\videopac\roms\Satellite Attack (1981)(Philips)(EU).bin"

env = BaseEnv(emulator_type="videopac", rom_path=ROM, bios_path=BIOS)

print("Observation space:", env.get_observation_space())
print("Action space:     ", env.get_action_space())
print("Reward modes:     ", env.available_reward_modes())

obs, info = env.reset(seed=42)
print(f"\nAfter reset: obs shape={obs.shape}, dtype={obs.dtype}, info={info}")
print(f"  obs min={obs.min()}, max={obs.max()}, mean={obs.mean():.1f}")

# Run 1000 steps with random actions
num_steps = 1000
action_n = env.get_action_space()["shape"][0]
total_reward = 0.0
t0 = time.perf_counter()

for i in range(num_steps):
    action = np.random.randint(0, action_n)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done:
        print(f"  Episode ended at step {i+1}, total_reward={total_reward:.1f}")
        obs, info = env.reset()
        total_reward = 0.0

elapsed = time.perf_counter() - t0
fps = num_steps / elapsed
print(f"\n{num_steps} steps in {elapsed:.3f}s = {fps:.0f} FPS")
print(f"Final obs: shape={obs.shape}, min={obs.min()}, max={obs.max()}")

# Test save/load state
state = env.save_state()
print(f"\nSaved state: {len(state)} bytes")
env.load_state(state)
print("State restored successfully")

print("\nSmoke test passed!")
