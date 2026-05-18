#!/usr/bin/env python3
"""Use the trained policy to get near princess, then force RIGHT
actions to walk into her. Confirms (or denies) the princess-touch
detection rule.

The trained policy reliably gets the agent onto floor 5 within ~76
pixels of princess (we observed this in approach 24). The remaining
gap is dodging one or two snowballs. This script lets the policy
handle navigation up through floor 5, then takes manual control
once the agent's pixel-y is in floor-5 standing range, walks RIGHT
until either we touch the princess or the agent dies. Per-frame RAM
trace is dumped."""
from __future__ import annotations

import pickle

import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

cfg = EnvConfig(
    profile="yeti_fruit",
    action_mode="joystick",
    max_steps=2000,
    stall_threshold=2000,
    resize=(84, 84),
)
stack = build_training_env("yeti_fruit", cfg)
base = stack.base
gym_env = stack.gym
base.reset(seed=0)

with open("output/mo5/yeti/seeds/v9_v4_cp4enriched.pkl", "rb") as f:
    seeds = pickle.load(f)["checkpoints"][4]
seed = seeds[0]
model = PPO.load("output/mo5/yeti/training/segment_4toP_v1/final_model.zip")

base._interface.load_state(seed)
obs = None
for _ in range(5):
    obs, _, _, _, _ = gym_env.step([0, 0, 0])

iface = base._interface


def st():
    return {
        "x": iface.read_ram_byte(11090),
        "y": iface.read_ram_byte(11089),
        "fr": iface.read_ram_byte(11055),
        "lv": iface.read_ram_byte(11095),
        "bonus": (iface.read_ram_byte(11010) << 8) | iface.read_ram_byte(11011),
        "score": (iface.read_ram_byte(11093) << 8) | iface.read_ram_byte(11094),
        "flag": iface.read_ram_byte(11050),
    }


print("Letting trained policy navigate up to floor 5...")
print(
    f"{'step':>4} {'x':>3} {'y':>3} {'fr':>2} {'lv':>2} "
    f"{'bonus':>5} {'score':>5} {'flag':>4} {'phase':>8}"
)

prev = st()
phase = "policy"
forced_steps = 0
MAX_STEPS = 400
for step in range(1, MAX_STEPS + 1):
    if phase == "policy":
        # Use trained policy.
        obs_chw = np.transpose(obs, (2, 0, 1))
        action, _ = model.predict(obs_chw, deterministic=False)
        # If we're near floor 5 (y in 48..64) AND ram_x >= 50, switch
        # to forced RIGHT.
        if 44 <= prev["y"] <= 60 and prev["x"] >= 55:
            phase = "forced"
            print(f"  --> switching to forced RIGHT at step {step}")
    if phase == "forced":
        # Cycle: jump-right, jump-right, right, right, right.
        # Roughly mimics holding right while occasionally jumping.
        action = [0, 1, 1] if forced_steps % 5 < 2 else [0, 1, 0]
        forced_steps += 1
    obs, _, done, trunc, _ = gym_env.step(action)
    cur = st()
    if cur != prev or step % 20 == 0 or done or trunc:
        print(
            f"{step:>4} {cur['x']:>3} {cur['y']:>3} {cur['fr']:>2} "
            f"{cur['lv']:>2} {cur['bonus']:>5} {cur['score']:>5} "
            f"{cur['flag']:>4} {phase:>8}"
        )
    # Princess touch: rising edge of byte 11050 (level-cleared flag).
    if cur["flag"] == 1 and prev["flag"] == 0:
        print(f"\n*** PRINCESS TOUCH DETECTED at step {step} ***")
        print(
            f"    flag rising edge: {prev['flag']} -> {cur['flag']}, "
            f"score {prev['score']} -> {cur['score']}, "
            f"bonus {prev['bonus']} -> {cur['bonus']}"
        )
        break
    if done or trunc:
        print(f"\n  episode terminated: done={done} trunc={trunc}")
        break
    prev = cur

print(f"\nForced steps: {forced_steps}")
