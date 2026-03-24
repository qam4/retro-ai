#!/usr/bin/env python3
"""Train a PPO agent on Yeti (Thomson MO5).

Usage:
    python scripts/train_yeti.py              # 100k steps
    python scripts/train_yeti.py --timesteps 500000
    python scripts/train_yeti.py --eval       # eval only
"""

import argparse
import os
import sys
import numpy as np

from retro_ai.envs.base_env import BaseEnv
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
from retro_ai.wrappers.survival_bonus import SurvivalBonusWrapper
from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

OUTPUT_DIR = "output/yeti"
ROM_DIR = os.environ.get("RETRO_AI_ROM_DIR", "roms")

REWARD_PARAMS = {
    "basic_rom_path": f"{ROM_DIR}/mo5/roms/basic5.rom",
    "monitor_rom_path": f"{ROM_DIR}/mo5/roms/mo5.rom",
    "startup_sequence": 'wait:100|type:LOAD""\\n|wait:200|type:RUN\\n|wait:800|type:1|wait:300',
    "lives_addr": "11095",
    "score_address_count": "1",
    "score_address_0_addr": "11093",
    "score_address_0_bytes": "2",
    "score_address_0_bcd": "0",
    "score_address_0_le": "0",
    "score_address_0_multiplier": "1",
}


def make_env():
    base = BaseEnv(
        emulator_type="mo5",
        rom_path=f"{ROM_DIR}/mo5/roms/Yeti (1984) (Loriciels).k7",
        reward_mode="memory",
        config={"reward_params": REWARD_PARAMS},
        action_mode="joystick",
    )
    pipeline = PreprocessingPipeline(
        grayscale=True, resize=(84, 84), frame_stack=4, frame_skip=4,
    )
    preprocessed = PreprocessedEnv(base, pipeline, frame_maxpool=True)
    env = GymnasiumWrapper(preprocessed)
    env = SurvivalBonusWrapper(env, bonus=0.01)
    env = Monitor(env)
    return env


def train(timesteps, num_envs=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vec_env = ThreadedVecEnv([make_env for _ in range(num_envs)])

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        device="auto",
    )

    print(f"=== Training Yeti PPO ({timesteps:,} steps, {num_envs} envs) ===")
    model.learn(total_timesteps=timesteps)
    model_path = os.path.join(OUTPUT_DIR, "final_model.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    vec_env.close()
    return model_path


def evaluate(model_path, episodes=5):
    env = make_env()
    model = PPO.load(model_path, env=env)

    results = []
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward
            steps += 1
        results.append(total_reward)
        print(f"  Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    print(f"Mean: {np.mean(results):.1f} ± {np.std(results):.1f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Yeti agent")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.eval:
        model_path = os.path.join(OUTPUT_DIR, "final_model.zip")
        evaluate(model_path)
    else:
        model_path = train(args.timesteps, args.envs)
        evaluate(model_path)


if __name__ == "__main__":
    main()
