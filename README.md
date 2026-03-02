# Retro-AI

> Reinforcement learning framework for retro game emulators

[![CI](https://github.com/qam4/retro-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/qam4/retro-ai/actions/workflows/ci.yml)

Train AI agents to play classic games using modern deep learning techniques.
Retro-AI provides a unified C++/Python interface for multiple emulators with
pluggable reward systems and optional Gymnasium integration.

## Features

- 🎮 Multi-emulator support — Videopac, MO5, and more
- 🔌 Framework agnostic — use with or without Gymnasium / Stable-Baselines3
- 🎯 Pluggable rewards — memory, vision, intrinsic motivation, or custom
- ⚡ High performance — C++ core with zero-copy NumPy observations
- 🔧 Extensible — easy to add new emulators and reward strategies
- 🎲 Deterministic — reproducible results via seeded resets

## Quick Start

```bash
git clone --recursive https://github.com/qam4/retro-ai.git
cd retro-ai
cmake --preset ci-linux      # or ci-macos / ci-windows / ci-mingw
cmake --build --preset ci-linux
export PYTHONPATH=$PWD/python:$PWD/build/ci-linux
```

```python
from retro_ai import BaseEnv

env = BaseEnv(
    emulator_type="videopac",
    rom_path="roms/satellite_attack.bin",
    bios_path="roms/videopac.bin",
    reward_mode="survival",
)

obs, info = env.reset(seed=42)
for _ in range(1000):
    obs, reward, done, truncated, info = env.step(0)
    if done:
        obs, info = env.reset()
```

### With Gymnasium + Stable-Baselines3

```bash
pip install gymnasium stable-baselines3
```

```python
from retro_ai import BaseEnv
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
from stable_baselines3 import PPO

env = GymnasiumWrapper(BaseEnv(
    emulator_type="videopac",
    rom_path="roms/satellite_attack.bin",
    bios_path="roms/videopac.bin",
))

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

## Reward Modes

| Mode | Description |
|------|-------------|
| `survival` | +1 per frame alive, −10 on game over |
| `memory` | Score delta read from emulator RAM |
| `vision` | OCR score from screen pixels |
| `intrinsic` | Curiosity-driven novelty reward |
| `custom` | User-defined callback |

Switch at runtime: `env.set_reward_mode("memory")`

## Documentation

- [Getting Started](docs/getting_started.md)
- [Reward Systems](docs/reward_systems.md)
- [Adding Emulators](docs/adding_emulators.md)
- [API Reference](docs/api_reference.md)

## Examples

See [`examples/`](examples/) for runnable scripts:
- `basic_training.py` — random agent loop
- `gymnasium_integration.py` — PPO with Stable-Baselines3
- `custom_rewards.py` — compare reward modes
- `multi_emulator.py` — run Videopac and MO5 side by side

## Development

```bash
# Build (MinGW example)
cmake --preset dev-mingw
cmake --build --preset dev-mingw

# Format & lint Python
cmake --build --preset dev-mingw --target format-python
cmake --build --preset dev-mingw --target lint-python

# Run tests
ctest --preset dev-mingw
python -m pytest tests/python/ -v
```

## License

MIT — see [LICENSE](LICENSE)
