# Retro-AI

> Reinforcement learning framework for retro game emulators

Train AI agents to play classic games using modern deep learning techniques. Retro-AI provides a unified interface for multiple emulators with pluggable reward systems and optional integration with popular RL frameworks.

## Features

- 🎮 **Multi-emulator support** - Videopac, MO5, and more
- 🔌 **Framework agnostic** - Use with or without Gymnasium/Stable-Baselines3
- 🎯 **Pluggable rewards** - Memory, vision, intrinsic motivation, or custom
- ⚡ **High performance** - Fast C++ core, >1000 FPS in headless mode
- 🔧 **Extensible** - Easy to add new emulators and reward strategies
- 🎲 **Deterministic** - Reproducible results for research

## Quick Start

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/qam4/retro-ai.git
cd retro-ai

# Install
pip install .

# Optional: Install with Gymnasium support
pip install .[gymnasium,training]
```

### Basic Usage

```python
from retro_ai.envs import VideopacEnv

# Create environment
env = VideopacEnv(
    bios_path="roms/videopac.bin",
    rom_path="roms/satellite_attack.bin",
    reward_mode="survival"
)

# Training loop
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Your policy here
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs = env.reset()
```

### With Stable-Baselines3

```python
from retro_ai.envs import VideopacEnv
from retro_ai.wrappers import GymnasiumWrapper
from stable_baselines3 import PPO

# Create and wrap environment
env = GymnasiumWrapper(VideopacEnv(
    bios_path="roms/videopac.bin",
    rom_path="roms/satellite_attack.bin"
))

# Train agent
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("my_agent")
```

## Supported Emulators

| Emulator | System | Status |
|----------|--------|--------|
| Videopac | Philips Videopac/Odyssey2 | ✅ Supported |
| MO5 | Thomson MO5 | 🚧 In Progress |

## Reward Modes

- **survival** - Simple time-based reward (+1 per frame, -10 on death)
- **memory** - Read score directly from RAM (fast, accurate)
- **vision** - OCR score from screen (slower, game-agnostic)
- **intrinsic** - Curiosity-driven exploration (no game knowledge needed)
- **custom** - Define your own reward function

## Documentation

- [Getting Started](docs/getting_started.md)
- [Adding Emulators](docs/adding_emulators.md)
- [Reward Systems](docs/reward_systems.md)
- [API Reference](docs/api_reference.md)
- [Full Specification](RETRO_AI_SPEC.md)

## Examples

See the [examples/](examples/) directory for:
- Basic training loop
- Custom reward functions
- Gymnasium integration
- Multi-emulator training
- Distributed training

## Architecture

```
Training Framework (Optional)
    ↓
Python Environment Layer (Framework-agnostic)
    ↓
C++ RL Interface (Generic)
    ↓
Emulator Cores (Submodules)
```

The layered design allows you to:
- Use any RL framework (or none)
- Swap frameworks without changing emulator code
- Add new emulators easily
- Customize reward functions per-game

## Development

### Building from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/qam4/retro-ai.git
cd retro-ai

# Build
mkdir build && cd build
cmake ..
make -j8

# Install Python package in development mode
cd ..
pip install -e .[dev]
```

### Running Tests

```bash
# C++ tests
cd build
ctest

# Python tests
pytest tests/
```

### Adding a New Emulator

1. Add emulator as git submodule: `git submodule add <url> emulators/<name>`
2. Implement `RLInterface` in `src/<name>_rl.cpp`
3. Add Python wrapper in `python/retro_ai/envs/<name>.py`
4. Update CMakeLists.txt and bindings
5. Add tests and examples

See [docs/adding_emulators.md](docs/adding_emulators.md) for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use Retro-AI in your research, please cite:

```bibtex
@software{retro_ai,
  title = {Retro-AI: Reinforcement Learning for Retro Game Emulators},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/qam4/retro-ai}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Inspired by OpenAI Gym Retro and Atari Learning Environment
- Built on top of excellent emulator projects
- Thanks to the RL and emulation communities

## Related Projects

- [Gymnasium](https://gymnasium.farama.org/) - Standard RL environment API
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gym Retro](https://github.com/openai/retro) - OpenAI's retro game RL platform
- [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
