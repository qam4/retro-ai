# Getting Started

## Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+, or MinGW)
- CMake 3.20+
- Python 3.10+ with NumPy
- pybind11

## Building from Source

```bash
# Clone with emulator submodules
git clone --recursive https://github.com/qam4/retro-ai.git
cd retro-ai

# Configure and build (Linux/macOS)
cmake --preset ci-linux    # or ci-macos
cmake --build --preset ci-linux

# Configure and build (Windows with MinGW)
cmake --preset ci-mingw
cmake --build --preset ci-mingw

# Configure and build (Windows with MSVC)
cmake --preset ci-windows
cmake --build --preset ci-windows
```

## Setting Up Python

Set `PYTHONPATH` so Python can find both the pure-Python package and the
compiled native module:

```bash
export PYTHONPATH=/path/to/retro-ai/python:/path/to/retro-ai/build/<preset>
```

On Windows with MinGW, use the helper script:

```bash
source setup_env.sh
```

## Quick Test

```python
from retro_ai import BaseEnv

env = BaseEnv(
    emulator_type="videopac",
    rom_path="roms/satellite_attack.bin",
    bios_path="roms/videopac.bin",
    reward_mode="survival",
)

obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")  # (200, 160, 3)

obs, reward, done, truncated, info = env.step(0)
print(f"Reward: {reward}, Done: {done}")
```

## Next Steps

- See `examples/` for training scripts
- Read [reward_systems.md](reward_systems.md) for reward mode details
- Read [adding_emulators.md](adding_emulators.md) to integrate a new emulator
