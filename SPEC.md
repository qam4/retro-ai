# Retro-AI: Requirements Specification

## Overview

Retro-AI is a reinforcement learning framework for training AI agents on retro game emulators. It provides a unified, framework-agnostic interface for multiple emulators (Videopac, MO5, etc.) with pluggable reward systems and optional integration with popular RL libraries.

## Goals

1. **Multi-emulator support** - Single interface for multiple retro emulators
2. **Framework agnostic** - Core functionality independent of Gymnasium/Stable-Baselines3
3. **Pluggable rewards** - Support memory-based, vision-based, and intrinsic motivation
4. **High performance** - Fast C++ core with minimal Python overhead
5. **Extensible** - Easy to add new emulators and reward strategies

## Architecture

```
┌─────────────────────────────────────────┐
│  Training Framework (Optional)          │
│  - Gymnasium + Stable-Baselines3        │
│  - Custom PyTorch/JAX loops             │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Python Environment Layer               │
│  - VideopacEnv, MO5Env                  │
│  - Framework-agnostic API               │
└──────────────┬──────────────────────────┘
               │ pybind11
┌──────────────▼──────────────────────────┐
│  C++ RL Interface (Generic)             │
│  - RLInterface base class               │
│  - Emulator-specific implementations    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Emulator Cores (Git Submodules)        │
│  - videopac, mo5, etc.                  │
└─────────────────────────────────────────┘
```

## Repository Structure

```
retro-ai/
├── README.md
├── LICENSE
├── .gitignore
├── .gitmodules
├── CMakeLists.txt
├── pyproject.toml
│
├── emulators/                  # Git submodules
│   ├── videopac/              # Videopac emulator
│   └── mo5/                   # MO5 emulator
│
├── include/
│   └── retro_ai/
│       ├── rl_interface.h     # Generic C++ interface
│       └── reward_system.h    # Reward computation
│
├── src/
│   ├── videopac_rl.cpp        # Videopac adapter
│   ├── mo5_rl.cpp             # MO5 adapter
│   └── reward_systems.cpp     # Reward implementations
│
├── python/
│   ├── retro_ai/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── base_env.py    # Framework-agnostic env
│   │   │   └── reward.py      # Reward strategies
│   │   ├── envs/
│   │   │   ├── __init__.py
│   │   │   ├── videopac.py
│   │   │   └── mo5.py
│   │   └── wrappers/
│   │       ├── __init__.py
│   │       ├── gymnasium.py   # Optional Gymnasium wrapper
│   │       └── preprocessing.py
│   └── bindings.cpp           # pybind11 bindings
│
├── examples/
│   ├── basic_training.py
│   ├── custom_reward.py
│   ├── gymnasium_training.py
│   └── multi_emulator.py
│
├── tests/
│   ├── test_interface.cpp
│   ├── test_videopac_env.py
│   └── test_rewards.py
│
└── docs/
    ├── getting_started.md
    ├── adding_emulators.md
    ├── reward_systems.md
    └── api_reference.md
```

## Core Requirements

### R1: C++ RL Interface

**Requirement:** Define a generic C++ interface that all emulators implement.

**Interface:**
```cpp
namespace retro_ai {

struct ObservationSpace {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t bits_per_channel;
};

struct ActionSpace {
    enum Type { DISCRETE, MULTI_DISCRETE, CONTINUOUS };
    Type type;
    std::vector<uint32_t> shape;
};

struct StepResult {
    std::vector<uint8_t> observation;
    float reward;
    bool done;
    bool truncated;
    std::string info;  // JSON metadata
};

class RLInterface {
public:
    virtual ~RLInterface() = default;
    
    // Core Gym-like API
    virtual StepResult reset() = 0;
    virtual StepResult step(const std::vector<int>& actions) = 0;
    
    // Space definitions
    virtual ObservationSpace observation_space() const = 0;
    virtual ActionSpace action_space() const = 0;
    
    // State management
    virtual std::vector<uint8_t> save_state() const = 0;
    virtual void load_state(const std::vector<uint8_t>& state) = 0;
    
    // Configuration
    virtual void set_reward_mode(const std::string& mode) = 0;
    virtual std::vector<std::string> available_reward_modes() const = 0;
    
    // Metadata
    virtual std::string emulator_name() const = 0;
    virtual std::string game_name() const = 0;
};

} // namespace retro_ai
```

**Acceptance Criteria:**
- Interface is pure virtual (no implementation)
- All methods are const-correct
- Uses standard C++ types (no emulator-specific types)
- Supports save/load state for experience replay

### R2: Python Base Environment

**Requirement:** Provide a framework-agnostic Python environment class.

**Interface:**
```python
class BaseEnv:
    """Framework-agnostic environment base class"""
    
    def __init__(self, emulator_type: str, **kwargs):
        """
        Args:
            emulator_type: "videopac", "mo5", etc.
            **kwargs: Emulator-specific arguments
        """
        pass
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation"""
        pass
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action and return (obs, reward, done, truncated, info)
        """
        pass
    
    def get_observation_space(self) -> dict:
        """Return observation space description"""
        pass
    
    def get_action_space(self) -> dict:
        """Return action space description"""
        pass
    
    def save_state(self) -> bytes:
        """Save current state"""
        pass
    
    def load_state(self, state: bytes):
        """Load saved state"""
        pass
    
    def set_reward_mode(self, mode: str):
        """Set reward computation mode"""
        pass
```

**Acceptance Criteria:**
- No dependency on Gymnasium or any RL framework
- Returns standard NumPy arrays
- Supports all reward modes
- Thread-safe for parallel training

### R3: Reward Systems

**Requirement:** Support multiple reward computation strategies.

**Modes:**

1. **Survival** - Simple time-based reward
   - +1 for each frame alive
   - -10 on game over
   - No game-specific knowledge needed

2. **Memory** - Read score from RAM
   - Requires game-specific memory addresses
   - Returns delta of score
   - Fast and accurate

3. **Vision** - OCR score from screen
   - Uses template matching or OCR
   - Slower but game-agnostic
   - Requires score to be visible

4. **Intrinsic** - Curiosity-driven
   - Rewards novel states
   - No game knowledge needed
   - Good for exploration

**Acceptance Criteria:**
- Each mode is a separate class
- Easy to add custom reward functions
- Configurable per-game
- Can combine multiple reward signals

### R4: Emulator Integration

**Requirement:** Each emulator provides an RLInterface implementation.

**Videopac Example:**
```cpp
class VideopacRLInterface : public retro_ai::RLInterface {
private:
    Emulator emulator_;
    std::string rom_path_;
    std::unique_ptr<RewardSystem> reward_system_;
    
public:
    VideopacRLInterface(const std::string& bios_path,
                        const std::string& rom_path,
                        const std::string& reward_mode = "survival");
    
    StepResult reset() override;
    StepResult step(const std::vector<int>& actions) override;
    // ... implement all interface methods
};
```

**Acceptance Criteria:**
- Emulator runs in headless mode (no SDL/graphics)
- Framebuffer accessible as raw pixels
- Can run at maximum speed (no frame limiting)
- Deterministic (same seed = same results)

### R5: Optional Gymnasium Wrapper

**Requirement:** Provide optional Gymnasium compatibility.

```python
import gymnasium as gym
from retro_ai.core import BaseEnv

class GymnasiumWrapper(gym.Env):
    """Optional wrapper for Gymnasium compatibility"""
    
    def __init__(self, base_env: BaseEnv):
        self.env = base_env
        
        # Convert to Gymnasium spaces
        obs_space = base_env.get_observation_space()
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_space['height'], obs_space['width'], obs_space['channels']),
            dtype=np.uint8
        )
        
        act_space = base_env.get_action_space()
        self.action_space = gym.spaces.Discrete(act_space['n'])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        return self.env.step(action)
```

**Acceptance Criteria:**
- Wrapper is optional (not required for core functionality)
- Compatible with Stable-Baselines3
- Supports all Gymnasium features (seeding, rendering, etc.)
- Can be easily replaced with other framework wrappers

### R6: Build System

**Requirement:** CMake-based build with Python packaging.

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.20)
project(retro_ai)

# Build emulator cores
add_subdirectory(emulators/videopac)
add_subdirectory(emulators/mo5)

# Build RL adapters
add_library(videopac_rl src/videopac_rl.cpp)
target_link_libraries(videopac_rl videopac_core)

add_library(mo5_rl src/mo5_rl.cpp)
target_link_libraries(mo5_rl mo5_core)

# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(retro_ai_native python/bindings.cpp)
target_link_libraries(retro_ai_native videopac_rl mo5_rl)
```

**pyproject.toml:**
```toml
[build-system]
requires = ["setuptools>=45", "wheel", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"

[project]
name = "retro-ai"
version = "0.1.0"
description = "Reinforcement learning framework for retro game emulators"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
gymnasium = ["gymnasium>=0.29.0"]
training = ["stable-baselines3>=2.0.0", "torch>=2.0.0"]
dev = ["pytest", "black", "mypy"]
```

**Acceptance Criteria:**
- Single command build: `pip install .`
- Submodules automatically initialized
- Optional dependencies for Gymnasium/training
- Works on Linux, macOS, Windows

## Usage Examples

### Basic Usage (No Framework)

```python
from retro_ai.envs import VideopacEnv
import numpy as np

# Create environment
env = VideopacEnv(
    bios_path="roms/videopac.bin",
    rom_path="roms/satellite_attack.bin",
    reward_mode="survival"
)

# Training loop
for episode in range(100):
    obs = env.reset()
    total_reward = 0
    
    while True:
        # Random policy
        action = np.random.randint(0, 32)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            print(f"Episode {episode}: {total_reward}")
            break
```

### With Gymnasium + Stable-Baselines3

```python
from retro_ai.envs import VideopacEnv
from retro_ai.wrappers import GymnasiumWrapper
from stable_baselines3 import PPO

# Create environment
base_env = VideopacEnv(
    bios_path="roms/videopac.bin",
    rom_path="roms/satellite_attack.bin",
    reward_mode="memory"
)

# Wrap for Gymnasium
env = GymnasiumWrapper(base_env)

# Train with PPO
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Save model
model.save("satellite_attack_agent")
```

### Custom Reward Function

```python
from retro_ai.envs import VideopacEnv
from retro_ai.core.reward import RewardFunction

class CustomReward(RewardFunction):
    def compute(self, obs, info):
        # Your custom logic
        score_delta = info.get('score_delta', 0)
        lives_lost = info.get('lives_lost', 0)
        return score_delta * 10 - lives_lost * 100

env = VideopacEnv(
    bios_path="roms/videopac.bin",
    rom_path="roms/satellite_attack.bin",
    reward_function=CustomReward()
)
```

## Success Criteria

1. Can train an agent on Videopac Satellite Attack
2. Can train an agent on MO5 game
3. Works with and without Gymnasium
4. Achieves >1000 FPS in headless mode
5. Deterministic (same seed = same results)
6. Easy to add new emulators (<200 lines of code)
7. Comprehensive documentation and examples

## Future Enhancements

- Multi-agent support (competitive/cooperative)
- Distributed training support
- Pre-trained model zoo
- Web-based visualization
- Integration with more RL frameworks (RLlib, Tianshou)
- Support for more emulators (NES, Game Boy, etc.)

## License

MIT License - allows commercial and academic use
