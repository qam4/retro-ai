# API Reference

All public Python classes include full docstrings.  Generate HTML docs
with `pydoc` or use your IDE's hover/autocomplete.

## Quick Reference

### retro_ai.BaseEnv

The main entry point.  Framework-agnostic environment wrapper.

```python
env = BaseEnv(emulator_type, rom_path, bios_path=None, reward_mode="survival")

obs, info = env.reset(seed=None)
obs, reward, done, truncated, info = env.step(action)

env.get_observation_space()   # dict with width, height, channels
env.get_action_space()        # dict with type, shape

state = env.save_state()
env.load_state(state)

env.set_reward_mode("memory")
env.available_reward_modes()  # list of strings
```

### retro_ai.wrappers.gymnasium_wrapper.GymnasiumWrapper

Wraps a `BaseEnv` as a `gymnasium.Env`.  Requires `pip install gymnasium`.

```python
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper

gym_env = GymnasiumWrapper(base_env, render_mode="rgb_array")
```

### retro_ai.core.config.ConfigParser

Load/save emulator configs in JSON or YAML.

```python
from retro_ai.core.config import ConfigParser

cfg = ConfigParser.from_json("config.json")
ConfigParser.to_yaml(cfg, "config.yaml")
```

### retro_ai.core.preprocessing.PreprocessingPipeline

Frame transforms: grayscale, resize, stack, skip.

```python
from retro_ai.core.preprocessing import PreprocessingPipeline, PreprocessedEnv

pipe = PreprocessingPipeline(grayscale=True, resize=(84, 84), frame_stack=4)
wrapped = PreprocessedEnv(base_env, pipe, frame_skip=4)
```

### retro_ai.core.logging.StructuredLogger

Structured logging with episode lifecycle tracking.

```python
from retro_ai.core.logging import StructuredLogger

log = StructuredLogger("retro_ai.train", json_output=True)
log.log_env_created("videopac", "game.bin", "survival")
log.log_reset(seed=42)
log.log_step(reward=1.0, done=False)
log.log_episode_end()
```

### Policy Architectures

The training pipeline uses Stable-Baselines3 policy networks. The `policy`
field in training configs selects the architecture:

**CnnPolicy (NatureCNN)** — for framebuffer observations (84×84 images):

```
Input: (84, 84, C) where C = channels × frame_stack
  → Conv2d(C, 32, kernel=8, stride=4) + ReLU
  → Conv2d(32, 64, kernel=4, stride=2) + ReLU
  → Conv2d(64, 64, kernel=3, stride=1) + ReLU
  → Flatten → Linear(3136, 512) + ReLU
  → Policy head (action logits) + Value head (scalar)
```

This is the architecture from Mnih et al. 2015 ("Human-level control through
deep reinforcement learning", Nature). Total ~1.7M parameters for a
Discrete(18) action space, slightly more for MultiDiscrete([2,2,2,2,2])
since it has 5 independent output heads.

**MlpPolicy** — for RAM observations (flat byte vectors):

```
Input: (ram_size,) uint8 vector
  → Linear(ram_size, 64) + Tanh
  → Linear(64, 64) + Tanh
  → Policy head (action logits) + Value head (scalar)
```

Much smaller (~8K parameters) and faster to train, but requires meaningful
RAM addresses to be identified for the game.

### Exceptions

All inherit from `retro_ai.RetroAIError`:

- `InitializationError` — bad ROM, missing BIOS
- `InvalidActionError` — out-of-range action
- `StateError` — save/load failure
- `ConfigurationError` — invalid config
