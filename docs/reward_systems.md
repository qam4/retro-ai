# Reward Systems

Retro-AI ships with four built-in reward modes plus support for custom
functions.  Switch modes at runtime with `env.set_reward_mode("name")`.

## Built-in Modes

### survival

Returns +1.0 every frame the agent is alive, −10.0 on game over.
Good for initial exploration when no score address is known.

### memory

Reads the game score directly from emulator RAM.  Returns the delta
between the current and previous score each frame.  Requires a
per-game memory address mapping (configured via `reward.parameters`).

### vision

Extracts the score from screen pixels using template-matching OCR.
Slower than memory mode but works without game-specific knowledge.
Configure the screen region via `reward.parameters`.

### intrinsic

Curiosity-driven reward based on state novelty.  Novel states receive
higher reward than previously visited ones.  Useful for exploration in
games with sparse external rewards.

## Switching Modes

```python
env = BaseEnv("videopac", rom_path="game.bin", bios_path="bios.bin")
print(env.available_reward_modes())  # ['survival', 'memory', 'vision', 'intrinsic', 'custom']

env.set_reward_mode("memory")
obs, reward, done, truncated, info = env.step(0)
```

## Custom Reward Functions

The C++ `CustomRewardSystem` accepts a callback that receives the
observation and info dict.  From Python, register a custom function
through the configuration system (see `examples/custom_rewards.py`).
