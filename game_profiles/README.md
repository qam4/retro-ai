# Game Profiles

Game profiles are YAML or JSON files that describe a specific game on a specific emulator. They capture ROM paths, action mappings, reward configuration, startup sequences, and preprocessing defaults so you can train on different games without manual configuration.

## Format

Each profile is a YAML (`.yaml`) or JSON (`.json`) file with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique identifier (e.g. `satellite_attack`) |
| `emulator_type` | string | yes | Emulator backend (`videopac`, `mo5`) |
| `rom_path` | string | yes | Path to the ROM file |
| `bios_path` | string | no | Path to the BIOS file (required for some emulators) |
| `display_name` | string | no | Human-readable name |
| `action_count` | int | no | Override the action space size |
| `reward_mode` | string | no | Reward system (`survival`, `memory`, `vision`, `intrinsic`) |
| `reward_params` | dict | no | Reward-specific parameters |
| `startup_sequence` | object | no | Actions to navigate from boot to gameplay |
| `grayscale` | bool | no | Convert frames to grayscale (default: true) |
| `resize` | [H, W] | no | Resize frames (default: [84, 84]) |
| `frame_stack` | int | no | Number of frames to stack (default: 4) |
| `frame_skip` | int | no | Number of frames to skip (default: 4) |

### Reward Parameters

The `reward_params` dictionary holds reward-mode-specific configuration. Its contents depend on the `reward_mode` field.

#### Vision Mode (`reward_mode: vision`)

Use `screen_region` to tell the vision reward system where the score is rendered on screen:

```yaml
reward_mode: vision
reward_params:
  screen_region:
    x: 112        # left edge in pixels
    y: 80         # top edge in pixels
    width: 40     # region width in pixels
    height: 14    # region height in pixels
```

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `x` | int | ≥ 0 | 112 | Left edge of the score region (pixels) |
| `y` | int | ≥ 0 | 80 | Top edge of the score region (pixels) |
| `width` | int | ≥ 0 | 40 | Width of the score region (pixels) |
| `height` | int | ≥ 0 | 14 | Height of the score region (pixels) |

All fields are optional. When `screen_region` is absent, the system uses the defaults above. Negative values raise a `ConfigurationError`.

Use `scripts/framebuffer_visualizer.py` to interactively find the score region for a new game.

#### Memory Mode (`reward_mode: memory`)

Use `score_addresses` to specify which RAM locations hold the game score:

```yaml
reward_mode: memory
reward_params:
  score_addresses:
    - address: 0x003A
      num_bytes: 1
      is_bcd: true
    - address: 0x003B
      num_bytes: 1
      is_bcd: true
```

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `address` | int | 0–65535 | *(required)* | RAM address to read |
| `num_bytes` | int | 1, 2, or 4 | 1 | Number of bytes to read |
| `is_bcd` | bool | true/false | false | Whether the value is BCD-encoded |

Addresses outside 0–65535 or `num_bytes` not in {1, 2, 4} raise a `ConfigurationError`. When `score_addresses` is absent in memory mode, the reward system returns 0.0.

Use `scripts/ram_watcher.py` to discover score addresses for a new game.

### Startup Sequence

The `startup_sequence` object contains:
- `actions`: list of `{action, frames}` pairs — each action index is held for the given number of frames
- `post_delay_frames`: number of no-op frames to wait after the sequence completes

## Usage

Profiles are discovered automatically from this directory. Reference them by name in a training config:

```yaml
game_profile: satellite_attack
```

Or load them programmatically:

```python
from retro_ai.training.game_profile import GameProfileRegistry

registry = GameProfileRegistry()
print(registry.list_profiles())
profile = registry.load("satellite_attack")
```
