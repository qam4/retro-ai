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
