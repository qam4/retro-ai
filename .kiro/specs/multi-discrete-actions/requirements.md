# Multi-Discrete Action Space

## Problem

The Videopac RL interface uses a flat discrete action space (18 actions) that enumerates specific input combinations. This means diagonal movement (Up+Left, Down+Right, etc.) is impossible — the agent can only move in 4 cardinal directions. The real Videopac joystick has 4 independent direction switches plus a fire button, so all combinations are valid on hardware.

Enumerating all combinations as flat discrete actions is the wrong abstraction. It inflates the action space unnecessarily and makes it harder for RL algorithms to discover that directions are independent.

## Solution

Add a multi-discrete action mode where the agent provides 5 independent binary values per step: `[up, down, left, right, fire]`. Each dimension has 2 values (0=off, 1=on). This maps directly to the emulator's `InputHandler` which already accepts each input independently.

The existing flat discrete mode (18 actions) remains available via an `action_mode` parameter, but the default switches to multi-discrete.

## Requirements

### R1: Multi-Discrete Action Mode

- WHEN `action_mode` is `"multi_discrete"` (the new default)
- THEN `action_space()` SHALL return `ActionType::MULTI_DISCRETE` with shape `[2, 2, 2, 2, 2]`
- AND `step()` SHALL accept a 5-element vector `[up, down, left, right, fire]` where each element is 0 or 1
- AND `apply_action()` SHALL set each joystick direction and fire button independently based on the 5 values
- AND diagonal movement (e.g. up=1, left=1) SHALL work correctly

### R2: Legacy Discrete Action Mode

- WHEN `action_mode` is `"discrete"`
- THEN the existing 18-action flat discrete behavior SHALL be preserved exactly as-is
- AND `action_space()` SHALL return `ActionType::DISCRETE` with shape `[18]`

### R3: Constructor Parameter

- The `VideopacRLInterface` constructor SHALL accept an `action_mode` parameter (string, default `"multi_discrete"`)
- Valid values: `"discrete"`, `"multi_discrete"`
- Invalid values SHALL throw `InitializationError`

### R4: Keyboard Keys

- Keyboard keys (0-7) are NOT included in the multi-discrete action space — they are for game selection, not gameplay
- A separate `press_key(int key)` method MAY be added later but is out of scope for this change

### R5: Python Layer

- `BaseEnv` SHALL pass `action_mode` through to the C++ interface
- `BaseEnv.step()` SHALL accept either `int` (discrete mode) or `list[int]` (multi-discrete mode) depending on the configured action mode
- `GymnasiumWrapper` already handles `MultiDiscrete` via the shape check — no changes needed there

### R6: Validation

- In multi-discrete mode, `step()` SHALL validate that the action vector has exactly 5 elements and each is 0 or 1
- Invalid actions SHALL set `truncated=true` and return an error in `info`, consistent with existing behavior

### R7: step_n Compatibility

- `step_n()` SHALL work with both action modes — the same action (flat int or 5-element vector) is repeated for N frames
