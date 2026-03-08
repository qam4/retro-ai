# Multi-Discrete Action Space — Design

## Overview

Switch the default Videopac action mode from flat discrete (18 actions) to multi-discrete (5 binary dimensions: up, down, left, right, fire). The emulator's `InputHandler` already supports simultaneous inputs — this change is purely in the RL interface layer.

## Changes

### C++ Layer

#### `include/retro_ai/videopac_rl.hpp`

- Add `action_mode` parameter to constructor (string, default `"multi_discrete"`)
- Store action mode in Impl
- Keep `kNumActions = 18` for discrete mode backward compat
- Add `static constexpr int kMultiDiscreteSize = 5;`

#### `src/videopac_rl.cpp`

**Constructor**: Store `action_mode` string in Impl. Validate it's `"discrete"` or `"multi_discrete"`, throw `InitializationError` otherwise.

**`action_space()`**: Return based on mode:
- `"discrete"` → `ActionType::DISCRETE`, shape `[18]`
- `"multi_discrete"` → `ActionType::MULTI_DISCRETE`, shape `[2, 2, 2, 2, 2]`

**`step()`**: Dispatch based on mode:
- `"discrete"` → existing `apply_action(action[0])` path (unchanged)
- `"multi_discrete"` → new `apply_multi_discrete_action(action)` that sets each input independently

**New `apply_multi_discrete_action(const std::vector<int>& action)`**:
```cpp
void apply_multi_discrete_action(const std::vector<int>& action) {
    InputHandler& input = emulator_->get_input_handler();
    const int joy = joystick_index_;
    if (action[0]) input.set_joystick_state(joy, Direction::Up, true);
    if (action[1]) input.set_joystick_state(joy, Direction::Down, true);
    if (action[2]) input.set_joystick_state(joy, Direction::Left, true);
    if (action[3]) input.set_joystick_state(joy, Direction::Right, true);
    if (action[4]) input.set_joystick_button(joy, true);
}
```

**Validation in multi-discrete mode**: Check `action.size() == 5` and each element is 0 or 1. Return truncated + error info on failure.

**`step_n()`**: Already takes `const std::vector<int>& action` — just needs to dispatch to the right apply function based on mode. The same action vector is reused for all N frames.

### Python Layer

#### `python/bindings.cpp`

- Add `action_mode` parameter to `VideopacRLInterface` constructor binding (string, default `"multi_discrete"`)

#### `python/retro_ai/envs/base_env.py`

- Accept `action_mode` kwarg, pass through to C++ constructor
- `step()` already takes `int` and wraps it as `[action]` — for multi-discrete, it should accept a list and pass it directly
- `get_action_space()` already returns the shape from C++ — no change needed

#### `python/retro_ai/wrappers/gymnasium_wrapper.py`

- Already handles `MultiDiscrete` via the shape length check — when shape has 5 elements, it creates `spaces.MultiDiscrete([2, 2, 2, 2, 2])`. No changes needed.
- The `step()` method signature accepts `int` but Gymnasium's `MultiDiscrete` passes `np.ndarray` — may need to convert to list for the C++ binding.

## What Doesn't Change

- `InputHandler` in the emulator — already supports simultaneous inputs
- `RLInterface` base class — already has `MULTI_DISCRETE` in the enum
- Reward systems — don't depend on action space
- Observation space — completely independent
- Save/load state — no action state to persist
- `clear_input()` — already clears everything after each frame
