# Implementation Plan

- [x] 1. Add `action_mode` to C++ constructor and Impl
  - Add `action_mode` string parameter to `VideopacRLInterface` constructor (default `"multi_discrete"`)
  - Store in Impl, validate is `"discrete"` or `"multi_discrete"` (throw `InitializationError` otherwise)
  - Add `static constexpr int kMultiDiscreteSize = 5;` to the header
  - Update header doc comment to reflect new action modes
  - _Requirements: R3_

- [x] 2. Implement `action_space()` dispatch and multi-discrete step logic
  - `action_space()`: return `MULTI_DISCRETE` shape `[2,2,2,2,2]` or `DISCRETE` shape `[18]` based on mode
  - Add `apply_multi_discrete_action()` in Impl: maps 5-element vector to independent `set_joystick_state`/`set_joystick_button` calls
  - `step()`: validate action size (5 elements, each 0 or 1) in multi-discrete mode, dispatch to `apply_multi_discrete_action()`
  - `step_n()`: same dispatch logic, reuse action vector across N frames
  - Discrete mode path remains completely unchanged
  - _Requirements: R1, R2, R6, R7_

- [x] 3. Update Python bindings and BaseEnv
  - Add `action_mode` parameter to `VideopacRLInterface` pybind11 constructor (default `"multi_discrete"`)
  - Update `BaseEnv.__init__()` to accept and pass through `action_mode`
  - Update `BaseEnv.step()` to handle list input for multi-discrete (pass list directly instead of wrapping in `[action]`)
  - _Requirements: R3, R5_

- [x] 4. Update GymnasiumWrapper for multi-discrete actions
  - Ensure `step()` converts `np.ndarray` action from Gymnasium to a Python list for the C++ binding
  - Verify `MultiDiscrete([2,2,2,2,2])` space is created correctly from shape
  - _Requirements: R5_

- [x] 5. Build and test
  - Build: `cmake --build build/ci-linux --target retro_ai_native -j4`
  - Verify both action modes work with a quick smoke test
  - Verify diagonal movement works in multi-discrete mode (up+left, down+right, etc.)
  - Verify discrete mode is unchanged
  - Run existing test suite to check for regressions
  - _Requirements: R1, R2, R3, R5, R6, R7_
