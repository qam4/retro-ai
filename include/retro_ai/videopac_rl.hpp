#pragma once

#include <memory>
#include <string>
#include <vector>

#include "retro_ai/rl_interface.hpp"
#include "retro_ai/reward_system.hpp"

namespace retro_ai {

/// Videopac (Odyssey 2) emulator adapter implementing the RLInterface.
///
/// Uses the PIMPL pattern to hide emulator-specific details from consumers.
/// The emulator runs in headless mode (no SDL, no graphics output).
///
/// Framebuffer: 160×240 palette-indexed pixels are converted to RGB888.
/// Action modes:
///   - "discrete": 18 flat discrete actions (joystick directions + fire + keyboard keys).
///   - "multi_discrete": 5 independent binary dimensions [up, down, left, right, fire],
///     enabling diagonal movement and simultaneous inputs.
///   - "joystick": 3 axes [vertical(3), horizontal(3), fire(2)] = MultiDiscrete([3,3,2]).
///     Physically correct joystick model: 0=neutral, 1=up/right, 2=down/left.
///     18 valid combinations, no impossible states like up+down.
class VideopacRLInterface : public RLInterface {
public:
    /// Construct a Videopac environment.
    /// @param bios_path  Path to the Videopac BIOS file.
    /// @param rom_path   Path to the ROM file to load.
    /// @param reward_mode Initial reward computation mode (default: "survival").
    /// @param joystick_index Which joystick to use for actions (0 or 1).
    /// @param reward_params Additional reward system parameters.
    /// @param action_mode Action space mode: "discrete" (18 flat actions) or
    ///        "multi_discrete" (5 binary dims: up/down/left/right/fire).
    /// @throws InitializationError if the emulator fails to initialize or
    ///         action_mode is invalid.
    explicit VideopacRLInterface(const std::string& bios_path,
                                  const std::string& rom_path,
                                  const std::string& reward_mode = "survival",
                                  int joystick_index = 0,
                                  const RewardParams& reward_params = {},
                                  const std::string& action_mode = "multi_discrete");

    ~VideopacRLInterface() override;

    // Non-copyable, movable
    VideopacRLInterface(const VideopacRLInterface&) = delete;
    VideopacRLInterface& operator=(const VideopacRLInterface&) = delete;
    VideopacRLInterface(VideopacRLInterface&&) noexcept;
    VideopacRLInterface& operator=(VideopacRLInterface&&) noexcept;

    // Core RL methods
    StepResult reset(int seed = -1) override;
    StepResult step(const std::vector<int>& action) override;
    StepResult step_n(const std::vector<int>& action, int n) override;

    // Space queries
    ObservationSpace observation_space() const override;
    ActionSpace action_space() const override;

    // State management
    std::vector<uint8_t> save_state() const override;
    void load_state(const std::vector<uint8_t>& state) override;

    // Reward configuration
    void set_reward_mode(const std::string& mode) override;
    std::vector<std::string> available_reward_modes() const override;

    // Metadata
    std::string emulator_name() const override;
    std::string game_name() const override;

    // RAM inspection
    std::vector<uint8_t> read_ram() const override;
    uint8_t read_ram_byte(uint16_t address) const override;
    int ram_size() const override;

    // Profiling
    FrameTimings get_last_frame_timings() const override;

    // Screen dimensions (from the real VDC)
    static constexpr int kScreenWidth = 160;
    static constexpr int kScreenHeight = 240;
    static constexpr int kScreenChannels = 3;
    static constexpr int kNumActions = 18;
    static constexpr int kMultiDiscreteSize = 5;
    static constexpr int kJoystickAxes = 3;  // [vertical, horizontal, fire]

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace retro_ai
