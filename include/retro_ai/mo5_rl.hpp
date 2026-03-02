#pragma once

#include <memory>
#include <string>
#include <vector>

#include "retro_ai/rl_interface.hpp"

namespace retro_ai {

/// Thomson MO5 emulator adapter implementing the RLInterface.
///
/// Uses the PIMPL pattern to hide emulator-specific details from consumers.
/// The emulator runs in headless mode (no SDL, no graphics output).
class MO5RLInterface : public RLInterface {
public:
    /// Construct a MO5 environment.
    /// @param rom_path    Path to the ROM or tape file to load.
    /// @param reward_mode Initial reward computation mode (default: "survival").
    /// @throws InitializationError if the emulator fails to initialize.
    explicit MO5RLInterface(const std::string& rom_path,
                            const std::string& reward_mode = "survival");

    ~MO5RLInterface() override;

    // Non-copyable, movable
    MO5RLInterface(const MO5RLInterface&) = delete;
    MO5RLInterface& operator=(const MO5RLInterface&) = delete;
    MO5RLInterface(MO5RLInterface&&) noexcept;
    MO5RLInterface& operator=(MO5RLInterface&&) noexcept;

    // Core RL methods
    StepResult reset(int seed = -1) override;
    StepResult step(const std::vector<int>& action) override;

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

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace retro_ai
