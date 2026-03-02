/**
 * @file videopac_rl.cpp
 * @brief VideopacRLInterface implementation using the PIMPL pattern.
 *
 * Wraps the Videopac emulator core to provide a standard RLInterface.
 * The emulator runs in headless mode; framebuffer data is extracted as
 * raw RGB888 pixels after each frame.
 */

#include "retro_ai/videopac_rl.hpp"
#include "retro_ai/exceptions.hpp"
#include "retro_ai/reward_system.hpp"

#include <videopac/videopac_core.h>

#include <algorithm>
#include <cstring>

namespace retro_ai {

// Screen constants from the emulator core
static constexpr int kScreenWidth    = videopac::SCREEN_WIDTH;   // 160
static constexpr int kScreenHeight   = videopac::SCREEN_HEIGHT;  // 200
static constexpr int kScreenChannels = videopac::SCREEN_CHANNELS; // 3
static constexpr int kFramebufferSize = kScreenWidth * kScreenHeight * kScreenChannels;
static constexpr int kNumActions     = videopac::NUM_ACTIONS;    // 18

// ---------------------------------------------------------------------------
// PIMPL implementation
// ---------------------------------------------------------------------------

class VideopacRLInterface::Impl {
public:
    Impl(const std::string& bios_path,
         const std::string& rom_path,
         const std::string& reward_mode)
        : reward_mode_(reward_mode)
        , frame_number_(0)
        , reward_system_(RewardSystemFactory::create(reward_mode))
    {
        if (!videopac::emulator_init(bios_path, rom_path)) {
            throw InitializationError(
                "Failed to initialize Videopac emulator with BIOS '" +
                bios_path + "' and ROM '" + rom_path + "'");
        }
    }

    ~Impl() {
        videopac::emulator_shutdown();
    }

    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    StepResult reset(int seed) {
        videopac::emulator_reset(seed);
        videopac::video_render_frame();
        frame_number_ = 0;

        if (reward_system_) {
            reward_system_->reset();
        }

        StepResult result;
        result.observation = extract_framebuffer();
        result.reward = 0.0f;
        result.done = false;
        result.truncated = false;
        result.info = "{\"frame_number\": 0}";

        previous_result_ = result;
        return result;
    }

    StepResult step(const std::vector<int>& action) {
        StepResult result;

        // Validate action: expect exactly one discrete action
        if (action.empty() || action[0] < 0 || action[0] >= kNumActions) {
            int bad_action = action.empty() ? -1 : action[0];
            result.observation = extract_framebuffer();
            result.reward = 0.0f;
            result.done = false;
            result.truncated = true;
            result.info = "{\"frame_number\": " + std::to_string(frame_number_) +
                          ", \"error\": \"Invalid action " + std::to_string(bad_action) +
                          ", must be in range [0, " + std::to_string(kNumActions) + ")\"}";
            return result;
        }

        int act = action[0];
        videopac::emulator_step(act);
        videopac::video_render_frame();
        ++frame_number_;

        result.observation = extract_framebuffer();
        // Compute reward via pluggable reward system
        if (reward_system_) {
            result.reward = reward_system_->compute_reward(result, previous_result_);
        } else {
            result.reward = 0.0f;
        }
        result.done = false;
        result.truncated = false;
        result.info = "{\"frame_number\": " + std::to_string(frame_number_) + "}";

        previous_result_ = result;
        return result;
    }

    ObservationSpace observation_space() const {
        return {kScreenWidth, kScreenHeight, kScreenChannels, 8};
    }

    ActionSpace action_space() const {
        return {ActionType::DISCRETE, {kNumActions}};
    }

    std::vector<uint8_t> save_state() const {
        return videopac::state_save();
    }

    void load_state(const std::vector<uint8_t>& state) {
        if (!videopac::state_load(state)) {
            throw StateError("Failed to load Videopac emulator state");
        }
    }

    void set_reward_mode(const std::string& mode) {
        reward_mode_ = mode;
        reward_system_ = RewardSystemFactory::create(mode);
        if (reward_system_) {
            reward_system_->reset();
        }
    }

    std::vector<std::string> available_reward_modes() const {
        return RewardSystemFactory::available_modes();
    }

    std::string game_name() const {
        return videopac::emulator_get_rom_name();
    }

private:
    /// Extract the current framebuffer as a flat RGB888 vector.
    std::vector<uint8_t> extract_framebuffer() const {
        const uint8_t* fb = videopac::video_get_framebuffer();
        return std::vector<uint8_t>(fb, fb + kFramebufferSize);
    }

    std::string reward_mode_;
    int frame_number_;
    std::unique_ptr<RewardSystem> reward_system_;
    StepResult previous_result_;
};

// ---------------------------------------------------------------------------
// VideopacRLInterface forwarding methods
// ---------------------------------------------------------------------------

VideopacRLInterface::VideopacRLInterface(const std::string& bios_path,
                                          const std::string& rom_path,
                                          const std::string& reward_mode)
    : impl_(std::make_unique<Impl>(bios_path, rom_path, reward_mode))
{
}

VideopacRLInterface::~VideopacRLInterface() = default;

VideopacRLInterface::VideopacRLInterface(VideopacRLInterface&&) noexcept = default;
VideopacRLInterface& VideopacRLInterface::operator=(VideopacRLInterface&&) noexcept = default;

StepResult VideopacRLInterface::reset(int seed) {
    return impl_->reset(seed);
}

StepResult VideopacRLInterface::step(const std::vector<int>& action) {
    return impl_->step(action);
}

ObservationSpace VideopacRLInterface::observation_space() const {
    return impl_->observation_space();
}

ActionSpace VideopacRLInterface::action_space() const {
    return impl_->action_space();
}

std::vector<uint8_t> VideopacRLInterface::save_state() const {
    return impl_->save_state();
}

void VideopacRLInterface::load_state(const std::vector<uint8_t>& state) {
    impl_->load_state(state);
}

void VideopacRLInterface::set_reward_mode(const std::string& mode) {
    impl_->set_reward_mode(mode);
}

std::vector<std::string> VideopacRLInterface::available_reward_modes() const {
    return impl_->available_reward_modes();
}

std::string VideopacRLInterface::emulator_name() const {
    return "Videopac";
}

std::string VideopacRLInterface::game_name() const {
    return impl_->game_name();
}

}  // namespace retro_ai
