/**
 * @file mo5_rl.cpp
 * @brief MO5RLInterface implementation using the PIMPL pattern.
 *
 * Wraps the Thomson MO5 emulator core to provide a standard RLInterface.
 * The emulator runs in headless mode; framebuffer data is extracted as
 * raw RGB888 pixels after each frame.
 */

#include "retro_ai/mo5_rl.hpp"
#include "retro_ai/exceptions.hpp"
#include "retro_ai/reward_system.hpp"

#include <mo5/mo5_core.h>

#include <algorithm>
#include <cstring>

namespace retro_ai {

// Screen constants from the emulator core
static constexpr int kScreenWidth    = mo5::SCREEN_WIDTH;    // 320
static constexpr int kScreenHeight   = mo5::SCREEN_HEIGHT;   // 200
static constexpr int kScreenChannels = mo5::SCREEN_CHANNELS; // 3
static constexpr int kFramebufferSize = kScreenWidth * kScreenHeight * kScreenChannels;
static constexpr int kNumActions     = mo5::NUM_ACTIONS;     // 60

// ---------------------------------------------------------------------------
// PIMPL implementation
// ---------------------------------------------------------------------------

class MO5RLInterface::Impl {
public:
    Impl(const std::string& rom_path,
         const std::string& reward_mode)
        : reward_mode_(reward_mode)
        , frame_number_(0)
        , reward_system_(RewardSystemFactory::create(reward_mode))
    {
        if (!mo5::emulator_init(rom_path)) {
            throw InitializationError(
                "Failed to initialize MO5 emulator with ROM '" +
                rom_path + "'");
        }
    }

    ~Impl() {
        mo5::emulator_shutdown();
    }

    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    StepResult reset(int seed) {
        mo5::emulator_reset(seed);
        mo5::video_render_frame();
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
        mo5::emulator_step(act);
        mo5::video_render_frame();
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
        return mo5::state_save();
    }

    void load_state(const std::vector<uint8_t>& state) {
        if (!mo5::state_load(state)) {
            throw StateError("Failed to load MO5 emulator state");
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
        return mo5::emulator_get_rom_name();
    }

private:
    /// Extract the current framebuffer as a flat RGB888 vector.
    std::vector<uint8_t> extract_framebuffer() const {
        const uint8_t* fb = mo5::video_get_framebuffer();
        return std::vector<uint8_t>(fb, fb + kFramebufferSize);
    }

    std::string reward_mode_;
    int frame_number_;
    std::unique_ptr<RewardSystem> reward_system_;
    StepResult previous_result_;
};

// ---------------------------------------------------------------------------
// MO5RLInterface forwarding methods
// ---------------------------------------------------------------------------

MO5RLInterface::MO5RLInterface(const std::string& rom_path,
                                const std::string& reward_mode)
    : impl_(std::make_unique<Impl>(rom_path, reward_mode))
{
}

MO5RLInterface::~MO5RLInterface() = default;

MO5RLInterface::MO5RLInterface(MO5RLInterface&&) noexcept = default;
MO5RLInterface& MO5RLInterface::operator=(MO5RLInterface&&) noexcept = default;

StepResult MO5RLInterface::reset(int seed) {
    return impl_->reset(seed);
}

StepResult MO5RLInterface::step(const std::vector<int>& action) {
    return impl_->step(action);
}

ObservationSpace MO5RLInterface::observation_space() const {
    return impl_->observation_space();
}

ActionSpace MO5RLInterface::action_space() const {
    return impl_->action_space();
}

std::vector<uint8_t> MO5RLInterface::save_state() const {
    return impl_->save_state();
}

void MO5RLInterface::load_state(const std::vector<uint8_t>& state) {
    impl_->load_state(state);
}

void MO5RLInterface::set_reward_mode(const std::string& mode) {
    impl_->set_reward_mode(mode);
}

std::vector<std::string> MO5RLInterface::available_reward_modes() const {
    return impl_->available_reward_modes();
}

std::string MO5RLInterface::emulator_name() const {
    return "MO5";
}

std::string MO5RLInterface::game_name() const {
    return impl_->game_name();
}

}  // namespace retro_ai
