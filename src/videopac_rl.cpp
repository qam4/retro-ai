/**
 * @file videopac_rl.cpp
 * @brief VideopacRLInterface implementation using the real Videopac emulator.
 *
 * Wraps videopac::EmulatorCore to provide a standard RLInterface.
 * The emulator runs headless; the palette-indexed framebuffer (160×240,
 * indices 0-15) is converted to RGB888 after each frame.
 *
 * Action mapping (18 discrete actions):
 *   0  = NOOP
 *   1  = Joystick Up
 *   2  = Joystick Down
 *   3  = Joystick Left
 *   4  = Joystick Right
 *   5  = Fire
 *   6  = Up + Fire
 *   7  = Down + Fire
 *   8  = Left + Fire
 *   9  = Right + Fire
 *  10  = Key 0
 *  11  = Key 1
 *  12  = Key 2
 *  13  = Key 3
 *  14  = Key 4
 *  15  = Key 5
 *  16  = Key 6
 *  17  = Key 7
 */

#include "retro_ai/videopac_rl.hpp"
#include "retro_ai/exceptions.hpp"
#include "retro_ai/reward_system.hpp"

// Real videopac emulator headers
#include "emulator.h"
#include "input.h"
#include "types.h"
#include "vdc.h"
// savestate.h not needed — we use EmulatorCore::save_state(path) directly

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace retro_ai {

using videopac::EmulatorCore;
using videopac::Configuration;
using videopac::VideoStandard;
using videopac::InputHandler;
using videopac::Direction;
using videopac::VidKey;
using videopac::PALETTE_STANDARD;
using videopac::FRAMEBUFFER_WIDTH;
using videopac::FRAMEBUFFER_HEIGHT;

static constexpr int kScreenWidth    = VideopacRLInterface::kScreenWidth;
static constexpr int kScreenHeight   = VideopacRLInterface::kScreenHeight;
static constexpr int kScreenChannels = VideopacRLInterface::kScreenChannels;
static constexpr int kFramebufferSize = kScreenWidth * kScreenHeight * kScreenChannels;
static constexpr int kNumActions     = VideopacRLInterface::kNumActions;

// Number of warmup frames after reset before pressing Key1
static constexpr int kWarmupFramesBeforeKey1 = 30;
// Number of frames to hold Key1 pressed
static constexpr int kKey1HoldFrames = 10;
// Number of frames to run after Key1 to let the game start
static constexpr int kWarmupFramesAfterKey1 = 60;

// ---------------------------------------------------------------------------
// PIMPL implementation
// ---------------------------------------------------------------------------

class VideopacRLInterface::Impl {
public:
    Impl(const std::string& bios_path,
         const std::string& rom_path,
         const std::string& reward_mode,
         int joystick_index = 0,
         const RewardParams& reward_params = {})
        : bios_path_(bios_path)
        , rom_path_(rom_path)
        , reward_mode_(reward_mode)
        , joystick_index_(joystick_index)
        , frame_number_(0)
        , reward_params_(reward_params)
        , reward_system_(RewardSystemFactory::create(reward_mode, reward_params))
    {
        // Configure headless emulator (NTSC = 60 Hz)
        Configuration config;
        config.video_standard = VideoStandard::NTSC;
        config.bios_path = bios_path;
        config.enable_profile = false;

        emulator_ = std::make_unique<EmulatorCore>(config);

        auto bios_result = emulator_->load_bios(bios_path);
        if (bios_result.is_err()) {
            throw InitializationError(
                "Failed to load Videopac BIOS '" + bios_path +
                "': " + bios_result.error);
        }

        auto rom_result = emulator_->load_rom(rom_path);
        if (rom_result.is_err()) {
            throw InitializationError(
                "Failed to load Videopac ROM '" + rom_path +
                "': " + rom_result.error);
        }
    }

    ~Impl() = default;

    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    StepResult reset(int /*seed*/) {
        emulator_->reset();
        frame_number_ = 0;

        // Run warmup frames to get past the BIOS splash
        for (int i = 0; i < kWarmupFramesBeforeKey1; ++i) {
            emulator_->run_frame();
        }

        // Press Key1 to select game (most games need this)
        InputHandler& input = emulator_->get_input_handler();
        input.set_key_state(VidKey::Key1, true);
        for (int i = 0; i < kKey1HoldFrames; ++i) {
            emulator_->run_frame();
        }
        input.set_key_state(VidKey::Key1, false);

        // Let the game initialize after key press
        for (int i = 0; i < kWarmupFramesAfterKey1; ++i) {
            emulator_->run_frame();
        }

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

        // Validate action
        if (action.empty() || action[0] < 0 || action[0] >= kNumActions) {
            int bad_action = action.empty() ? -1 : action[0];
            result.observation = extract_framebuffer();
            result.reward = 0.0f;
            result.done = false;
            result.truncated = true;
            result.info = "{\"frame_number\": " + std::to_string(frame_number_) +
                          ", \"error\": \"Invalid action " +
                          std::to_string(bad_action) +
                          ", must be in range [0, " +
                          std::to_string(kNumActions) + ")\"}";
            return result;
        }

        int act = action[0];
        apply_action(act);
        emulator_->run_frame();
        clear_input();
        ++frame_number_;

        result.observation = extract_framebuffer();
        if (reward_system_) {
            result.reward =
                reward_system_->compute_reward(result, previous_result_);
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
        // Use the emulator's file-based save_state, then read the file
        // into memory. This handles all component serialization including
        // non-trivially-copyable types like std::vector in MemoryState.
        std::string tmp_path = get_temp_state_path();
        auto result = emulator_->save_state(tmp_path);
        if (result.is_err()) {
            throw StateError("Failed to save Videopac state: " + result.error);
        }

        // Read the file into a vector
        std::ifstream ifs(tmp_path, std::ios::binary | std::ios::ate);
        if (!ifs.good()) {
            throw StateError("Failed to read save state temp file");
        }
        auto size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(static_cast<size_t>(size));
        ifs.read(reinterpret_cast<char*>(data.data()), size);

        // Append our own frame_number_ at the end
        auto fn = frame_number_;
        const auto* fn_bytes = reinterpret_cast<const uint8_t*>(&fn);
        data.insert(data.end(), fn_bytes, fn_bytes + sizeof(fn));

        std::remove(tmp_path.c_str());
        return data;
    }

    void load_state(const std::vector<uint8_t>& state) {
        if (state.size() <= sizeof(int)) {
            throw StateError("Save state data too small");
        }

        // Extract our frame_number_ from the end
        size_t emu_size = state.size() - sizeof(int);
        std::memcpy(&frame_number_,
                     state.data() + emu_size, sizeof(frame_number_));

        // Write the emulator portion to a temp file, then load it
        std::string tmp_path = get_temp_state_path();
        {
            std::ofstream ofs(tmp_path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(state.data()), emu_size);
        }

        auto result = emulator_->load_state(tmp_path);
        std::remove(tmp_path.c_str());

        if (result.is_err()) {
            throw StateError("Failed to load Videopac state: " + result.error);
        }
    }

    static std::string get_temp_state_path() {
        // Use a fixed temp path — only one save/load at a time per process
        return "retro_ai_videopac_state.tmp";
    }

    void set_reward_mode(const std::string& mode) {
        reward_mode_ = mode;
        reward_system_ = RewardSystemFactory::create(mode, reward_params_);
        if (reward_system_) {
            reward_system_->reset();
        }
    }

    std::vector<std::string> available_reward_modes() const {
        return RewardSystemFactory::available_modes();
    }

    std::string game_name() const {
        // Extract filename from rom_path_
        auto pos = rom_path_.find_last_of("/\\");
        if (pos != std::string::npos) {
            return rom_path_.substr(pos + 1);
        }
        return rom_path_;
    }

private:
    /// Convert palette-indexed framebuffer to RGB888.
    std::vector<uint8_t> extract_framebuffer() const {
        const uint8_t* indexed_fb = emulator_->get_framebuffer();
        std::vector<uint8_t> rgb(kFramebufferSize);

        for (int i = 0; i < kScreenWidth * kScreenHeight; ++i) {
            uint8_t idx = indexed_fb[i] & 0x0F;  // clamp to 0-15
            const auto& c = PALETTE_STANDARD[idx];
            rgb[i * 3 + 0] = c.r;
            rgb[i * 3 + 1] = c.g;
            rgb[i * 3 + 2] = c.b;
        }
        return rgb;
    }

    /// Map a discrete action to emulator input.
    void apply_action(int action) {
        InputHandler& input = emulator_->get_input_handler();
        const int joy = joystick_index_;

        switch (action) {
        case 0:  // NOOP
            break;
        case 1:  // Up
            input.set_joystick_state(joy, Direction::Up, true);
            break;
        case 2:  // Down
            input.set_joystick_state(joy, Direction::Down, true);
            break;
        case 3:  // Left
            input.set_joystick_state(joy, Direction::Left, true);
            break;
        case 4:  // Right
            input.set_joystick_state(joy, Direction::Right, true);
            break;
        case 5:  // Fire
            input.set_joystick_button(joy, true);
            break;
        case 6:  // Up + Fire
            input.set_joystick_state(joy, Direction::Up, true);
            input.set_joystick_button(joy, true);
            break;
        case 7:  // Down + Fire
            input.set_joystick_state(joy, Direction::Down, true);
            input.set_joystick_button(joy, true);
            break;
        case 8:  // Left + Fire
            input.set_joystick_state(joy, Direction::Left, true);
            input.set_joystick_button(joy, true);
            break;
        case 9:  // Right + Fire
            input.set_joystick_state(joy, Direction::Right, true);
            input.set_joystick_button(joy, true);
            break;
        case 10: // Key 0
            input.set_key_state(VidKey::Key0, true);
            break;
        case 11: // Key 1
            input.set_key_state(VidKey::Key1, true);
            break;
        case 12: // Key 2
            input.set_key_state(VidKey::Key2, true);
            break;
        case 13: // Key 3
            input.set_key_state(VidKey::Key3, true);
            break;
        case 14: // Key 4
            input.set_key_state(VidKey::Key4, true);
            break;
        case 15: // Key 5
            input.set_key_state(VidKey::Key5, true);
            break;
        case 16: // Key 6
            input.set_key_state(VidKey::Key6, true);
            break;
        case 17: // Key 7
            input.set_key_state(VidKey::Key7, true);
            break;
        }
    }

    /// Clear all input after a frame so keys don't stick.
    void clear_input() {
        emulator_->get_input_handler().reset();
    }

    std::string bios_path_;
    std::string rom_path_;
    std::string reward_mode_;
    int joystick_index_;
    int frame_number_;
    RewardParams reward_params_;
    std::unique_ptr<RewardSystem> reward_system_;
    StepResult previous_result_;
    std::unique_ptr<EmulatorCore> emulator_;
};

// ---------------------------------------------------------------------------
// VideopacRLInterface forwarding methods
// ---------------------------------------------------------------------------

VideopacRLInterface::VideopacRLInterface(const std::string& bios_path,
                                          const std::string& rom_path,
                                          const std::string& reward_mode,
                                          int joystick_index,
                                          const RewardParams& reward_params)
    : impl_(std::make_unique<Impl>(bios_path, rom_path, reward_mode, joystick_index, reward_params))
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
