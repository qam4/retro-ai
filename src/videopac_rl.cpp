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

#ifdef RETRO_AI_PROFILING
#include <chrono>
#endif

// For wiring up memory-based rewards
#include "retro_ai/reward_systems/memory.hpp"

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
        , rgb_buffer_(kFramebufferSize)
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

        // Wire up memory reader for RAM-based reward systems
        wire_memory_reward_system();

        // Parse timer addresses from reward_params for episode termination
        parse_timer_params();
    }

    ~Impl() = default;

    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    /// Wire up the MemoryRewardSystem with a reader that accesses our emulator RAM.
    void wire_memory_reward_system() {
        auto* mem_reward = dynamic_cast<MemoryRewardSystem*>(reward_system_.get());
        if (!mem_reward) return;

        // Provide a reader lambda that reads from our flat 192-byte RAM layout
        // (64 internal + 128 external, same as read_ram()).
        mem_reward->set_memory_reader([this](uint16_t addr) -> uint8_t {
            return read_ram_byte(addr);
        });

        // If no score addresses were configured via the factory params,
        // set up the Course Automobile defaults:
        // Score = 2-byte BCD at IntRAM[54] (little-endian: low byte first)
        int count = 0;
        auto it = reward_params_.find("score_address_count");
        if (it != reward_params_.end()) {
            try { count = std::stoi(it->second); } catch (...) {}
        }
        if (count == 0) {
            // Use the discovered addresses: IntRAM[54] (addr 54), 2 bytes BCD LE
            std::vector<MemoryAddress> addrs;
            addrs.push_back({54, 2, true, true});  // 2-byte BCD LE score at IntRAM[54..55]
            mem_reward->set_score_addresses(std::move(addrs));
        }
    }

    /// Parse timer-related params for episode termination.
    void parse_timer_params() {
        // Default timer addresses for Course Automobile:
        // ExtRAM[0x01] (addr 65) = minutes, ExtRAM[0x02] (addr 66) = seconds
        // Both BCD. Game over when both are 0.
        timer_minutes_addr_ = 65;
        timer_seconds_addr_ = 66;
        done_when_timer_zero_ = false;

        auto it = reward_params_.find("done_when_timer_zero");
        if (it != reward_params_.end() && it->second == "true") {
            done_when_timer_zero_ = true;
        }

        // Allow overriding timer addresses from params
        it = reward_params_.find("timer_minutes_addr");
        if (it != reward_params_.end()) {
            try { timer_minutes_addr_ = std::stoi(it->second); } catch (...) {}
        }
        it = reward_params_.find("timer_seconds_addr");
        if (it != reward_params_.end()) {
            try { timer_seconds_addr_ = std::stoi(it->second); } catch (...) {}
        }
    }

    /// Check if the game timer has reached zero.
    bool is_timer_expired() const {
        if (!done_when_timer_zero_) return false;
        if (timer_minutes_addr_ >= 192 || timer_seconds_addr_ >= 192) {
            return false;
        }
        return read_ram_byte(static_cast<uint16_t>(timer_minutes_addr_)) == 0 &&
               read_ram_byte(static_cast<uint16_t>(timer_seconds_addr_)) == 0;
    }

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
        const auto& fb = extract_framebuffer();
        result.observation.assign(fb.begin(), fb.end());
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
            const auto& fb = extract_framebuffer();
            result.observation.assign(fb.begin(), fb.end());
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

#ifdef RETRO_AI_PROFILING
        using Clock = std::chrono::high_resolution_clock;
        auto step_start = Clock::now();
#endif

        int act = action[0];
        apply_action(act);

#ifdef RETRO_AI_PROFILING
        auto cpu_start = Clock::now();
#endif
        emulator_->run_frame();
#ifdef RETRO_AI_PROFILING
        auto cpu_end = Clock::now();
#endif

        clear_input();
        ++frame_number_;

#ifdef RETRO_AI_PROFILING
        auto fb_start = Clock::now();
#endif
        const auto& fb = extract_framebuffer();
        result.observation.assign(fb.begin(), fb.end());
#ifdef RETRO_AI_PROFILING
        auto fb_end = Clock::now();
#endif

#ifdef RETRO_AI_PROFILING
        auto reward_start = Clock::now();
#endif
        if (reward_system_) {
            result.reward =
                reward_system_->compute_reward(result, previous_result_);
        } else {
            result.reward = 0.0f;
        }
#ifdef RETRO_AI_PROFILING
        auto reward_end = Clock::now();
#endif

        result.done = is_timer_expired();
        result.truncated = false;
        result.info = "{\"frame_number\": " + std::to_string(frame_number_) + "}";

#ifdef RETRO_AI_PROFILING
        auto step_end = Clock::now();
        auto to_us = [](auto dur) {
            return std::chrono::duration<double, std::micro>(dur).count();
        };
        last_timings_.cpu_us = to_us(cpu_end - cpu_start);
        last_timings_.framebuffer_us = to_us(fb_end - fb_start);
        last_timings_.reward_us = to_us(reward_end - reward_start);
        last_timings_.vdc_us = 0.0;  // VDC is part of run_frame; separate instrumentation requires emulator changes
        last_timings_.total_us = to_us(step_end - step_start);
#endif

        previous_result_ = result;
        return result;
    }

    StepResult step_n(const std::vector<int>& action, int n) {
        StepResult result;

        // Handle n < 1: return current state with 0 reward
        if (n < 1) {
            const auto& fb = extract_framebuffer();
            result.observation.assign(fb.begin(), fb.end());
            result.reward = 0.0f;
            result.done = false;
            result.truncated = false;
            result.info = "{\"frame_number\": " + std::to_string(frame_number_) + "}";
            return result;
        }

        // Validate action
        if (action.empty() || action[0] < 0 || action[0] >= kNumActions) {
            int bad_action = action.empty() ? -1 : action[0];
            const auto& fb = extract_framebuffer();
            result.observation.assign(fb.begin(), fb.end());
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
        float total_reward = 0.0f;
        bool done = false;

        for (int i = 0; i < n; ++i) {
            apply_action(act);
            emulator_->run_frame();
            clear_input();
            ++frame_number_;

            // Compute reward without framebuffer extraction.
            // MemoryRewardSystem uses read_ram_byte() via its reader callback,
            // so it doesn't need the observation. Create a minimal StepResult
            // with just frame info for the reward computation.
            if (reward_system_) {
                StepResult intermediate;
                intermediate.reward = 0.0f;
                intermediate.done = false;
                intermediate.truncated = false;
                intermediate.info = "{\"frame_number\": " + std::to_string(frame_number_) + "}";
                total_reward += reward_system_->compute_reward(intermediate, previous_result_);
                previous_result_ = intermediate;
            }

            done = is_timer_expired();
            if (done) break;
        }

        // Extract framebuffer only for the final frame
        const auto& fb = extract_framebuffer();
        result.observation.assign(fb.begin(), fb.end());
        result.reward = total_reward;
        result.done = done;
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
        wire_memory_reward_system();
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

    std::vector<uint8_t> read_ram() const {
        // Return 64 bytes internal RAM (8048 CPU) + 128 bytes external RAM = 192 bytes
        // This is the game-relevant state for score/timer discovery.
        auto cpu_state = emulator_->get_cpu_state();
        auto mem_state = emulator_->get_memory_state();

        std::vector<uint8_t> ram;
        ram.reserve(64 + 128);
        ram.insert(ram.end(), cpu_state.ram, cpu_state.ram + 64);
        ram.insert(ram.end(), mem_state.external_ram, mem_state.external_ram + 128);
        return ram;
    }

    uint8_t read_ram_byte(uint16_t address) const {
        if (address < 64) {
            return emulator_->get_cpu_state().ram[address];
        }
        if (address < 192) {
            return emulator_->get_memory_state().external_ram[address - 64];
        }
        return 0;
    }

    int ram_size() const {
        return 192;
    }

private:
    /// Convert palette-indexed framebuffer to RGB888 into pre-allocated buffer.
    const std::vector<uint8_t>& extract_framebuffer() {
        const uint8_t* indexed_fb = emulator_->get_framebuffer();

        for (int i = 0; i < kScreenWidth * kScreenHeight; ++i) {
            uint8_t idx = indexed_fb[i] & 0x0F;  // clamp to 0-15
            const auto& c = PALETTE_STANDARD[idx];
            rgb_buffer_[i * 3 + 0] = c.r;
            rgb_buffer_[i * 3 + 1] = c.g;
            rgb_buffer_[i * 3 + 2] = c.b;
        }
        return rgb_buffer_;
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
    std::vector<uint8_t> rgb_buffer_;  // pre-allocated framebuffer (Requirement 4)

    // Timer-based episode termination
    int timer_minutes_addr_ = -1;
    int timer_seconds_addr_ = -1;
    bool done_when_timer_zero_ = false;

#ifdef RETRO_AI_PROFILING
    FrameTimings last_timings_{};
#endif

public:
    FrameTimings get_last_frame_timings() const {
#ifdef RETRO_AI_PROFILING
        return last_timings_;
#else
        return {};
#endif
    }
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

StepResult VideopacRLInterface::step_n(const std::vector<int>& action, int n) {
    return impl_->step_n(action, n);
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

std::vector<uint8_t> VideopacRLInterface::read_ram() const {
    return impl_->read_ram();
}

uint8_t VideopacRLInterface::read_ram_byte(uint16_t address) const {
    return impl_->read_ram_byte(address);
}

int VideopacRLInterface::ram_size() const {
    return impl_->ram_size();
}

FrameTimings VideopacRLInterface::get_last_frame_timings() const {
    return impl_->get_last_frame_timings();
}

}  // namespace retro_ai
