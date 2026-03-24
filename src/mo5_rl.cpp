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
#include "retro_ai/reward_systems/memory.hpp"

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
         const std::string& reward_mode,
         const RewardParams& reward_params = {},
         const std::string& action_mode = "discrete")
        : reward_mode_(reward_mode)
        , frame_number_(0)
        , reward_params_(reward_params)
        , reward_system_(RewardSystemFactory::create(reward_mode, reward_params))
        , action_mode_(action_mode)
    {
        // Pass BIOS ROM paths to the MO5 adapter before init
        auto basic_it = reward_params.find("basic_rom_path");
        auto monitor_it = reward_params.find("monitor_rom_path");
        if (basic_it != reward_params.end() && monitor_it != reward_params.end()) {
            mo5::set_rom_paths(basic_it->second, monitor_it->second);
        }

        if (!mo5::emulator_init(rom_path)) {
            throw InitializationError(
                "Failed to initialize MO5 emulator with ROM '" +
                rom_path + "'");
        }

        // Parse lives address for episode termination
        auto lives_it = reward_params.find("lives_addr");
        if (lives_it != reward_params.end()) {
            try { lives_addr_ = std::stoi(lives_it->second); } catch (...) {}
        }

        // Wire up memory reader for RAM-based reward systems
        wire_memory_reward_system();
    }

    void wire_memory_reward_system() {
        auto* mem_reward = dynamic_cast<MemoryRewardSystem*>(reward_system_.get());
        if (!mem_reward) return;
        mem_reward->set_memory_reader([this](uint16_t addr) -> uint8_t {
            return read_ram_byte(addr);
        });
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

        // Run startup sequence if configured
        // Format: "command1\nwait:N\ncommand2\nwait:N\n..."
        auto it = reward_params_.find("startup_sequence");
        if (it != reward_params_.end()) {
            run_startup_sequence(it->second);
        }

        // Initialize lives tracking
        if (lives_addr_ >= 0) {
            size_t ram_size = 0;
            const uint8_t* ram = mo5::memory_get_ram(ram_size);
            if (ram && static_cast<size_t>(lives_addr_) < ram_size) {
                previous_lives_ = ram[lives_addr_];
            }
        }

        if (reward_system_) {
            reward_system_->reset();
        }

        StepResult result;
        result.observation = extract_framebuffer();
        result.reward = 0.0f;
        result.done = false;
        result.truncated = false;
        result.info = "{\"frame_number\": " + std::to_string(frame_number_) + "}";

        previous_result_ = result;
        return result;
    }

    StepResult step(const std::vector<int>& action) {
        StepResult result;

        if (action_mode_ == "joystick") {
            // Joystick mode: [vert, horiz, fire]
            // vert: 0=neutral, 1=up, 2=down
            // horiz: 0=neutral, 1=right, 2=left
            // fire: 0=no, 1=yes (jump)
            if (action.size() != 3) {
                result.observation = extract_framebuffer();
                result.reward = 0.0f;
                result.done = false;
                result.truncated = true;
                result.info = "{\"frame_number\": " + std::to_string(frame_number_) +
                              ", \"error\": \"Joystick action must have 3 elements\"}";
                return result;
            }
            // Map to discrete actions: up=1, down=2, left=3, right=4, space=5
            // Apply all simultaneously via emulator_step_multi
            apply_joystick(action);
        } else {
            // Discrete mode: single action index
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
            mo5::emulator_step(action[0]);
        }
        mo5::video_render_frame();
        ++frame_number_;

        result.observation = extract_framebuffer();
        // Compute reward via pluggable reward system
        if (reward_system_) {
            result.reward = reward_system_->compute_reward(result, previous_result_);
        } else {
            result.reward = 0.0f;
        }

        // Check for life loss (episode termination)
        result.done = false;
        if (lives_addr_ >= 0) {
            size_t ram_size = 0;
            const uint8_t* ram = mo5::memory_get_ram(ram_size);
            if (ram && static_cast<size_t>(lives_addr_) < ram_size) {
                int current_lives = ram[lives_addr_];
                if (current_lives < previous_lives_ && previous_lives_ > 0) {
                    result.done = true;
                }
                previous_lives_ = current_lives;
            }
        }

        result.truncated = false;
        result.info = "{\"frame_number\": " + std::to_string(frame_number_) + "}";

        previous_result_ = result;
        return result;
    }

    ObservationSpace observation_space() const {
        return {kScreenWidth, kScreenHeight, kScreenChannels, 8};
    }

    ActionSpace action_space() const {
        if (action_mode_ == "joystick") {
            // [vertical(3), horizontal(3), fire(2)]
            return {ActionType::MULTI_DISCRETE, {3, 3, 2}};
        }
        return {ActionType::DISCRETE, {kNumActions}};
    }

    /// Apply joystick action [vert, horiz, fire] and run one frame.
    /// vert: 0=neutral, 1=up, 2=down
    /// horiz: 0=neutral, 1=right, 2=left
    /// fire: 0=no, 1=yes (jump/space)
    void apply_joystick(const std::vector<int>& action) {
        // Build a set of keys to press simultaneously
        // We need a multi-key step in the adapter
        std::vector<int> keys;
        if (action[0] == 1) keys.push_back(1);  // up
        if (action[0] == 2) keys.push_back(2);  // down
        if (action[1] == 1) keys.push_back(4);  // right
        if (action[1] == 2) keys.push_back(3);  // left
        if (action[2] == 1) keys.push_back(5);  // space/jump

        if (keys.empty()) {
            mo5::emulator_step(0);  // noop
        } else if (keys.size() == 1) {
            mo5::emulator_step(keys[0]);
        } else {
            // Multiple keys: need to press all at once
            // Use emulator_step_multi if available, otherwise just press first
            mo5::emulator_step_multi(keys);
        }
    }

    /// Parse and execute a startup sequence string.
    /// Format: pipe-separated commands: "wait:N|type:TEXT|wait:N|type:TEXT"
    /// type: types the text using AZERTY mapping (\n = ENTER)
    /// wait: runs N noop frames
    void run_startup_sequence(const std::string& seq) {
        size_t pos = 0;
        while (pos < seq.size()) {
            size_t next = seq.find('|', pos);
            std::string cmd = seq.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
            pos = (next == std::string::npos) ? seq.size() : next + 1;

            if (cmd.substr(0, 5) == "wait:") {
                int frames = std::stoi(cmd.substr(5));
                wait_frames(frames);
            } else if (cmd.substr(0, 5) == "type:") {
                std::string text = cmd.substr(5);
                // Replace literal \n with newline
                size_t p;
                while ((p = text.find("\\n")) != std::string::npos) {
                    text.replace(p, 2, "\n");
                }
                type_string(text);
            }
        }
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
        reward_system_ = RewardSystemFactory::create(mode, reward_params_);
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

    std::vector<uint8_t> read_ram() const {
        size_t size = 0;
        const uint8_t* ram = mo5::memory_get_ram(size);
        if (!ram || size == 0) return {};
        return std::vector<uint8_t>(ram, ram + size);
    }

    uint8_t read_ram_byte(uint16_t address) const {
        size_t size = 0;
        const uint8_t* ram = mo5::memory_get_ram(size);
        if (!ram || address >= size) return 0;
        return ram[address];
    }

    int ram_size() const {
        return static_cast<int>(mo5::RAM_SIZE);
    }

    /// Type a string on the MO5 keyboard with proper timing.
    /// Each character is held for `hold_frames` then released for `gap_frames`.
    void type_string(const std::string& text, int hold_frames = 3, int gap_frames = 3) {
        for (char c : text) {
            for (int i = 0; i < hold_frames; ++i) {
                mo5::emulator_type_char(c);
                mo5::video_render_frame();
                ++frame_number_;
            }
            for (int i = 0; i < gap_frames; ++i) {
                mo5::emulator_step(0);  // noop
                mo5::video_render_frame();
                ++frame_number_;
            }
        }
    }

    /// Run N noop frames (for waiting).
    void wait_frames(int n) {
        for (int i = 0; i < n; ++i) {
            mo5::emulator_step(0);
            mo5::video_render_frame();
            ++frame_number_;
        }
    }

private:
    /// Extract the current framebuffer as a flat RGB888 vector.
    std::vector<uint8_t> extract_framebuffer() const {
        const uint8_t* fb = mo5::video_get_framebuffer();
        return std::vector<uint8_t>(fb, fb + kFramebufferSize);
    }

    std::string reward_mode_;
    int frame_number_;
    RewardParams reward_params_;
    std::unique_ptr<RewardSystem> reward_system_;
    StepResult previous_result_;
    std::string action_mode_;
    int lives_addr_ = -1;
    int previous_lives_ = 0;
};

// ---------------------------------------------------------------------------
// MO5RLInterface forwarding methods
// ---------------------------------------------------------------------------

MO5RLInterface::MO5RLInterface(const std::string& rom_path,
                                const std::string& reward_mode,
                                const RewardParams& reward_params,
                                const std::string& action_mode)
    : impl_(std::make_unique<Impl>(rom_path, reward_mode, reward_params, action_mode))
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

void MO5RLInterface::type_string(const std::string& text, int hold_frames, int gap_frames) {
    impl_->type_string(text, hold_frames, gap_frames);
}

void MO5RLInterface::wait_frames(int n) {
    impl_->wait_frames(n);
}

std::vector<uint8_t> MO5RLInterface::read_ram() const {
    return impl_->read_ram();
}

uint8_t MO5RLInterface::read_ram_byte(uint16_t address) const {
    return impl_->read_ram_byte(address);
}

int MO5RLInterface::ram_size() const {
    return impl_->ram_size();
}

}  // namespace retro_ai
