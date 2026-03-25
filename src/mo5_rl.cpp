/**
 * @file mo5_rl.cpp
 * @brief MO5RLInterface implementation using Crayon EmulatorCore directly.
 *
 * Each Impl instance owns its own EmulatorCore — no global state.
 * This allows multiple parallel environments (ThreadedVecEnv).
 */

#include "retro_ai/mo5_rl.hpp"
#include "retro_ai/exceptions.hpp"
#include "retro_ai/reward_system.hpp"
#include "retro_ai/reward_systems/memory.hpp"

#include "emulator_core.h"
#include "input_handler.h"
#include "cassette_interface.h"
#include "char_mapping.h"
#include "savestate.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace retro_ai {

static constexpr int kScreenWidth    = 320;
static constexpr int kScreenHeight   = 200;
static constexpr int kScreenChannels = 3;
static constexpr int kFramebufferSize = kScreenWidth * kScreenHeight * kScreenChannels;
static constexpr int kNumActions     = 60;
static constexpr int kRamSize        = 49152;  // 48KB: video_ram[16K] + user_ram[32K]

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
        , rgb_buffer_(kFramebufferSize)
    {
        // Create emulator
        crayon::Configuration config;
        auto basic_it = reward_params.find("basic_rom_path");
        auto monitor_it = reward_params.find("monitor_rom_path");
        if (basic_it != reward_params.end())
            config.basic_rom_path = basic_it->second;
        if (monitor_it != reward_params.end())
            config.monitor_rom_path = monitor_it->second;

        emulator_ = std::make_unique<crayon::EmulatorCore>(config);

        // Load ROMs
        if (!config.basic_rom_path.empty() && !config.monitor_rom_path.empty()) {
            auto result = emulator_->load_roms(config.basic_rom_path, config.monitor_rom_path);
            if (result.is_err()) {
                throw InitializationError("Failed to load MO5 ROMs: " + result.error);
            }
        }

        // Load cassette or cartridge
        if (rom_path.size() >= 3) {
            std::string ext = rom_path.substr(rom_path.size() - 3);
            if (ext == ".k7" || ext == ".K7") {
                auto& cassette = emulator_->get_cassette();
                auto r = cassette.load_k7(rom_path);
                if (r.is_err()) {
                    throw InitializationError("Failed to load K7: " + r.error);
                }
                emulator_->play_cassette();
            } else if (ext == "mo5" || ext == "MO5") {
                auto r = emulator_->load_cartridge(rom_path);
                if (r.is_err()) {
                    throw InitializationError("Failed to load cartridge: " + r.error);
                }
            }
        }

        emulator_->reset();
        rom_name_ = rom_path;

        // Parse lives address
        auto lives_it = reward_params.find("lives_addr");
        if (lives_it != reward_params.end()) {
            try { lives_addr_ = std::stoi(lives_it->second); } catch (...) {}
        }

        // Parse height reward params
        auto height_it = reward_params.find("height_reward_addr");
        if (height_it != reward_params.end()) {
            try { height_addr_ = std::stoi(height_it->second); } catch (...) {}
        }
        auto coeff_it = reward_params.find("height_reward_coeff");
        if (coeff_it != reward_params.end()) {
            try { height_coeff_ = std::stof(coeff_it->second); } catch (...) {}
        }

        // Wire up memory reader for reward system
        wire_memory_reward_system();
    }

    ~Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void wire_memory_reward_system() {
        auto* mem_reward = dynamic_cast<MemoryRewardSystem*>(reward_system_.get());
        if (!mem_reward) return;
        mem_reward->set_memory_reader([this](uint16_t addr) -> uint8_t {
            return read_ram_byte(addr);
        });
    }

    StepResult reset(int /*seed*/) {
        if (!startup_state_.empty()) {
            // Fast path: restore cached post-startup state
            auto result = crayon::SaveStateManager::deserialize_from_buffer(
                startup_state_.data(), startup_state_.size());
            if (result.is_ok() && result.value.has_value()) {
                const auto& state = result.value.value();
                emulator_->get_cpu().set_state(state.cpu_state);
                emulator_->get_gate_array().set_state(state.gate_array_state);
                emulator_->get_memory().set_state(state.memory_state);
                emulator_->get_pia().set_state(state.pia_state);
            }
        } else {
            emulator_->reset();
            // Run startup sequence if configured
            auto it = reward_params_.find("startup_sequence");
            if (it != reward_params_.end()) {
                run_startup_sequence(it->second);
                // Cache state for fast subsequent resets
                cache_startup_state();
            }
        }

        render_frame();
        frame_number_ = 0;

        // Initialize lives tracking
        if (lives_addr_ >= 0) {
            previous_lives_ = read_ram_byte(static_cast<uint16_t>(lives_addr_));
        }

        // Initialize height tracking
        if (height_addr_ >= 0) {
            previous_y_ = read_ram_byte(static_cast<uint16_t>(height_addr_));
            best_y_ = previous_y_;  // reset best height each episode
        }

        if (reward_system_) {
            reward_system_->reset();
        }

        StepResult result;
        result.observation = std::vector<uint8_t>(rgb_buffer_.begin(), rgb_buffer_.end());
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
            if (action.size() != 3) {
                result.observation = std::vector<uint8_t>(rgb_buffer_.begin(), rgb_buffer_.end());
                result.reward = 0.0f;
                result.done = false;
                result.truncated = true;
                result.info = "{\"error\": \"Joystick action must have 3 elements\"}";
                return result;
            }
            apply_joystick(action);
        } else {
            if (action.empty() || action[0] < 0 || action[0] >= kNumActions) {
                result.observation = std::vector<uint8_t>(rgb_buffer_.begin(), rgb_buffer_.end());
                result.reward = 0.0f;
                result.done = false;
                result.truncated = true;
                result.info = "{\"error\": \"Invalid action\"}";
                return result;
            }
            apply_discrete_action(action[0]);
        }

        emulator_->run_frame();
        render_frame();
        ++frame_number_;

        result.observation = std::vector<uint8_t>(rgb_buffer_.begin(), rgb_buffer_.end());
        if (reward_system_) {
            result.reward = reward_system_->compute_reward(result, previous_result_);
        } else {
            result.reward = 0.0f;
        }

        // Height milestone reward: one-time bonus when reaching new platform levels
        // Only rewards upward progress, no penalty for falling back
        if (height_addr_ >= 0 && height_coeff_ > 0.0f) {
            int current_y = read_ram_byte(static_cast<uint16_t>(height_addr_));
            if (current_y < best_y_) {
                // Reached a new height — reward proportional to progress
                int gain = best_y_ - current_y;
                result.reward += static_cast<float>(gain) * height_coeff_;
                best_y_ = current_y;
            }
        }

        // Check for life loss
        result.done = false;
        if (lives_addr_ >= 0) {
            int current_lives = read_ram_byte(static_cast<uint16_t>(lives_addr_));
            if (current_lives < previous_lives_ && previous_lives_ > 0) {
                result.done = true;
            }
            previous_lives_ = current_lives;
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
            return {ActionType::MULTI_DISCRETE, {3, 3, 2}};
        }
        return {ActionType::DISCRETE, {kNumActions}};
    }

    std::vector<uint8_t> save_state() const {
        crayon::SaveState state;
        state.cpu_state = emulator_->get_cpu_state();
        state.gate_array_state = emulator_->get_gate_array_state();
        state.memory_state = emulator_->get_memory_state();
        state.pia_state = emulator_->get_pia_state();
        state.frame_count = emulator_->get_frame_count();
        auto result = crayon::SaveStateManager::serialize_to_buffer(state);
        if (result.is_err() || !result.value.has_value()) return {};
        return result.value.value();
    }

    void load_state(const std::vector<uint8_t>& data) {
        auto result = crayon::SaveStateManager::deserialize_from_buffer(data.data(), data.size());
        if (result.is_err() || !result.value.has_value()) {
            throw StateError("Failed to load MO5 emulator state");
        }
        const auto& state = result.value.value();
        emulator_->get_cpu().set_state(state.cpu_state);
        emulator_->get_gate_array().set_state(state.gate_array_state);
        emulator_->get_memory().set_state(state.memory_state);
        emulator_->get_pia().set_state(state.pia_state);
    }

    void set_reward_mode(const std::string& mode) {
        reward_mode_ = mode;
        reward_system_ = RewardSystemFactory::create(mode, reward_params_);
        wire_memory_reward_system();
        if (reward_system_) reward_system_->reset();
    }

    std::vector<std::string> available_reward_modes() const {
        return RewardSystemFactory::available_modes();
    }

    std::string game_name() const { return rom_name_; }

    std::vector<uint8_t> read_ram() const {
        auto ms = emulator_->get_memory_state();
        // Return user_ram[32K] padded to 48K to match the old adapter layout.
        // Game profile addresses (e.g. lives at 11095) index into this buffer.
        std::vector<uint8_t> ram(kRamSize, 0);
        std::memcpy(ram.data(), ms.user_ram, sizeof(ms.user_ram));
        return ram;
    }

    uint8_t read_ram_byte(uint16_t address) const {
        // Same layout as read_ram(): user_ram starting at offset 0
        auto ms = emulator_->get_memory_state();
        if (address < sizeof(ms.user_ram)) {
            return ms.user_ram[address];
        }
        return 0;
    }

    int ram_size() const { return kRamSize; }

    void type_string(const std::string& text, int hold_frames = 3, int gap_frames = 3) {
        for (char c : text) {
            crayon::CharMapping mapping;
            if (crayon::char_to_mo5(c, mapping)) {
                auto& input = emulator_->get_input_handler();
                for (int i = 0; i < hold_frames; ++i) {
                    input.reset();
                    if (mapping.shift) input.set_key_state(crayon::MO5Key::SHIFT, true);
                    input.set_key_state(mapping.key, true);
                    emulator_->run_frame();
                    ++frame_number_;
                }
                input.reset();
            }
            for (int i = 0; i < gap_frames; ++i) {
                emulator_->run_frame();
                ++frame_number_;
            }
        }
    }

    void wait_frames(int n) {
        for (int i = 0; i < n; ++i) {
            emulator_->run_frame();
            ++frame_number_;
        }
    }

private:
    void render_frame() {
        const uint32_t* fb32 = reinterpret_cast<const uint32_t*>(emulator_->get_framebuffer());
        if (!fb32) return;
        for (int i = 0; i < kScreenWidth * kScreenHeight; ++i) {
            uint32_t pixel = fb32[i];
            rgb_buffer_[i * 3 + 0] = (pixel >> 24) & 0xFF;  // R (RGBA format)
            rgb_buffer_[i * 3 + 1] = (pixel >> 16) & 0xFF;  // G
            rgb_buffer_[i * 3 + 2] = (pixel >>  8) & 0xFF;  // B
        }
    }

    void apply_joystick(const std::vector<int>& action) {
        auto& input = emulator_->get_input_handler();
        input.reset();
        if (action[0] == 1) input.set_key_state(crayon::MO5Key::UP, true);
        if (action[0] == 2) input.set_key_state(crayon::MO5Key::DOWN, true);
        if (action[1] == 1) input.set_key_state(crayon::MO5Key::RIGHT, true);
        if (action[1] == 2) input.set_key_state(crayon::MO5Key::LEFT, true);
        if (action[2] == 1) input.set_key_state(crayon::MO5Key::SPACE, true);
    }

    void apply_discrete_action(int action) {
        auto& input = emulator_->get_input_handler();
        input.reset();
        static const crayon::MO5Key keys[] = {
            crayon::MO5Key::UP, crayon::MO5Key::DOWN,
            crayon::MO5Key::LEFT, crayon::MO5Key::RIGHT,
            crayon::MO5Key::SPACE, crayon::MO5Key::ENTER,
        };
        if (action > 0 && action <= 6) {
            input.set_key_state(keys[action - 1], true);
        }
    }

    void run_startup_sequence(const std::string& seq) {
        size_t pos = 0;
        while (pos < seq.size()) {
            size_t next = seq.find('|', pos);
            std::string cmd = seq.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
            pos = (next == std::string::npos) ? seq.size() : next + 1;
            if (cmd.substr(0, 5) == "wait:") {
                wait_frames(std::stoi(cmd.substr(5)));
            } else if (cmd.substr(0, 5) == "type:") {
                std::string text = cmd.substr(5);
                size_t p;
                while ((p = text.find("\\n")) != std::string::npos) {
                    text.replace(p, 2, "\n");
                }
                type_string(text);
            }
        }
    }

    void cache_startup_state() {
        crayon::SaveState state;
        state.cpu_state = emulator_->get_cpu_state();
        state.gate_array_state = emulator_->get_gate_array_state();
        state.memory_state = emulator_->get_memory_state();
        state.pia_state = emulator_->get_pia_state();
        state.frame_count = emulator_->get_frame_count();
        auto result = crayon::SaveStateManager::serialize_to_buffer(state);
        if (result.is_ok() && result.value.has_value()) {
            startup_state_ = result.value.value();
        }
    }

    std::unique_ptr<crayon::EmulatorCore> emulator_;
    std::string rom_name_;
    std::string reward_mode_;
    std::string action_mode_;
    int frame_number_;
    RewardParams reward_params_;
    std::unique_ptr<RewardSystem> reward_system_;
    StepResult previous_result_;
    std::vector<uint8_t> rgb_buffer_;
    int lives_addr_ = -1;
    int previous_lives_ = 0;
    int height_addr_ = -1;
    float height_coeff_ = 0.0f;
    int previous_y_ = 0;
    int best_y_ = 255;  // worst (lowest) height
    std::vector<uint8_t> startup_state_;
};

// ---------------------------------------------------------------------------
// MO5RLInterface forwarding methods
// ---------------------------------------------------------------------------

MO5RLInterface::MO5RLInterface(const std::string& rom_path,
                                const std::string& reward_mode,
                                const RewardParams& reward_params,
                                const std::string& action_mode)
    : impl_(std::make_unique<Impl>(rom_path, reward_mode, reward_params, action_mode))
{}

MO5RLInterface::~MO5RLInterface() = default;
MO5RLInterface::MO5RLInterface(MO5RLInterface&&) noexcept = default;
MO5RLInterface& MO5RLInterface::operator=(MO5RLInterface&&) noexcept = default;

StepResult MO5RLInterface::reset(int seed) { return impl_->reset(seed); }
StepResult MO5RLInterface::step(const std::vector<int>& action) { return impl_->step(action); }
ObservationSpace MO5RLInterface::observation_space() const { return impl_->observation_space(); }
ActionSpace MO5RLInterface::action_space() const { return impl_->action_space(); }
std::vector<uint8_t> MO5RLInterface::save_state() const { return impl_->save_state(); }
void MO5RLInterface::load_state(const std::vector<uint8_t>& state) { impl_->load_state(state); }
void MO5RLInterface::set_reward_mode(const std::string& mode) { impl_->set_reward_mode(mode); }
std::vector<std::string> MO5RLInterface::available_reward_modes() const { return impl_->available_reward_modes(); }
std::string MO5RLInterface::emulator_name() const { return "MO5"; }
std::string MO5RLInterface::game_name() const { return impl_->game_name(); }
std::vector<uint8_t> MO5RLInterface::read_ram() const { return impl_->read_ram(); }
uint8_t MO5RLInterface::read_ram_byte(uint16_t address) const { return impl_->read_ram_byte(address); }
int MO5RLInterface::ram_size() const { return impl_->ram_size(); }
void MO5RLInterface::type_string(const std::string& text, int hold, int gap) { impl_->type_string(text, hold, gap); }
void MO5RLInterface::wait_frames(int n) { impl_->wait_frames(n); }

}  // namespace retro_ai
