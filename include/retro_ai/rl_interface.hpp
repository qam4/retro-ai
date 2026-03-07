#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef RETRO_AI_PROFILING
#include <chrono>
#endif

namespace retro_ai {

/// Per-component timing breakdown for a single frame (microseconds).
/// Only populated when built with RETRO_AI_PROFILING; all zeros otherwise.
struct FrameTimings {
    double cpu_us = 0.0;           // CPU emulation (run_frame)
    double vdc_us = 0.0;           // VDC rendering (included in run_frame for now)
    double framebuffer_us = 0.0;   // Framebuffer extraction / RGB conversion
    double reward_us = 0.0;        // Reward computation
    double total_us = 0.0;         // Total step time
};

/// Describes the dimensions and format of observations returned by the environment.
struct ObservationSpace {
    int width;
    int height;
    int channels;         // 1 for grayscale, 3 for RGB
    int bits_per_channel;  // typically 8
};

/// Specifies the type of action space.
enum class ActionType {
    DISCRETE,
    MULTI_DISCRETE,
    CONTINUOUS
};

/// Describes the valid actions an agent can take.
struct ActionSpace {
    ActionType type;
    std::vector<int> shape;  // DISCRETE: [n_actions], MULTI_DISCRETE: [n1, n2, ...], CONTINUOUS: [dim]
};

/// Contains all information returned from a single environment step.
struct StepResult {
    std::vector<uint8_t> observation;  // Flattened pixel data
    float reward;
    bool done;
    bool truncated;
    std::string info;  // JSON-encoded metadata
};

/// Abstract base class defining the contract for all emulator adapters.
class RLInterface {
public:
    virtual ~RLInterface() = default;

    // Core RL methods
    virtual StepResult reset(int seed = -1) = 0;
    virtual StepResult step(const std::vector<int>& action) = 0;

    // Space queries
    virtual ObservationSpace observation_space() const = 0;
    virtual ActionSpace action_space() const = 0;

    // State management
    virtual std::vector<uint8_t> save_state() const = 0;
    virtual void load_state(const std::vector<uint8_t>& state) = 0;

    // Reward configuration
    virtual void set_reward_mode(const std::string& mode) = 0;
    virtual std::vector<std::string> available_reward_modes() const = 0;

    // RAM inspection (for reward discovery tools)
    // Returns the game-relevant RAM as a flat byte vector.
    // Videopac: 64 bytes internal (8048) + 128 bytes external = 192 bytes.
    // Default returns empty (emulator doesn't support RAM inspection).
    virtual std::vector<uint8_t> read_ram() const { return {}; }

    // Single-byte RAM read without allocation (hot-path friendly).
    // Returns 0 for out-of-range addresses. Default returns 0.
    virtual uint8_t read_ram_byte(uint16_t address) const { return 0; }

    // Return total RAM size in bytes (for RAM-based observation space).
    // Default returns 0 (emulator doesn't support RAM inspection).
    virtual int ram_size() const { return 0; }

    // Return RAM contents as observation (flat byte vector).
    // Default calls read_ram(). Emulator-specific implementations can optimize.
    virtual std::vector<uint8_t> read_ram_observation() const { return read_ram(); }

    // Step N frames with the same action, return final observation + accumulated reward.
    // Default implementation calls step() N times.
    // Returns 0 reward for n < 1 (no steps taken).
    virtual StepResult step_n(const std::vector<int>& action, int n) {
        StepResult result;
        if (n < 1) {
            result.reward = 0.0f;
            result.done = false;
            result.truncated = false;
            return result;
        }
        float total_reward = 0.0f;
        for (int i = 0; i < n; ++i) {
            result = step(action);
            total_reward += result.reward;
            if (result.done || result.truncated) break;
        }
        result.reward = total_reward;
        return result;
    }

    // Profiling: return per-component timings from the last step() call.
    // Only meaningful when built with RETRO_AI_PROFILING; returns zeros otherwise.
    virtual FrameTimings get_last_frame_timings() const { return {}; }

    // Metadata
    virtual std::string emulator_name() const = 0;
    virtual std::string game_name() const = 0;
};

}  // namespace retro_ai
