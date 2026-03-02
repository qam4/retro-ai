#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace retro_ai {

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

    // Metadata
    virtual std::string emulator_name() const = 0;
    virtual std::string game_name() const = 0;
};

}  // namespace retro_ai
