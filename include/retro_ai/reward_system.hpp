#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace retro_ai {

// Forward declaration
struct StepResult;

/// Abstract base class for all reward computation strategies.
///
/// Reward systems are pluggable components that compute scalar reward signals
/// from emulator state transitions. Implementations include survival-based,
/// memory-based, vision-based, and intrinsic motivation approaches.
class RewardSystem {
public:
    virtual ~RewardSystem() = default;

    /// Compute reward from the current and previous step results.
    /// @param current  The StepResult from the most recent step.
    /// @param previous The StepResult from the preceding step.
    /// @return Scalar reward value for this transition.
    virtual float compute_reward(const StepResult& current,
                                 const StepResult& previous) = 0;

    /// Reset any internal state (e.g. visit counts, previous scores).
    virtual void reset() = 0;

    /// Return the canonical name of this reward mode (e.g. "survival").
    virtual std::string name() const = 0;
};

/// String-based parameter map for per-game reward configuration.
using RewardParams = std::unordered_map<std::string, std::string>;

/// Factory for creating RewardSystem instances by mode name.
///
/// New reward systems are registered at compile time. The factory owns the
/// mapping from mode strings to concrete constructors.
class RewardSystemFactory {
public:
    /// Create a RewardSystem for the given mode name.
    /// @param mode  One of the strings returned by available_modes().
    /// @return A new RewardSystem instance, or nullptr if mode is unknown.
    static std::unique_ptr<RewardSystem> create(const std::string& mode);

    /// Create a RewardSystem for the given mode name with per-game parameters.
    /// @param mode   One of the strings returned by available_modes().
    /// @param params Key-value parameter map (e.g. screen_region_x, score_address_0_addr).
    /// @return A new RewardSystem instance, or nullptr if mode is unknown.
    static std::unique_ptr<RewardSystem> create(const std::string& mode,
                                                 const RewardParams& params);

    /// List all registered reward mode names.
    static std::vector<std::string> available_modes();
};

}  // namespace retro_ai
