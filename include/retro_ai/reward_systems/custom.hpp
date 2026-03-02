#pragma once

#include "retro_ai/reward_system.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace retro_ai {

// Forward declaration
struct StepResult;

/// Type alias for user-provided reward callbacks.
/// The callback receives the current and previous StepResult and returns a
/// scalar reward.
using RewardCallback = std::function<float(const StepResult&, const StepResult&)>;

/// Custom reward system that delegates computation to a user-provided callback.
///
/// This allows researchers to define arbitrary reward logic without
/// sub-classing RewardSystem.  The callback receives the full StepResult
/// (observation, reward, done, truncated, info) for both the current and
/// previous steps.
class CustomRewardSystem : public RewardSystem {
public:
    /// Construct with a user-defined callback.
    /// @param callback  Function invoked on each step to compute the reward.
    ///                  Defaults to a no-op that returns 0.0f.
    explicit CustomRewardSystem(
        RewardCallback callback = [](const StepResult&, const StepResult&) {
            return 0.0f;
        });

    float compute_reward(const StepResult& current,
                         const StepResult& previous) override;

    void reset() override;
    std::string name() const override;

    /// Replace the current callback at runtime.
    void set_callback(RewardCallback callback);

private:
    RewardCallback callback_;
};

/// Composite reward system that combines multiple RewardSystem instances with
/// configurable weights.
///
/// The final reward is the weighted sum:  Σ(weight_i × system_i.compute_reward(...))
class CompositeRewardSystem : public RewardSystem {
public:
    CompositeRewardSystem() = default;

    /// Add a reward system with an associated weight.
    /// @param system  Reward system to include (ownership transferred).
    /// @param weight  Multiplicative weight applied to this system's output.
    void add(std::unique_ptr<RewardSystem> system, float weight = 1.0f);

    float compute_reward(const StepResult& current,
                         const StepResult& previous) override;

    void reset() override;
    std::string name() const override;

private:
    std::vector<std::pair<std::unique_ptr<RewardSystem>, float>> systems_;
};

}  // namespace retro_ai
