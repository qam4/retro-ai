#include "retro_ai/reward_systems/custom.hpp"
#include "retro_ai/rl_interface.hpp"

#include <utility>

namespace retro_ai {

// ---------------------------------------------------------------------------
// CustomRewardSystem
// ---------------------------------------------------------------------------

CustomRewardSystem::CustomRewardSystem(RewardCallback callback)
    : callback_(std::move(callback)) {}

float CustomRewardSystem::compute_reward(const StepResult& current,
                                         const StepResult& previous) {
    return callback_(current, previous);
}

void CustomRewardSystem::reset() {
    // No internal state to reset — the callback is stateless from our
    // perspective.  Users who capture mutable state in their lambda are
    // responsible for resetting it externally.
}

std::string CustomRewardSystem::name() const {
    return "custom";
}

void CustomRewardSystem::set_callback(RewardCallback callback) {
    callback_ = std::move(callback);
}

// ---------------------------------------------------------------------------
// CompositeRewardSystem
// ---------------------------------------------------------------------------

void CompositeRewardSystem::add(std::unique_ptr<RewardSystem> system,
                                float weight) {
    systems_.emplace_back(std::move(system), weight);
}

float CompositeRewardSystem::compute_reward(const StepResult& current,
                                            const StepResult& previous) {
    float total = 0.0f;
    for (auto& [system, weight] : systems_) {
        total += weight * system->compute_reward(current, previous);
    }
    return total;
}

void CompositeRewardSystem::reset() {
    for (auto& [system, _] : systems_) {
        system->reset();
    }
}

std::string CompositeRewardSystem::name() const {
    return "composite";
}

}  // namespace retro_ai
