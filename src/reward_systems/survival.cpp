#include "retro_ai/reward_systems/survival.hpp"
#include "retro_ai/rl_interface.hpp"

namespace retro_ai {

SurvivalRewardSystem::SurvivalRewardSystem(float alive_reward,
                                           float death_penalty)
    : alive_reward_(alive_reward), death_penalty_(death_penalty) {}

float SurvivalRewardSystem::compute_reward(const StepResult& current,
                                           const StepResult& /*previous*/) {
    return current.done ? death_penalty_ : alive_reward_;
}

void SurvivalRewardSystem::reset() {
    // No internal state to reset for survival reward.
}

std::string SurvivalRewardSystem::name() const {
    return "survival";
}

}  // namespace retro_ai
