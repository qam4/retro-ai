#pragma once

#include "retro_ai/reward_system.hpp"

#include <string>

namespace retro_ai {

/// Survival-based reward: +1.0 per frame alive, penalty on death.
class SurvivalRewardSystem : public RewardSystem {
public:
    explicit SurvivalRewardSystem(float alive_reward = 1.0f,
                                   float death_penalty = -10.0f);

    float compute_reward(const StepResult& current,
                         const StepResult& previous) override;

    void reset() override;
    std::string name() const override;

private:
    float alive_reward_;
    float death_penalty_;
};

}  // namespace retro_ai
