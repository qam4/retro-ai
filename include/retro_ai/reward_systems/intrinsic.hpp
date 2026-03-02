#pragma once

#include "retro_ai/reward_system.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace retro_ai {

/// Novelty detection strategy used by IntrinsicRewardSystem.
enum class NoveltyMethod {
    HASH_BASED,       ///< Count state visit frequency via observation hashing.
    EMBEDDING_BASED   ///< Placeholder for future neural-embedding approach.
};

/// Intrinsic motivation reward: computes novelty-based rewards to encourage
/// exploration.  Novel (rarely-seen) states receive higher rewards than
/// familiar ones.  Reward for a state with visit count *n* is:
///
///     reward = novelty_scale / sqrt(n)
///
/// First visit → 1.0 × scale, second visit → ~0.707 × scale, etc.
class IntrinsicRewardSystem : public RewardSystem {
public:
    /// @param method         Novelty detection strategy.
    /// @param novelty_scale  Multiplicative scale applied to the raw novelty
    ///                       score (default 1.0).
    explicit IntrinsicRewardSystem(
        NoveltyMethod method = NoveltyMethod::HASH_BASED,
        float novelty_scale = 1.0f);

    float compute_reward(const StepResult& current,
                         const StepResult& previous) override;

    void reset() override;
    std::string name() const override;

private:
    /// Hash an observation byte vector into a single size_t key.
    static std::size_t hash_observation(const std::vector<uint8_t>& obs);

    /// Compute the novelty score for a state, updating visit counts.
    float compute_novelty_score(std::size_t state_hash);

    NoveltyMethod method_;
    float novelty_scale_;
    std::unordered_map<std::size_t, int> state_visit_counts_;
};

}  // namespace retro_ai
