#include "retro_ai/reward_systems/intrinsic.hpp"
#include "retro_ai/rl_interface.hpp"

#include <cmath>
#include <functional>

namespace retro_ai {

IntrinsicRewardSystem::IntrinsicRewardSystem(NoveltyMethod method,
                                             float novelty_scale)
    : method_(method), novelty_scale_(novelty_scale) {}

float IntrinsicRewardSystem::compute_reward(const StepResult& current,
                                            const StepResult& /*previous*/) {
    // Both HASH_BASED and EMBEDDING_BASED currently use the hash path.
    // EMBEDDING_BASED is a placeholder for a future neural-embedding approach.
    std::size_t h = hash_observation(current.observation);
    return compute_novelty_score(h);
}

void IntrinsicRewardSystem::reset() {
    state_visit_counts_.clear();
}

std::string IntrinsicRewardSystem::name() const {
    return "intrinsic";
}

std::size_t IntrinsicRewardSystem::hash_observation(
    const std::vector<uint8_t>& obs) {
    // Use std::hash on the raw byte range.  We combine per-byte hashes with
    // a simple shift-xor scheme so that different observations with the same
    // set of bytes in different order produce distinct hashes.
    std::size_t seed = obs.size();
    std::hash<uint8_t> hasher;
    for (auto byte : obs) {
        seed ^= hasher(byte) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

float IntrinsicRewardSystem::compute_novelty_score(std::size_t state_hash) {
    int& count = state_visit_counts_[state_hash];
    ++count;
    // reward = scale / sqrt(visit_count)
    return novelty_scale_ / std::sqrt(static_cast<float>(count));
}

}  // namespace retro_ai
