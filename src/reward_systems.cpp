/**
 * @file reward_systems.cpp
 * @brief RewardSystemFactory implementation.
 *
 * Registers all built-in reward modes and provides the factory method
 * for creating RewardSystem instances by name.
 */

#include "retro_ai/reward_system.hpp"
#include "retro_ai/reward_systems/custom.hpp"
#include "retro_ai/reward_systems/memory.hpp"
#include "retro_ai/reward_systems/intrinsic.hpp"
#include "retro_ai/reward_systems/survival.hpp"
#include "retro_ai/reward_systems/vision.hpp"
#include "retro_ai/rl_interface.hpp"

#include <algorithm>
#include <unordered_map>
#include <functional>

namespace retro_ai {

// Type alias for reward system constructors.
using Creator = std::function<std::unique_ptr<RewardSystem>()>;

/// Internal registry mapping mode names to their constructors.
static const std::unordered_map<std::string, Creator>& registry() {
    static const std::unordered_map<std::string, Creator> reg = {
        {"survival", []() { return std::make_unique<SurvivalRewardSystem>(); }},
        {"memory",   []() { return std::make_unique<MemoryRewardSystem>(); }},
        {"vision",   []() {
            // Score region for Videopac: counter at x=112, y=80, 4 chars wide
            ScreenRegion region{112, 80, 40, 14};
            ObservationSpace obs{160, 240, 3, 8};
            return std::make_unique<VisionRewardSystem>(region, obs);
        }},
        {"intrinsic", []() { return std::make_unique<IntrinsicRewardSystem>(); }},
        {"custom",    []() { return std::make_unique<CustomRewardSystem>(); }},
    };
    return reg;
}

std::unique_ptr<RewardSystem> RewardSystemFactory::create(const std::string& mode) {
    const auto& reg = registry();
    auto it = reg.find(mode);
    if (it == reg.end()) {
        return nullptr;
    }
    return it->second();
}

std::vector<std::string> RewardSystemFactory::available_modes() {
    const auto& reg = registry();
    std::vector<std::string> modes;
    modes.reserve(reg.size());
    for (const auto& [name, _] : reg) {
        modes.push_back(name);
    }
    std::sort(modes.begin(), modes.end());
    return modes;
}

}  // namespace retro_ai
