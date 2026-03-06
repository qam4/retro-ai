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

/// Helper: look up a key in the param map and convert to int.
/// Returns @p fallback if the key is absent or conversion fails.
static int param_int(const RewardParams& params,
                     const std::string& key, int fallback) {
    auto it = params.find(key);
    if (it == params.end()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

/// Helper: look up a key in the param map and convert to unsigned long.
/// Returns @p fallback if the key is absent or conversion fails.
static unsigned long param_ulong(const RewardParams& params,
                                  const std::string& key, unsigned long fallback) {
    auto it = params.find(key);
    if (it == params.end()) return fallback;
    try {
        return std::stoul(it->second);
    } catch (...) {
        return fallback;
    }
}

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

std::unique_ptr<RewardSystem> RewardSystemFactory::create(
        const std::string& mode, const RewardParams& params) {
    if (params.empty()) {
        return create(mode);
    }

    if (mode == "vision") {
        ScreenRegion region{
            param_int(params, "screen_region_x", 112),
            param_int(params, "screen_region_y", 80),
            param_int(params, "screen_region_w", 40),
            param_int(params, "screen_region_h", 14)
        };
        ObservationSpace obs{160, 240, 3, 8};
        return std::make_unique<VisionRewardSystem>(region, obs);
    }

    if (mode == "memory") {
        int count = param_int(params, "score_address_count", 0);
        std::vector<MemoryAddress> addresses;
        addresses.reserve(count);
        for (int i = 0; i < count; ++i) {
            std::string prefix = "score_address_" + std::to_string(i);
            MemoryAddress addr{};
            addr.address = static_cast<uint16_t>(
                param_ulong(params, prefix + "_addr", 0));
            addr.num_bytes = param_int(params, prefix + "_bytes", 1);
            addr.is_bcd = param_int(params, prefix + "_bcd", 0) != 0;
            addresses.push_back(addr);
        }
        return std::make_unique<MemoryRewardSystem>(std::move(addresses));
    }

    // For survival, intrinsic, custom: delegate to the parameterless factory.
    return create(mode);
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
