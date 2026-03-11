#include "retro_ai/reward_systems/memory.hpp"
#include "retro_ai/rl_interface.hpp"

#include <utility>

namespace retro_ai {

MemoryRewardSystem::MemoryRewardSystem(std::vector<MemoryAddress> score_addresses,
                                       MemoryReader reader)
    : score_addresses_(std::move(score_addresses)),
      reader_(std::move(reader)),
      previous_score_(0),
      has_previous_(false) {}

float MemoryRewardSystem::compute_reward(const StepResult& /*current*/,
                                         const StepResult& /*previous*/) {
    if (!reader_ || score_addresses_.empty()) {
        return 0.0f;
    }

    int64_t current_score = read_score();

    if (!has_previous_) {
        has_previous_ = true;
        previous_score_ = current_score;
        return 0.0f;
    }

    float delta = static_cast<float>(current_score - previous_score_);
    previous_score_ = current_score;
    return delta;
}

void MemoryRewardSystem::reset() {
    previous_score_ = 0;
    has_previous_ = false;
}

std::string MemoryRewardSystem::name() const {
    return "memory";
}

void MemoryRewardSystem::set_memory_reader(MemoryReader reader) {
    reader_ = std::move(reader);
}

void MemoryRewardSystem::set_score_addresses(std::vector<MemoryAddress> addresses) {
    score_addresses_ = std::move(addresses);
}

int64_t MemoryRewardSystem::read_score() const {
    int64_t total = 0;
    for (const auto& addr : score_addresses_) {
        total += decode_value(addr);
    }
    return total;
}

int64_t MemoryRewardSystem::decode_value(const MemoryAddress& addr) const {
    int64_t raw = 0;

    if (addr.is_bcd) {
        // BCD: each byte encodes two decimal digits (high nibble × 10 + low nibble).
        for (int i = 0; i < addr.num_bytes; ++i) {
            int byte_idx = addr.little_endian
                ? (addr.num_bytes - 1 - i)
                : i;
            uint8_t byte = reader_(static_cast<uint16_t>(addr.address + byte_idx));
            raw = raw * 100 + decode_bcd_byte(byte);
        }
    } else if (addr.little_endian) {
        // Binary LE: least-significant byte at start address.
        for (int i = 0; i < addr.num_bytes; ++i) {
            uint8_t byte = reader_(static_cast<uint16_t>(addr.address + i));
            raw |= static_cast<int64_t>(byte) << (8 * i);
        }
    } else {
        // Binary BE: most-significant byte at start address.
        for (int i = 0; i < addr.num_bytes; ++i) {
            uint8_t byte = reader_(static_cast<uint16_t>(addr.address + i));
            raw = (raw << 8) | byte;
        }
    }

    return raw * addr.multiplier;
}

int MemoryRewardSystem::decode_bcd_byte(uint8_t byte) {
    int high = (byte >> 4) & 0x0F;
    int low = byte & 0x0F;
    return high * 10 + low;
}

}  // namespace retro_ai
