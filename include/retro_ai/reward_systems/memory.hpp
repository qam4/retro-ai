#pragma once

#include "retro_ai/reward_system.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace retro_ai {

/// Describes a single memory location that contributes to the game score.
struct MemoryAddress {
    uint16_t address;   ///< Start address in emulator RAM.
    int num_bytes;      ///< Number of consecutive bytes (1, 2, or 4).
    bool is_bcd;        ///< True if the value is stored as binary-coded decimal.
    bool little_endian; ///< True if least-significant byte is at the start address.
                        ///< For BCD: low-order digits first. For binary: standard LE.
                        ///< Default false (big-endian / most-significant byte first).
    int64_t multiplier; ///< Positional weight applied after decoding (default 1).
                        ///< Useful for digit-array scores where each byte holds a
                        ///< single digit: {addr=80, multiplier=1000}, {addr=81,
                        ///< multiplier=100}, etc.
};

/// Callback type for reading a single byte from emulator RAM.
using MemoryReader = std::function<uint8_t(uint16_t)>;

/// Memory-based reward: reads score from configured RAM addresses and returns
/// the delta between the current and previous score each frame.
class MemoryRewardSystem : public RewardSystem {
public:
    /// Construct with score address configuration and a RAM reader callback.
    /// @param score_addresses  Locations in RAM that encode the game score.
    /// @param reader           Callback that reads one byte at a given address.
    explicit MemoryRewardSystem(std::vector<MemoryAddress> score_addresses = {},
                                MemoryReader reader = nullptr);

    float compute_reward(const StepResult& current,
                         const StepResult& previous) override;

    void reset() override;
    std::string name() const override;

    /// Replace the RAM reader callback (e.g. when the emulator is re-created).
    void set_memory_reader(MemoryReader reader);

    /// Replace the score address configuration at runtime.
    void set_score_addresses(std::vector<MemoryAddress> addresses);

private:
    /// Read the composite score from all configured addresses.
    int64_t read_score() const;

    /// Decode a single MemoryAddress entry into an integer value.
    int64_t decode_value(const MemoryAddress& addr) const;

    /// Decode a BCD byte (e.g. 0x42 → 42).
    static int decode_bcd_byte(uint8_t byte);

    std::vector<MemoryAddress> score_addresses_;
    MemoryReader reader_;
    int64_t previous_score_;
    bool has_previous_;
};

}  // namespace retro_ai
