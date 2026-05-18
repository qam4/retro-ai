// Regression test for v2 -> v3 save-state backward compatibility.
//
// Background
// ----------
// The Crayon SaveStateManager bumped to version 3 in two steps:
//
//   * 850d1d7 — added master clock state and cassette cycle/block fields
//   * 7274687 — added extended audio fields (cycle_counter, prev_sample, ...)
//
// `read_master_clock` is correctly gated by `version >= 3`, but the audio
// and cassette readers initially consumed the new fields unconditionally.
// That made the deserialiser over-read v2 buffers by 39 bytes (audio) and
// 29 bytes (cassette), corrupting everything that followed and producing
// "Failed to load MO5 emulator state" at the Python boundary.
//
// This test constructs a synthetic v2-on-the-wire buffer (matching the
// pre-v3 byte layout exactly) and verifies that `deserialize_from_buffer`
// reads it without error and recovers the user-RAM payload byte-for-byte.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "savestate.h"

namespace {

// Match crayon's CRC32 (poly 0xEDB88320, init 0xFFFFFFFF, final XOR).
uint32_t crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

void put_u8(std::vector<uint8_t>& v, uint8_t x) { v.push_back(x); }
void put_u16(std::vector<uint8_t>& v, uint16_t x) {
    put_u8(v, x & 0xFF); put_u8(v, (x >> 8) & 0xFF);
}
void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    put_u16(v, x & 0xFFFF); put_u16(v, (x >> 16) & 0xFFFF);
}
void put_u64(std::vector<uint8_t>& v, uint64_t x) {
    put_u32(v, x & 0xFFFFFFFF); put_u32(v, (x >> 32) & 0xFFFFFFFF);
}
void put_i16(std::vector<uint8_t>& v, int16_t x) {
    put_u16(v, static_cast<uint16_t>(x));
}

constexpr uint32_t MAGIC = 0x4D4F3543;  // 'MO5C' little-endian
constexpr int KEY_COUNT = 58;

std::vector<uint8_t> build_v2_buffer(uint8_t user_ram_marker) {
    std::vector<uint8_t> w;

    // Header
    put_u32(w, MAGIC);
    put_u32(w, /*version=*/2);

    // CPU: a, b, x, y, u, s, pc, dp, cc, clock_cycles, six bools
    put_u8(w, 0x12); put_u8(w, 0x34);
    put_u16(w, 0xABCD); put_u16(w, 0x1111);
    put_u16(w, 0x2222); put_u16(w, 0x3333);
    put_u16(w, 0x44AB); put_u8(w, 0x55); put_u8(w, 0x66);
    put_u64(w, 0x1234567890ABCDEFull);
    for (int i = 0; i < 6; ++i) put_u8(w, 0);

    // GateArray
    put_u16(w, 100); put_u16(w, 50);
    put_u64(w, 999);
    put_u8(w, 1); put_u8(w, 0);
    put_u8(w, 7);

    // Memory: 0x4000 video_ram, 0x8000 user_ram, 3 bools, vec(cart_rom)
    for (int i = 0; i < 0x4000; ++i) put_u8(w, 0);
    for (int i = 0; i < 0x8000; ++i) put_u8(w, user_ram_marker);
    put_u8(w, 0); put_u8(w, 1); put_u8(w, 1);
    put_u32(w, 0); // empty cart vec

    // PIA: dra ddra cra drb ddrb crb output_latch_a output_latch_b
    //      input_pins_a input_pins_b irqa1 irqa2 irqb1 irqb2 (14 bytes)
    for (int i = 0; i < 14; ++i) put_u8(w, 0);

    // Audio v2: buzzer_state(bool=1) sample_accumulator(u32=4) host_sample_rate(u32=4) = 9 bytes
    put_u8(w, 1);
    put_u32(w, 0xDEADBEEF);
    put_u32(w, 44100);

    // Input v2: 58 keys + 2 ports * 5 bools
    for (int i = 0; i < KEY_COUNT; ++i) put_u8(w, 0);
    for (int p = 0; p < 2; ++p) for (int b = 0; b < 5; ++b) put_u8(w, 0);

    // LightPen: 2 i16 + 3 bools
    put_i16(w, 0); put_i16(w, 0);
    put_u8(w, 0); put_u8(w, 0); put_u8(w, 0);

    // Cassette v2: vec k7_data, u32 read_pos, u8 bit_pos, 2 bools, vec record_buf
    put_u32(w, 0);  // empty k7
    put_u32(w, 0);
    put_u8(w, 0);
    put_u8(w, 0); put_u8(w, 0);
    put_u32(w, 0);  // empty record buffer

    // frame_count
    put_u64(w, 999);

    // CRC32 of everything before
    uint32_t crc = crc32(w.data(), w.size());
    put_u32(w, crc);
    return w;
}

}  // namespace

TEST(SaveStateV2Compat, V2BufferLoadsCleanly) {
    auto buf = build_v2_buffer(/*user_ram_marker=*/0xAB);
    auto result = crayon::SaveStateManager::deserialize_from_buffer(
        buf.data(), buf.size());
    ASSERT_FALSE(result.is_err()) << "v2 deserialise failed: "
                                  << result.error;
    ASSERT_TRUE(result.value.has_value());
    const auto& state = result.value.value();

    EXPECT_EQ(state.version, 2u);
    EXPECT_EQ(state.cpu_state.pc, 0x44ABu);
    EXPECT_EQ(state.cpu_state.clock_cycles, 0x1234567890ABCDEFull);
    EXPECT_EQ(state.gate_array_state.frame_number, 999u);

    // user_ram should round-trip the marker byte at every position.
    for (size_t i = 0; i < sizeof(state.memory_state.user_ram); ++i) {
        ASSERT_EQ(state.memory_state.user_ram[i], 0xAB)
            << "user_ram corruption at offset " << i;
    }

    // v3-only fields should be zero-defaulted, not garbage from
    // over-reading subsequent bytes.
    EXPECT_EQ(state.audio_state.cycle_counter, 0u);
    EXPECT_EQ(state.audio_state.prev_sample, 0);
    EXPECT_EQ(state.audio_state.write_pos, 0u);
    EXPECT_EQ(state.cassette_state.play_start_cycle, 0u);
    EXPECT_EQ(state.cassette_state.current_block, 0u);
    EXPECT_EQ(state.cassette_state.fast_bit_pos, 0u);
    // master_clock is gated by version >= 3, so it stays default-init.

    // The basic v2 audio fields we did populate must survive intact.
    EXPECT_TRUE(state.audio_state.buzzer_state);
    EXPECT_EQ(state.audio_state.sample_accumulator, 0xDEADBEEFu);
    EXPECT_EQ(state.audio_state.host_sample_rate, 44100u);

    EXPECT_EQ(state.frame_count, 999u);
}
