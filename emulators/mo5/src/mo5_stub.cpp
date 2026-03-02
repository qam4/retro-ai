/**
 * @file mo5_stub.cpp
 * @brief Stub implementation of the Thomson MO5 emulator core.
 *
 * This file provides minimal implementations so the mo5_core library
 * compiles and links. Replace with the real emulator sources once the git
 * submodule is checked out.
 */

#include "mo5/mo5_core.h"

#include <cstring>
#include <algorithm>

namespace mo5 {

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

static CpuState   s_cpu   = {};
static VideoState  s_video = {};
static uint8_t     s_ram[RAM_SIZE] = {};
static uint8_t     s_framebuffer[FRAMEBUFFER_SIZE] = {};
static std::string s_rom_name;
static bool        s_initialized = false;

// ---------------------------------------------------------------------------
// CPU Interface (Motorola 6809)
// ---------------------------------------------------------------------------

void cpu_reset() {
    s_cpu = {};
}

int cpu_step() {
    // Stub: no-op, return 1 cycle
    return 1;
}

CpuState cpu_get_state() {
    return s_cpu;
}

void cpu_set_state(const CpuState& state) {
    s_cpu = state;
}

// ---------------------------------------------------------------------------
// Video Interface
// ---------------------------------------------------------------------------

const uint8_t* video_get_framebuffer() {
    return s_framebuffer;
}

void video_render_frame() {
    // Stub: framebuffer stays zeroed (black screen)
}

VideoState video_get_state() {
    return s_video;
}

void video_set_state(const VideoState& state) {
    s_video = state;
}

// ---------------------------------------------------------------------------
// Memory Interface
// ---------------------------------------------------------------------------

uint8_t memory_read(uint16_t address) {
    if (address < RAM_SIZE) {
        return s_ram[address];
    }
    return 0xFF;
}

void memory_write(uint16_t address, uint8_t value) {
    if (address < RAM_SIZE) {
        s_ram[address] = value;
    }
}

const uint8_t* memory_get_ram(size_t& size_out) {
    size_out = RAM_SIZE;
    return s_ram;
}

size_t memory_save_ram(uint8_t* buffer, size_t buffer_size) {
    size_t to_copy = std::min(buffer_size, static_cast<size_t>(RAM_SIZE));
    std::memcpy(buffer, s_ram, to_copy);
    return to_copy;
}

size_t memory_load_ram(const uint8_t* buffer, size_t buffer_size) {
    size_t to_copy = std::min(buffer_size, static_cast<size_t>(RAM_SIZE));
    std::memcpy(s_ram, buffer, to_copy);
    return to_copy;
}

// ---------------------------------------------------------------------------
// Emulator Lifecycle
// ---------------------------------------------------------------------------

bool emulator_init(const std::string& rom_path) {
    s_rom_name = rom_path;
    s_initialized = true;
    emulator_reset(-1);
    return true;
}

void emulator_shutdown() {
    s_initialized = false;
    s_rom_name.clear();
}

void emulator_reset(int seed) {
    (void)seed;
    cpu_reset();
    std::memset(&s_video, 0, sizeof(s_video));
    std::memset(s_ram, 0, sizeof(s_ram));
    std::memset(s_framebuffer, 0, sizeof(s_framebuffer));
}

void emulator_step(int action) {
    if (action < 0 || action >= NUM_ACTIONS) {
        return;
    }
    // Stub: advance one frame (no-op)
    cpu_step();
    video_render_frame();
}

std::string emulator_get_rom_name() {
    return s_rom_name;
}

// ---------------------------------------------------------------------------
// State Serialization
// ---------------------------------------------------------------------------

std::vector<uint8_t> state_save() {
    std::vector<uint8_t> data;

    // Reserve approximate size
    data.reserve(sizeof(CpuState) + sizeof(VideoState) + RAM_SIZE);

    // CPU state
    const auto* cpu_bytes = reinterpret_cast<const uint8_t*>(&s_cpu);
    data.insert(data.end(), cpu_bytes, cpu_bytes + sizeof(CpuState));

    // Video state
    const auto* vid_bytes = reinterpret_cast<const uint8_t*>(&s_video);
    data.insert(data.end(), vid_bytes, vid_bytes + sizeof(VideoState));

    // RAM
    data.insert(data.end(), s_ram, s_ram + RAM_SIZE);

    return data;
}

bool state_load(const std::vector<uint8_t>& data) {
    size_t expected = sizeof(CpuState) + sizeof(VideoState) + RAM_SIZE;
    if (data.size() < expected) {
        return false;
    }

    size_t offset = 0;

    std::memcpy(&s_cpu, data.data() + offset, sizeof(CpuState));
    offset += sizeof(CpuState);

    std::memcpy(&s_video, data.data() + offset, sizeof(VideoState));
    offset += sizeof(VideoState);

    std::memcpy(s_ram, data.data() + offset, RAM_SIZE);

    return true;
}

}  // namespace mo5
