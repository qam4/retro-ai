/**
 * @file videopac_stub.cpp
 * @brief Stub implementation of the Videopac emulator core.
 *
 * This file provides minimal implementations so the videopac_core library
 * compiles and links. Replace with the real emulator sources once the git
 * submodule is checked out.
 */

#include "videopac/videopac_core.h"

#include <cstring>
#include <algorithm>

namespace videopac {

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

static CpuState   s_cpu   = {};
static VideoState  s_video = {};
static AudioState  s_audio = {};
static uint8_t     s_ram[4096] = {};
static uint8_t     s_framebuffer[FRAMEBUFFER_SIZE] = {};
static std::string s_rom_name;
static bool        s_initialized = false;

// ---------------------------------------------------------------------------
// CPU Interface
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
// Audio Interface
// ---------------------------------------------------------------------------

AudioState audio_get_state() {
    return s_audio;
}

void audio_set_state(const AudioState& state) {
    s_audio = state;
}

// ---------------------------------------------------------------------------
// Memory Interface
// ---------------------------------------------------------------------------

uint8_t memory_read(uint16_t address) {
    if (address < sizeof(s_ram)) {
        return s_ram[address];
    }
    return 0xFF;
}

void memory_write(uint16_t address, uint8_t value) {
    if (address < sizeof(s_ram)) {
        s_ram[address] = value;
    }
}

const uint8_t* memory_get_ram(size_t& size_out) {
    size_out = sizeof(s_ram);
    return s_ram;
}

size_t memory_save_ram(uint8_t* buffer, size_t buffer_size) {
    size_t to_copy = std::min(buffer_size, sizeof(s_ram));
    std::memcpy(buffer, s_ram, to_copy);
    return to_copy;
}

size_t memory_load_ram(const uint8_t* buffer, size_t buffer_size) {
    size_t to_copy = std::min(buffer_size, sizeof(s_ram));
    std::memcpy(s_ram, buffer, to_copy);
    return to_copy;
}

// ---------------------------------------------------------------------------
// Emulator Lifecycle
// ---------------------------------------------------------------------------

bool emulator_init(const std::string& bios_path, const std::string& rom_path) {
    (void)bios_path;  // Stub ignores paths
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
    std::memset(&s_audio, 0, sizeof(s_audio));
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
    data.reserve(sizeof(CpuState) + sizeof(VideoState) + sizeof(AudioState) + sizeof(s_ram));

    // CPU state
    const auto* cpu_bytes = reinterpret_cast<const uint8_t*>(&s_cpu);
    data.insert(data.end(), cpu_bytes, cpu_bytes + sizeof(CpuState));

    // Video state
    const auto* vid_bytes = reinterpret_cast<const uint8_t*>(&s_video);
    data.insert(data.end(), vid_bytes, vid_bytes + sizeof(VideoState));

    // Audio state
    const auto* aud_bytes = reinterpret_cast<const uint8_t*>(&s_audio);
    data.insert(data.end(), aud_bytes, aud_bytes + sizeof(AudioState));

    // RAM
    data.insert(data.end(), s_ram, s_ram + sizeof(s_ram));

    return data;
}

bool state_load(const std::vector<uint8_t>& data) {
    size_t expected = sizeof(CpuState) + sizeof(VideoState) + sizeof(AudioState) + sizeof(s_ram);
    if (data.size() < expected) {
        return false;
    }

    size_t offset = 0;

    std::memcpy(&s_cpu, data.data() + offset, sizeof(CpuState));
    offset += sizeof(CpuState);

    std::memcpy(&s_video, data.data() + offset, sizeof(VideoState));
    offset += sizeof(VideoState);

    std::memcpy(&s_audio, data.data() + offset, sizeof(AudioState));
    offset += sizeof(AudioState);

    std::memcpy(s_ram, data.data() + offset, sizeof(s_ram));

    return true;
}

}  // namespace videopac
