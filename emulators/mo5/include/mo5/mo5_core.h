#pragma once

/**
 * @file mo5_core.h
 * @brief Public interface for the Thomson MO5 emulator core.
 *
 * This header exposes the CPU, video, and memory interfaces needed
 * by the MO5RLInterface adapter. When the real emulator submodule is
 * checked out, this header should be replaced or updated to match the
 * actual emulator API.
 *
 * The Thomson MO5 is a French 8-bit home computer from 1984, featuring
 * a Motorola 6809 CPU, 48KB RAM, and 320x200 16-color graphics.
 * Software is loaded via cassette tapes and cartridges.
 *
 * Build configuration:
 *   HEADLESS_MODE=1  - Disables display output
 *   NO_SDL=1         - Removes SDL dependency
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace mo5 {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// MO5 screen dimensions (320x200, 16 colors)
constexpr int SCREEN_WIDTH  = 320;
constexpr int SCREEN_HEIGHT = 200;
constexpr int SCREEN_CHANNELS = 3;  // RGB
constexpr int FRAMEBUFFER_SIZE = SCREEN_WIDTH * SCREEN_HEIGHT * SCREEN_CHANNELS;

/// Number of discrete actions (keyboard-based: A-Z, 0-9, arrows, specials)
constexpr int NUM_ACTIONS = 60;

/// MO5 RAM size: 48KB
constexpr size_t RAM_SIZE = 0xC000;

// ---------------------------------------------------------------------------
// CPU Interface (Motorola 6809)
// ---------------------------------------------------------------------------

struct CpuState {
    uint16_t pc;       // Program counter
    uint16_t sp;       // Stack pointer (S)
    uint16_t usp;      // User stack pointer (U)
    uint16_t x;        // Index register X
    uint16_t y;        // Index register Y
    uint8_t  a;        // Accumulator A
    uint8_t  b;        // Accumulator B
    uint8_t  dp;       // Direct page register
    uint8_t  cc;       // Condition code register
};

/// Reset the CPU to its initial state.
void cpu_reset();

/// Execute a single CPU instruction. Returns the number of cycles consumed.
int cpu_step();

/// Get the current CPU state snapshot.
CpuState cpu_get_state();

/// Restore CPU state from a snapshot.
void cpu_set_state(const CpuState& state);

// ---------------------------------------------------------------------------
// Video Interface
// ---------------------------------------------------------------------------

struct VideoState {
    uint8_t vram[8192];       // Video RAM (8KB)
    uint8_t palette[16 * 3];  // 16-color palette, RGB
    uint8_t registers[8];     // Video control registers
};

/// Get a pointer to the current RGB888 framebuffer (320 * 200 * 3 bytes).
const uint8_t* video_get_framebuffer();

/// Render the current frame into the internal framebuffer.
void video_render_frame();

/// Get the current video state snapshot.
VideoState video_get_state();

/// Restore video state from a snapshot.
void video_set_state(const VideoState& state);

// ---------------------------------------------------------------------------
// Memory Interface
// ---------------------------------------------------------------------------

/// Read a byte from the emulator's address space.
uint8_t memory_read(uint16_t address);

/// Write a byte to the emulator's address space.
void memory_write(uint16_t address, uint8_t value);

/// Get a pointer to the raw RAM buffer and its size.
const uint8_t* memory_get_ram(size_t& size_out);

/// Bulk-read RAM into a caller-provided buffer. Returns bytes copied.
size_t memory_save_ram(uint8_t* buffer, size_t buffer_size);

/// Bulk-write RAM from a caller-provided buffer. Returns bytes written.
size_t memory_load_ram(const uint8_t* buffer, size_t buffer_size);

// ---------------------------------------------------------------------------
// Emulator Lifecycle
// ---------------------------------------------------------------------------

/// Initialize the emulator with a ROM or tape file.
/// Returns true on success.
bool emulator_init(const std::string& rom_path);

/// Shut down the emulator and free resources.
void emulator_shutdown();

/// Reset the emulator to power-on state. If seed >= 0, seeds the RNG.
void emulator_reset(int seed = -1);

/// Advance the emulator by one frame, applying the given action index [0, NUM_ACTIONS).
void emulator_step(int action);

/// Get the name of the currently loaded ROM.
std::string emulator_get_rom_name();

// ---------------------------------------------------------------------------
// State Serialization
// ---------------------------------------------------------------------------

/// Serialize the complete emulator state (CPU + video + RAM + RNG).
std::vector<uint8_t> state_save();

/// Restore the complete emulator state from serialized data.
/// Returns true on success.
bool state_load(const std::vector<uint8_t>& data);

}  // namespace mo5
