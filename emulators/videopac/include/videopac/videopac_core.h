#pragma once

/**
 * @file videopac_core.h
 * @brief Public interface for the Videopac (Odyssey 2) emulator core.
 *
 * This header exposes the CPU, video, audio, and memory interfaces needed
 * by the VideopacRLInterface adapter. When the real emulator submodule is
 * checked out, this header should be replaced or updated to match the
 * actual emulator API.
 *
 * Build configuration:
 *   HEADLESS_MODE=1  - Disables display output
 *   NO_SDL=1         - Removes SDL dependency
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace videopac {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Videopac screen dimensions
constexpr int SCREEN_WIDTH  = 160;
constexpr int SCREEN_HEIGHT = 200;
constexpr int SCREEN_CHANNELS = 3;  // RGB
constexpr int FRAMEBUFFER_SIZE = SCREEN_WIDTH * SCREEN_HEIGHT * SCREEN_CHANNELS;

/// Number of discrete actions (NOOP + directions + button combos)
constexpr int NUM_ACTIONS = 18;

// ---------------------------------------------------------------------------
// CPU Interface
// ---------------------------------------------------------------------------

struct CpuState {
    uint16_t pc;       // Program counter
    uint16_t sp;       // Stack pointer
    uint8_t  a;        // Accumulator
    uint8_t  flags;    // Status flags
    uint8_t  regs[8];  // General-purpose registers
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
    uint8_t vram[4096];       // Video RAM
    uint8_t palette[16 * 3];  // 16-color palette, RGB
    uint8_t registers[16];    // Video control registers
};

/// Get a pointer to the current RGB888 framebuffer (SCREEN_WIDTH * SCREEN_HEIGHT * 3 bytes).
const uint8_t* video_get_framebuffer();

/// Render the current frame into the internal framebuffer.
void video_render_frame();

/// Get the current video state snapshot.
VideoState video_get_state();

/// Restore video state from a snapshot.
void video_set_state(const VideoState& state);

// ---------------------------------------------------------------------------
// Audio Interface
// ---------------------------------------------------------------------------

struct AudioState {
    uint8_t registers[8];  // Audio control registers
    uint32_t counter;      // Audio cycle counter
};

/// Get the current audio state snapshot.
AudioState audio_get_state();

/// Restore audio state from a snapshot.
void audio_set_state(const AudioState& state);

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

/// Initialize the emulator with BIOS and ROM data.
/// Returns true on success.
bool emulator_init(const std::string& bios_path, const std::string& rom_path);

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

/// Serialize the complete emulator state (CPU + video + audio + RAM + RNG).
std::vector<uint8_t> state_save();

/// Restore the complete emulator state from serialized data.
/// Returns true on success.
bool state_load(const std::vector<uint8_t>& data);

}  // namespace videopac
