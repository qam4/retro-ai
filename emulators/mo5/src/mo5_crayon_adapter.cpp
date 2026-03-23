/**
 * @file mo5_crayon_adapter.cpp
 * @brief Adapter mapping the mo5_core.h interface to Crayon's EmulatorCore.
 *
 * Replaces the stub implementation with real MO5 emulation via Crayon.
 * The adapter manages a single global EmulatorCore instance and translates
 * between the flat C-style mo5:: API and Crayon's C++ object API.
 */

#include "mo5/mo5_core.h"
#include "emulator_core.h"
#include "input_handler.h"
#include "cassette_interface.h"

#include <memory>
#include <cstring>
#include <algorithm>

namespace mo5 {

// ---------------------------------------------------------------------------
// Global emulator instance
// ---------------------------------------------------------------------------

static std::unique_ptr<crayon::EmulatorCore> s_emu;
static std::string s_rom_name;
static std::string s_basic_rom_path;
static std::string s_monitor_rom_path;

// RGB888 framebuffer cache (Crayon outputs ARGB32, we convert to RGB888)
static uint8_t s_rgb_buffer[FRAMEBUFFER_SIZE] = {};

// Action-to-key mapping for RL discrete actions
// MO5 games typically use arrow keys + space/enter
static const crayon::MO5Key ACTION_KEYS[] = {
    // 0: no-op (handled specially)
    crayon::MO5Key::UP,      // 1: up
    crayon::MO5Key::DOWN,    // 2: down
    crayon::MO5Key::LEFT,    // 3: left
    crayon::MO5Key::RIGHT,   // 4: right
    crayon::MO5Key::SPACE,   // 5: space (fire/action)
    crayon::MO5Key::ENTER,   // 6: enter
};
static constexpr int NUM_BASIC_ACTIONS = 7;  // 0-6

// ---------------------------------------------------------------------------
// CPU Interface
// ---------------------------------------------------------------------------

void cpu_reset() {
    // Handled by emulator_reset()
}

int cpu_step() {
    if (!s_emu) return 0;
    s_emu->step();
    return 1;
}

CpuState cpu_get_state() {
    CpuState state = {};
    if (!s_emu) return state;
    auto cs = s_emu->get_cpu_state();
    state.pc  = cs.pc;
    state.sp  = cs.s;
    state.usp = cs.u;
    state.x   = cs.x;
    state.y   = cs.y;
    state.a   = cs.a;
    state.b   = cs.b;
    state.dp  = cs.dp;
    state.cc  = cs.cc;
    return state;
}

void cpu_set_state(const CpuState& /*state*/) {
    // Not implemented — use save/load state instead
}

// ---------------------------------------------------------------------------
// Video Interface
// ---------------------------------------------------------------------------

const uint8_t* video_get_framebuffer() {
    return s_rgb_buffer;
}

void video_render_frame() {
    if (!s_emu) return;
    // Crayon renders during run_frame(), framebuffer is already up to date.
    // Convert ARGB32 → RGB888
    const uint32_t* argb = reinterpret_cast<const uint32_t*>(s_emu->get_framebuffer());
    if (!argb) return;
    for (int i = 0; i < SCREEN_WIDTH * SCREEN_HEIGHT; ++i) {
        uint32_t pixel = argb[i];
        s_rgb_buffer[i * 3 + 0] = (pixel >> 16) & 0xFF;  // R
        s_rgb_buffer[i * 3 + 1] = (pixel >>  8) & 0xFF;  // G
        s_rgb_buffer[i * 3 + 2] = (pixel >>  0) & 0xFF;  // B
    }
}

VideoState video_get_state() {
    VideoState state = {};
    if (!s_emu) return state;
    auto gs = s_emu->get_gate_array_state();
    // GateArrayState has framebuffer[200][320], not vram
    // Copy what we can into the VideoState vram field
    return state;
}

void video_set_state(const VideoState& /*state*/) {
    // Not implemented — use save/load state instead
}

// ---------------------------------------------------------------------------
// Memory Interface
// ---------------------------------------------------------------------------

uint8_t memory_read(uint16_t address) {
    if (!s_emu) return 0xFF;
    return s_emu->get_memory().read(address);
}

void memory_write(uint16_t address, uint8_t value) {
    if (!s_emu) return;
    s_emu->get_memory().write(address, value);
}

const uint8_t* memory_get_ram(size_t& size_out) {
    if (!s_emu) {
        size_out = 0;
        return nullptr;
    }
    auto ms = s_emu->get_memory_state();
    // MO5MemoryState has video_ram[16KB] + user_ram[32KB]
    // Copy user_ram (the main 32KB at 0x2000-0x9FFF) into our buffer
    static uint8_t ram_buf[RAM_SIZE];
    std::memset(ram_buf, 0, sizeof(ram_buf));
    std::memcpy(ram_buf, ms.user_ram, std::min(sizeof(ram_buf), sizeof(ms.user_ram)));
    size_out = RAM_SIZE;
    return ram_buf;
}

size_t memory_save_ram(uint8_t* buffer, size_t buffer_size) {
    size_t size;
    const uint8_t* ram = memory_get_ram(size);
    if (!ram) return 0;
    size_t to_copy = std::min(buffer_size, size);
    std::memcpy(buffer, ram, to_copy);
    return to_copy;
}

size_t memory_load_ram(const uint8_t* /*buffer*/, size_t /*buffer_size*/) {
    // Not implemented — use save/load state instead
    return 0;
}

// ---------------------------------------------------------------------------
// Emulator Lifecycle
// ---------------------------------------------------------------------------

bool emulator_init(const std::string& rom_path) {
    // rom_path can be:
    //   1. A .k7 cassette file (needs basic + monitor ROMs set separately)
    //   2. A .mo5 cartridge file
    s_rom_name = rom_path;

    crayon::Configuration config;
    config.basic_rom_path = s_basic_rom_path;
    config.monitor_rom_path = s_monitor_rom_path;

    s_emu = std::make_unique<crayon::EmulatorCore>(config);

    // Load ROMs
    if (!s_basic_rom_path.empty() && !s_monitor_rom_path.empty()) {
        auto result = s_emu->load_roms(s_basic_rom_path, s_monitor_rom_path);
        if (result.is_err()) {
            s_emu.reset();
            return false;
        }
    }

    // Load cassette or cartridge
    if (rom_path.size() >= 3) {
        std::string ext = rom_path.substr(rom_path.size() - 3);
        if (ext == ".k7" || ext == ".K7") {
            auto& cassette = s_emu->get_cassette();
            if (!cassette.load_k7(rom_path).is_ok()) {
                s_emu.reset();
                return false;
            }
        } else if (ext == "mo5" || ext == "MO5") {
            auto result = s_emu->load_cartridge(rom_path);
            if (result.is_err()) {
                s_emu.reset();
                return false;
            }
        }
    }

    s_emu->reset();
    return true;
}

void emulator_shutdown() {
    s_emu.reset();
}

void emulator_reset(int /*seed*/) {
    if (!s_emu) return;
    s_emu->reset();
}

void emulator_step(int action) {
    if (!s_emu) return;

    auto& input = s_emu->get_input_handler();
    input.reset();

    // Map discrete action to key press
    if (action > 0 && action < NUM_BASIC_ACTIONS) {
        input.set_key_state(ACTION_KEYS[action], true);
    }

    s_emu->run_frame();
}

std::string emulator_get_rom_name() {
    return s_rom_name;
}

// ---------------------------------------------------------------------------
// State Serialization
// ---------------------------------------------------------------------------

std::vector<uint8_t> state_save() {
    // TODO: implement via Crayon's save_state
    return {};
}

bool state_load(const std::vector<uint8_t>& /*data*/) {
    // TODO: implement via Crayon's load_state
    return false;
}

// ---------------------------------------------------------------------------
// Extended API for RL integration
// ---------------------------------------------------------------------------

void set_rom_paths(const std::string& basic_path, const std::string& monitor_path) {
    s_basic_rom_path = basic_path;
    s_monitor_rom_path = monitor_path;
}

}  // namespace mo5
