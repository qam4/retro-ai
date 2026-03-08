/**
 * @file test_vision_ocr_bug.cpp
 * @brief Bug condition exploration test for VisionRewardSystem OCR.
 *
 * Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4
 *
 * Property 1: Bug Condition — VDC Digit Detection Failure
 *
 * These tests construct synthetic framebuffers containing valid Videopac
 * digit characters rendered at the correct Intel 8245 VDC dimensions
 * (8px wide, 14 scanlines tall, 16px quad spacing) and verify that
 * extract_score() returns the correct numeric score.
 *
 * On UNFIXED code these tests MUST FAIL — failure confirms the bug exists.
 * The buggy constants (kDigitWidth=7, kDigitHeight=10, step=7) cause the
 * extraction pipeline to produce undersized, misaligned patches that cannot
 * match the ROM templates at the 0.70 acceptance threshold.
 */

#include <gtest/gtest.h>

#include "retro_ai/reward_systems/vision.hpp"
#include "retro_ai/rl_interface.hpp"

#include <cstdint>
#include <vector>

using namespace retro_ai;

// ---------------------------------------------------------------------------
// Intel 8245 VDC character ROM digit patterns (copied from vision.cpp).
// Each digit: 8 bytes (8 rows), MSB-first (bit 7 = leftmost pixel).
// Row 7 is always 0x00 (blank).
// ---------------------------------------------------------------------------
static const uint8_t videopac_digit_rom[10][8] = {
    {0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00}, // 0
    {0x18,0x38,0x18,0x18,0x18,0x18,0x3C,0x00}, // 1
    {0x3C,0x66,0x0C,0x18,0x30,0x60,0x7E,0x00}, // 2
    {0x7C,0xC6,0x06,0x3C,0x06,0xC6,0x7C,0x00}, // 3
    {0xCC,0xCC,0xCC,0xFE,0x0C,0x0C,0x0C,0x00}, // 4
    {0xFE,0xC0,0xC0,0x7C,0x06,0xC6,0x7C,0x00}, // 5
    {0x7C,0xC6,0xC0,0xFC,0xC6,0xC6,0x7C,0x00}, // 6
    {0xFE,0x06,0x0C,0x18,0x30,0x60,0xC0,0x00}, // 7
    {0x7C,0xC6,0xC6,0x7C,0xC6,0xC6,0x7C,0x00}, // 8
    {0x7C,0xC6,0xC6,0x7E,0x06,0xC6,0x7C,0x00}, // 9
};

// Videopac observation space: 160 height × 240 width × 3 channels (RGB888).
static constexpr int kObsHeight   = 160;
static constexpr int kObsWidth    = 240;
static constexpr int kObsChannels = 3;

// Correct VDC digit rendering dimensions.
static constexpr int kVDCCharWidth     = 8;   // 8 pixels wide
static constexpr int kVDCCharHeight    = 14;  // 7 ROM rows × 2 scanlines each
static constexpr int kVDCQuadSpacing   = 16;  // 8px char + 8px gap

// ---------------------------------------------------------------------------
// Helper: render a single VDC digit into a flat RGB888 framebuffer.
//
// The digit is placed at pixel position (x, y) in the framebuffer.
// Each of the 7 ROM rows (0–6) is doubled to 2 scanlines = 14 scanlines.
// Each ROM row is 8 bits, MSB-first (bit 7 = leftmost pixel).
// Foreground = white (255,255,255), background = black (0,0,0).
// Row 7 (0x00) produces 2 blank scanlines at the bottom.
// ---------------------------------------------------------------------------
static void render_vdc_digit(std::vector<uint8_t>& framebuffer,
                             int digit, int x, int y) {
    const int stride = kObsWidth * kObsChannels;

    for (int rom_row = 0; rom_row < 8; ++rom_row) {
        uint8_t pattern = videopac_digit_rom[digit][rom_row];

        // Each ROM row produces 2 scanlines (doubled).
        for (int sub = 0; sub < 2; ++sub) {
            int scanline = y + rom_row * 2 + sub;
            if (scanline < 0 || scanline >= kObsHeight) continue;

            for (int col = 0; col < 8; ++col) {
                int px = x + col;
                if (px < 0 || px >= kObsWidth) continue;

                // MSB-first: bit 7 = leftmost pixel
                bool fg = (pattern & (0x80 >> col)) != 0;
                uint8_t val = fg ? 255 : 0;

                size_t idx = static_cast<size_t>(scanline) * stride +
                             static_cast<size_t>(px) * kObsChannels;
                framebuffer[idx + 0] = val;  // R
                framebuffer[idx + 1] = val;  // G
                framebuffer[idx + 2] = val;  // B
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: create a blank (all-black) framebuffer.
// ---------------------------------------------------------------------------
static std::vector<uint8_t> make_blank_framebuffer() {
    return std::vector<uint8_t>(
        static_cast<size_t>(kObsHeight) * kObsWidth * kObsChannels, 0);
}

// ---------------------------------------------------------------------------
// Helper: create a VisionRewardSystem with a score region starting at (rx, ry)
// with given width and height.
// ---------------------------------------------------------------------------
static VisionRewardSystem make_vision_system(int rx, int ry, int rw, int rh) {
    ScreenRegion region{rx, ry, rw, rh};
    ObservationSpace obs{kObsWidth, kObsHeight, kObsChannels, 8};
    return VisionRewardSystem(region, obs);
}

// ---------------------------------------------------------------------------
// Test: Single digit detection (digits 0–9)
//
// Render each digit at offset 0 in a ScreenRegion(0, 0, 48, 14) and verify
// extract_score() returns the correct digit value.
//
// On UNFIXED code: kDigitWidth=7, kDigitHeight=10, step=7 → the 7×10 patch
// misses the rightmost column and bottom 4 scanlines → match score falls
// below 0.70 → returns -1 instead of the digit.
// ---------------------------------------------------------------------------
class VisionOCRBugTest : public ::testing::TestWithParam<int> {};

TEST_P(VisionOCRBugTest, SingleDigitDetection) {
    int digit = GetParam();

    // Score region at (0, 0) with width=48 (3 digits × 16px), height=14
    auto vision = make_vision_system(0, 0, 48, kVDCCharHeight);

    auto fb = make_blank_framebuffer();
    render_vdc_digit(fb, digit, 0, 0);

    // Use compute_reward with a StepResult containing our synthetic framebuffer.
    // On first call with has_previous_=false, compute_reward returns 0.0 but
    // internally calls extract_score(). We need to call it twice to get a delta.
    //
    // Actually, we need to test extract_score() directly. Since it's private,
    // we test through compute_reward():
    //   - First call: detects score, stores as previous, returns 0.0
    //   - Second call with blank fb: detects -1 (no score), returns 0.0
    //
    // Better approach: call compute_reward twice with the SAME framebuffer.
    //   - First call: detects digit, previous_score_ = digit, returns 0.0
    //   - Second call: detects digit again, delta = digit - digit = 0.0
    //
    // To actually verify detection, we need two different scores:
    //   - First call with blank fb → extract_score returns -1 → reward = 0.0
    //   - Reset, then call with digit fb → extract_score returns digit
    //     → first detection → reward = 0.0
    //   - Call again with different digit → reward = delta
    //
    // Simplest: render digit d, then render digit 0.
    //   - Call 1 (digit d): first detection, previous = d, reward = 0.0
    //   - Call 2 (digit 0): detected = 0, delta = 0 - d = -d
    //   If d != 0, reward should be -d (negative).
    //   If d == 0, reward should be 0.
    //
    // But if extract_score returns -1 (bug), both calls return 0.0.
    //
    // Strategy: call with digit d first, then with a DIFFERENT known digit.
    // If detection works, the second call produces a non-zero delta.
    // If detection fails (bug), both calls return 0.0.

    vision.reset();

    StepResult step1;
    step1.observation = fb;
    step1.reward = 0.0f;
    step1.done = false;
    step1.truncated = false;

    StepResult dummy;
    dummy.observation = {};
    dummy.reward = 0.0f;
    dummy.done = false;
    dummy.truncated = false;

    // First call: should detect digit d, store as previous, return 0.0
    float r1 = vision.compute_reward(step1, dummy);
    EXPECT_FLOAT_EQ(r1, 0.0f) << "First detection should return 0.0 delta";

    // Second call with a different score to verify detection worked.
    // Render a different digit to get a non-zero delta.
    int other_digit = (digit + 5) % 10;  // guaranteed different from digit
    auto fb2 = make_blank_framebuffer();
    render_vdc_digit(fb2, other_digit, 0, 0);

    StepResult step2;
    step2.observation = fb2;
    step2.reward = 0.0f;
    step2.done = false;
    step2.truncated = false;

    float r2 = vision.compute_reward(step2, dummy);

    // If extract_score works correctly:
    //   r2 = other_digit - digit
    // If extract_score is broken (returns -1 for both):
    //   r2 = 0.0 (no score detected either time)
    float expected_delta = static_cast<float>(other_digit - digit);
    EXPECT_FLOAT_EQ(r2, expected_delta)
        << "For digit " << digit << " → " << other_digit
        << ": expected delta " << expected_delta << " but got " << r2
        << ". This indicates extract_score() failed to detect VDC digits "
        << "(likely due to incorrect kDigitWidth/kDigitHeight/step constants).";
}

INSTANTIATE_TEST_SUITE_P(
    AllDigits,
    VisionOCRBugTest,
    ::testing::Range(0, 10),
    [](const ::testing::TestParamInfo<int>& info) {
        return "Digit" + std::to_string(info.param);
    });

// ---------------------------------------------------------------------------
// Test: Multi-digit score detection
//
// Render "123" at offsets 0, 16, 32 with 16px quad spacing in a
// ScreenRegion(0, 0, 48, 14) and verify extract_score() returns 123.
//
// On UNFIXED code: step=7 means windows land at offsets 0, 7, 14, 21, 28, 35
// which never align with the actual digit positions at 0, 16, 32 → returns -1.
// ---------------------------------------------------------------------------
TEST(VisionOCRBugMultiDigit, ThreeDigitScore123) {
    auto vision = make_vision_system(0, 0, 48, kVDCCharHeight);

    auto fb = make_blank_framebuffer();
    render_vdc_digit(fb, 1, 0 * kVDCQuadSpacing, 0);   // '1' at x=0
    render_vdc_digit(fb, 2, 1 * kVDCQuadSpacing, 0);   // '2' at x=16
    render_vdc_digit(fb, 3, 2 * kVDCQuadSpacing, 0);   // '3' at x=32

    vision.reset();

    StepResult step1;
    step1.observation = fb;
    step1.reward = 0.0f;
    step1.done = false;
    step1.truncated = false;

    StepResult dummy;
    dummy.observation = {};

    // First call: detects 123, stores as previous, returns 0.0
    float r1 = vision.compute_reward(step1, dummy);
    EXPECT_FLOAT_EQ(r1, 0.0f);

    // Second call with score "456" to verify via delta
    auto fb2 = make_blank_framebuffer();
    render_vdc_digit(fb2, 4, 0 * kVDCQuadSpacing, 0);
    render_vdc_digit(fb2, 5, 1 * kVDCQuadSpacing, 0);
    render_vdc_digit(fb2, 6, 2 * kVDCQuadSpacing, 0);

    StepResult step2;
    step2.observation = fb2;
    step2.reward = 0.0f;
    step2.done = false;
    step2.truncated = false;

    float r2 = vision.compute_reward(step2, dummy);

    // If detection works: 456 - 123 = 333
    // If broken: 0.0 (no scores detected)
    EXPECT_FLOAT_EQ(r2, 333.0f)
        << "Expected delta 456-123=333 but got " << r2
        << ". Multi-digit detection failed — likely due to incorrect "
        << "sliding window step (7 instead of 16px quad spacing).";
}

// ---------------------------------------------------------------------------
// Test: Score "00" detection (edge case — zero score)
//
// Render "00" at offsets 0, 16 and verify extract_score() returns 0 (not -1).
// On UNFIXED code: even the first digit at offset 0 is cropped to 7×10
// instead of 8×14, so the partial "0" pattern doesn't match → returns -1.
// ---------------------------------------------------------------------------
TEST(VisionOCRBugMultiDigit, TwoDigitScore00) {
    auto vision = make_vision_system(0, 0, 48, kVDCCharHeight);

    auto fb = make_blank_framebuffer();
    render_vdc_digit(fb, 0, 0 * kVDCQuadSpacing, 0);   // '0' at x=0
    render_vdc_digit(fb, 0, 1 * kVDCQuadSpacing, 0);   // '0' at x=16

    vision.reset();

    StepResult step1;
    step1.observation = fb;
    step1.reward = 0.0f;
    step1.done = false;
    step1.truncated = false;

    StepResult dummy;
    dummy.observation = {};

    // First call: should detect 00 (= 0), store as previous, return 0.0
    float r1 = vision.compute_reward(step1, dummy);
    EXPECT_FLOAT_EQ(r1, 0.0f);

    // Second call with "05" to verify "00" was actually detected (not -1)
    auto fb2 = make_blank_framebuffer();
    render_vdc_digit(fb2, 0, 0 * kVDCQuadSpacing, 0);
    render_vdc_digit(fb2, 5, 1 * kVDCQuadSpacing, 0);

    StepResult step2;
    step2.observation = fb2;
    step2.reward = 0.0f;
    step2.done = false;
    step2.truncated = false;

    float r2 = vision.compute_reward(step2, dummy);

    // If detection works: 5 - 0 = 5
    // If broken (both return -1): 0.0
    EXPECT_FLOAT_EQ(r2, 5.0f)
        << "Expected delta 05-00=5 but got " << r2
        << ". Score '00' was not detected — the 7×10 patch is too small "
        << "to match the 8×14 VDC digit '0'.";
}
