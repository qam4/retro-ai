/**
 * @file test_vision_ocr_preservation.cpp
 * @brief Preservation property tests for VisionRewardSystem OCR.
 *
 * **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
 *
 * Property 2: Preservation — Non-Digit and Empty Input Behavior
 *
 * These tests verify that extract_score() and compute_reward() behave
 * correctly for inputs where the bug condition does NOT hold: empty
 * observations, blank framebuffers, all-white framebuffers, random noise,
 * and reward delta / reset semantics.
 *
 * These tests MUST PASS on the current UNFIXED code — they capture
 * baseline behavior that must be preserved after the fix.
 */

#include <gtest/gtest.h>

#include "retro_ai/reward_systems/vision.hpp"
#include "retro_ai/rl_interface.hpp"

#include <cstdint>
#include <random>
#include <vector>

using namespace retro_ai;

// Videopac observation space: 160 height × 240 width × 3 channels (RGB888).
static constexpr int kObsHeight   = 160;
static constexpr int kObsWidth    = 240;
static constexpr int kObsChannels = 3;
static constexpr size_t kFramebufferSize =
    static_cast<size_t>(kObsHeight) * kObsWidth * kObsChannels;

// Default ScreenRegion used in the existing code.
static constexpr int kDefaultRegionX = 112;
static constexpr int kDefaultRegionY = 80;
static constexpr int kDefaultRegionW = 40;
static constexpr int kDefaultRegionH = 14;

// ---------------------------------------------------------------------------
// Helper: create a VisionRewardSystem with given score region.
// ---------------------------------------------------------------------------
static VisionRewardSystem make_vision_system(int rx, int ry, int rw, int rh) {
    ScreenRegion region{rx, ry, rw, rh};
    ObservationSpace obs{kObsWidth, kObsHeight, kObsChannels, 8};
    return VisionRewardSystem(region, obs);
}

// ---------------------------------------------------------------------------
// Helper: make a StepResult from a framebuffer.
// ---------------------------------------------------------------------------
static StepResult make_step(const std::vector<uint8_t>& fb) {
    StepResult s;
    s.observation = fb;
    s.reward = 0.0f;
    s.done = false;
    s.truncated = false;
    return s;
}

static StepResult make_empty_step() {
    return make_step({});
}

// ---------------------------------------------------------------------------
// Test 1: Empty observation → compute_reward() returns 0.0
//
// Validates: Requirement 3.1
// extract_score({}) returns -1 → compute_reward returns 0.0
// ---------------------------------------------------------------------------
TEST(VisionOCRPreservation, EmptyObservationReturnsZeroReward) {
    auto vision = make_vision_system(
        kDefaultRegionX, kDefaultRegionY, kDefaultRegionW, kDefaultRegionH);
    vision.reset();

    auto step = make_empty_step();
    StepResult dummy;
    dummy.observation = {};

    float reward = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(reward, 0.0f)
        << "Empty observation should produce 0.0 reward (extract_score returns -1).";
}

// ---------------------------------------------------------------------------
// Test 2: Blank (all-zero / black) framebuffer → compute_reward() returns 0.0
//
// Validates: Requirement 3.2
// All-zero pixels → extract_score returns -1 → reward = 0.0
// ---------------------------------------------------------------------------
TEST(VisionOCRPreservation, BlankFramebufferReturnsZeroReward) {
    auto vision = make_vision_system(
        kDefaultRegionX, kDefaultRegionY, kDefaultRegionW, kDefaultRegionH);
    vision.reset();

    std::vector<uint8_t> fb(kFramebufferSize, 0);
    auto step = make_step(fb);
    StepResult dummy;
    dummy.observation = {};

    float reward = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(reward, 0.0f)
        << "Blank (all-black) framebuffer should produce 0.0 reward.";
}

// ---------------------------------------------------------------------------
// Test 3: All-white framebuffer → compute_reward() returns 0.0
//
// Validates: Requirement 3.2
// All-255 pixels → Otsu threshold has no contrast → extract_score returns -1
// ---------------------------------------------------------------------------
TEST(VisionOCRPreservation, AllWhiteFramebufferReturnsZeroReward) {
    auto vision = make_vision_system(
        kDefaultRegionX, kDefaultRegionY, kDefaultRegionW, kDefaultRegionH);
    vision.reset();

    std::vector<uint8_t> fb(kFramebufferSize, 255);
    auto step = make_step(fb);
    StepResult dummy;
    dummy.observation = {};

    float reward = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(reward, 0.0f)
        << "All-white framebuffer should produce 0.0 reward (no contrast for Otsu).";
}


// ---------------------------------------------------------------------------
// Test 4: Random noise framebuffer → compute_reward() returns 0.0
//
// Validates: Requirements 3.2, 3.3
// Random uint8 noise won't match ROM templates at 0.70 threshold.
// We test multiple seeds to increase confidence.
// ---------------------------------------------------------------------------
class VisionOCRPreservationNoise : public ::testing::TestWithParam<uint32_t> {};

TEST_P(VisionOCRPreservationNoise, RandomNoiseReturnsZeroReward) {
    uint32_t seed = GetParam();

    auto vision = make_vision_system(
        kDefaultRegionX, kDefaultRegionY, kDefaultRegionW, kDefaultRegionH);
    vision.reset();

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<uint8_t> fb(kFramebufferSize);
    for (auto& pixel : fb) {
        pixel = static_cast<uint8_t>(dist(rng));
    }

    auto step = make_step(fb);
    StepResult dummy;
    dummy.observation = {};

    float reward = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(reward, 0.0f)
        << "Random noise framebuffer (seed=" << seed
        << ") should produce 0.0 reward (no digit patterns at 0.70 threshold).";
}

INSTANTIATE_TEST_SUITE_P(
    MultipleSeeds,
    VisionOCRPreservationNoise,
    ::testing::Values(42, 123, 999, 2024, 31415, 65535, 100000, 271828, 314159, 999999),
    [](const ::testing::TestParamInfo<uint32_t>& info) {
        return "Seed" + std::to_string(info.param);
    });

// ---------------------------------------------------------------------------
// Test 5: Reward delta preservation — no score on consecutive calls → 0.0
//
// Validates: Requirement 3.4
// When no score is detected on consecutive calls, compute_reward() returns 0.0.
// ---------------------------------------------------------------------------
TEST(VisionOCRPreservation, RewardDeltaPreservationNoScore) {
    auto vision = make_vision_system(
        kDefaultRegionX, kDefaultRegionY, kDefaultRegionW, kDefaultRegionH);
    vision.reset();

    std::vector<uint8_t> fb(kFramebufferSize, 0);
    auto step = make_step(fb);
    StepResult dummy;
    dummy.observation = {};

    // Multiple consecutive calls with blank framebuffer
    float r1 = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(r1, 0.0f) << "First call with blank fb should return 0.0";

    float r2 = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(r2, 0.0f) << "Second call with blank fb should return 0.0";

    float r3 = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(r3, 0.0f) << "Third call with blank fb should return 0.0";
}

// ---------------------------------------------------------------------------
// Test 6: Reset preservation — after reset(), first compute_reward() returns 0.0
//
// Validates: Requirement 3.4
// After reset(), has_previous_ is false, so first detection returns 0.0.
// ---------------------------------------------------------------------------
TEST(VisionOCRPreservation, ResetPreservation) {
    auto vision = make_vision_system(
        kDefaultRegionX, kDefaultRegionY, kDefaultRegionW, kDefaultRegionH);

    // First cycle
    vision.reset();

    std::vector<uint8_t> fb(kFramebufferSize, 0);
    auto step = make_step(fb);
    StepResult dummy;
    dummy.observation = {};

    float r1 = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(r1, 0.0f) << "First call after reset should return 0.0";

    // Reset again
    vision.reset();

    float r2 = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(r2, 0.0f) << "First call after second reset should return 0.0";

    // Third reset
    vision.reset();

    float r3 = vision.compute_reward(step, dummy);
    EXPECT_FLOAT_EQ(r3, 0.0f) << "First call after third reset should return 0.0";
}
