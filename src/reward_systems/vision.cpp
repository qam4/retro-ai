#include "retro_ai/reward_systems/vision.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace retro_ai {

// ---------------------------------------------------------------------------
// Hardcoded 7×10 binary digit templates (legacy / fallback).
// Each template is a 70-element array stored row-major.
// 1 = foreground, 0 = background.
// ---------------------------------------------------------------------------
// clang-format off
const std::array<std::array<uint8_t, VisionRewardSystem::kDigitWidth *
                                     VisionRewardSystem::kDigitHeight>,
                 VisionRewardSystem::kNumDigits>
    VisionRewardSystem::digit_templates_ = {{
    // 0
    {{
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
    }},
    // 1
    {{
        0,0,0,1,1,0,0,
        0,0,1,1,1,0,0,
        0,1,1,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,1,1,1,1,1,0,
    }},
    // 2
    {{
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,1,1,0,
        0,0,0,1,1,0,0,
        0,0,1,1,0,0,0,
        0,1,1,0,0,0,0,
        1,1,0,0,0,0,0,
        1,1,0,0,0,1,1,
        1,1,1,1,1,1,1,
    }},
    // 3
    {{
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,1,1,1,1,0,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
    }},
    // 4
    {{
        0,0,0,0,1,1,0,
        0,0,0,1,1,1,0,
        0,0,1,1,1,1,0,
        0,1,1,0,1,1,0,
        1,1,0,0,1,1,0,
        1,1,1,1,1,1,1,
        0,0,0,0,1,1,0,
        0,0,0,0,1,1,0,
        0,0,0,0,1,1,0,
        0,0,0,0,1,1,0,
    }},
    // 5
    {{
        1,1,1,1,1,1,1,
        1,1,0,0,0,0,0,
        1,1,0,0,0,0,0,
        1,1,1,1,1,1,0,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
    }},
    // 6
    {{
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        1,1,0,0,0,0,0,
        1,1,0,0,0,0,0,
        1,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
    }},
    // 7
    {{
        1,1,1,1,1,1,1,
        1,1,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,1,1,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
        0,0,0,1,1,0,0,
    }},
    // 8
    {{
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
    }},
    // 9
    {{
        0,1,1,1,1,1,0,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,
        1,1,0,0,0,1,1,
        0,1,1,1,1,1,0,
    }},
}};
// clang-format on

// ---------------------------------------------------------------------------
// Intel 8245 VDC character ROM digit patterns (0-9).
// Each digit is 8 bytes (8 rows), MSB-first (bit 7 = leftmost pixel).
// Only 7 of 8 columns are used; row 7 is always 0x00 (blank).
// On screen each row is doubled (14 scanlines per character).
// ---------------------------------------------------------------------------
// clang-format off
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
// clang-format on

// ---------------------------------------------------------------------------
// Construction / reset
// ---------------------------------------------------------------------------

VisionRewardSystem::VisionRewardSystem(const ScreenRegion& score_region,
                                       const ObservationSpace& obs_space)
    : score_region_(score_region),
      obs_space_(obs_space),
      previous_score_(0),
      has_previous_(false) {}

void VisionRewardSystem::reset() {
    previous_score_ = 0;
    has_previous_ = false;
}

std::string VisionRewardSystem::name() const {
    return "vision";
}

void VisionRewardSystem::load_digit_templates(
    const std::string& /*template_dir*/) {
    // Templates are currently sourced from the Intel 8245 character ROM.
    // This hook exists so that future versions can load custom digit bitmaps
    // for non-Videopac emulators.
}

// ---------------------------------------------------------------------------
// Reward computation
// ---------------------------------------------------------------------------

float VisionRewardSystem::compute_reward(const StepResult& current,
                                         const StepResult& /*previous*/) {
    int detected = extract_score(current.observation);

    if (detected < 0) {
        // No score visible – return neutral reward.
        return 0.0f;
    }

    int64_t current_score = static_cast<int64_t>(detected);

    if (!has_previous_) {
        has_previous_ = true;
        previous_score_ = current_score;
        return 0.0f;
    }

    float delta = static_cast<float>(current_score - previous_score_);
    previous_score_ = current_score;
    return delta;
}

// ---------------------------------------------------------------------------
// Score extraction helpers
// ---------------------------------------------------------------------------

std::vector<uint8_t> VisionRewardSystem::crop_to_grayscale(
    const std::vector<uint8_t>& observation) const {
    const int stride = obs_space_.width * obs_space_.channels;
    std::vector<uint8_t> gray(
        static_cast<size_t>(score_region_.width) * score_region_.height);

    for (int row = 0; row < score_region_.height; ++row) {
        int src_y = score_region_.y + row;
        if (src_y < 0 || src_y >= obs_space_.height) continue;

        for (int col = 0; col < score_region_.width; ++col) {
            int src_x = score_region_.x + col;
            if (src_x < 0 || src_x >= obs_space_.width) continue;

            size_t idx =
                static_cast<size_t>(src_y) * stride +
                static_cast<size_t>(src_x) * obs_space_.channels;

            if (idx + 2 >= observation.size()) continue;

            uint8_t r = observation[idx];
            uint8_t g = observation[idx + 1];
            uint8_t b = observation[idx + 2];

            uint8_t lum = static_cast<uint8_t>(
                0.299f * r + 0.587f * g + 0.114f * b);

            gray[static_cast<size_t>(row) * score_region_.width + col] = lum;
        }
    }
    return gray;
}

/// Binarize a grayscale patch using Otsu's method for automatic thresholding.
/// Much more robust than mean-based thresholding when the foreground/background
/// ratio varies across patches.
static uint8_t otsu_threshold(const std::vector<uint8_t>& patch) {
    // Build histogram
    int hist[256] = {};
    for (auto v : patch) hist[v]++;

    int total = static_cast<int>(patch.size());
    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) sum += i * hist[i];

    float sum_bg = 0.0f;
    int weight_bg = 0;
    float max_variance = 0.0f;
    uint8_t best_t = 0;

    for (int t = 0; t < 256; ++t) {
        weight_bg += hist[t];
        if (weight_bg == 0) continue;
        int weight_fg = total - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += t * hist[t];
        float mean_bg = sum_bg / weight_bg;
        float mean_fg = (sum - sum_bg) / weight_fg;
        float diff = mean_bg - mean_fg;
        float variance = static_cast<float>(weight_bg) * weight_fg * diff * diff;

        if (variance > max_variance) {
            max_variance = variance;
            best_t = static_cast<uint8_t>(t);
        }
    }
    return best_t;
}

/// Match a grayscale patch against the Intel 8245 character ROM digit patterns.
/// The patch is expected to be (patch_w × patch_h) pixels containing one digit.
/// The ROM pattern is 8×7 pixels (row 7 is blank) rendered with doubled scanlines
/// (so 8×14 on screen).  We scale the ROM pattern to the patch dimensions and
/// compare using normalised cross-correlation.
static std::pair<int, float> match_videopac_digit(
    const std::vector<uint8_t>& patch, int patch_w, int patch_h) {

    // Binarize patch with Otsu
    uint8_t threshold = otsu_threshold(patch);
    std::vector<uint8_t> binary(patch.size());
    for (size_t i = 0; i < patch.size(); ++i) {
        binary[i] = (patch[i] > threshold) ? 1 : 0;
    }

    int best_digit = -1;
    float best_score = 0.0f;
    constexpr float kAcceptThreshold = 0.70f;

    for (int d = 0; d < 10; ++d) {
        int matches = 0;
        int total = patch_w * patch_h;

        for (int py = 0; py < patch_h; ++py) {
            // Map patch row to ROM row (0-6, skip row 7 which is blank).
            // Videopac doubles each row, so 14 scanlines map to 7 ROM rows.
            int rom_row = (py * 7) / patch_h;
            if (rom_row > 6) rom_row = 6;
            uint8_t pattern = videopac_digit_rom[d][rom_row];

            for (int px = 0; px < patch_w; ++px) {
                // Map patch column to ROM column (0-7, MSB-first)
                int rom_col = (px * 8) / patch_w;
                if (rom_col > 7) rom_col = 7;
                uint8_t rom_pixel = (pattern & (0x80 >> rom_col)) ? 1 : 0;

                if (binary[py * patch_w + px] == rom_pixel) {
                    ++matches;
                }
            }
        }

        float score = static_cast<float>(matches) / static_cast<float>(total);
        if (score > best_score) {
            best_score = score;
            best_digit = d;
        }
    }

    if (best_score < kAcceptThreshold) {
        return {-1, 0.0f};
    }
    return {best_digit, best_score};
}

std::pair<int, float> VisionRewardSystem::match_digit(
    const std::vector<uint8_t>& patch) const {
    // Delegate to the Videopac ROM-based matcher.
    return match_videopac_digit(patch, kDigitWidth, kDigitHeight);
}

int VisionRewardSystem::extract_score(
    const std::vector<uint8_t>& observation) const {
    if (observation.empty()) return -1;

    if (score_region_.width < kDigitWidth ||
        score_region_.height < kDigitHeight) {
        return -1;
    }

    std::vector<uint8_t> gray = crop_to_grayscale(observation);

    // Slide a window across the cropped region.
    // Videopac characters are 8px wide on screen; in quad mode they are
    // spaced 16px apart.  We try the configured kDigitWidth first
    // (non-overlapping), which works when the score region is tightly cropped.
    int step = kDigitWidth;
    int max_digits = score_region_.width / step;
    int score = 0;
    bool found_any = false;

    for (int d = 0; d < max_digits; ++d) {
        int x_off = d * step;

        std::vector<uint8_t> patch(
            static_cast<size_t>(kDigitWidth) * kDigitHeight);
        for (int row = 0; row < kDigitHeight; ++row) {
            for (int col = 0; col < kDigitWidth; ++col) {
                size_t src =
                    static_cast<size_t>(row) * score_region_.width +
                    (x_off + col);
                if (src < gray.size()) {
                    patch[static_cast<size_t>(row) * kDigitWidth + col] =
                        gray[src];
                }
            }
        }

        auto [digit, confidence] = match_videopac_digit(patch, kDigitWidth, kDigitHeight);
        if (digit >= 0) {
            score = score * 10 + digit;
            found_any = true;
        }
        // Skip blank slots silently (no false positives on empty space)
    }

    return found_any ? score : -1;
}

}  // namespace retro_ai
