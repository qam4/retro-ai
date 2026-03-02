#include "retro_ai/reward_systems/vision.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace retro_ai {

// ---------------------------------------------------------------------------
// Hardcoded 7×10 binary digit templates.  Each template is a 70-element array
// stored row-major (7 columns × 10 rows).  1 = foreground, 0 = background.
// These are simple block-style digits suitable for matching retro game fonts.
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
    // Templates are currently hardcoded.  This hook exists so that future
    // versions can load custom digit bitmaps from image files.
}

// ---------------------------------------------------------------------------
// Reward computation
// ---------------------------------------------------------------------------

float VisionRewardSystem::compute_reward(const StepResult& current,
                                         const StepResult& /*previous*/) {
    int detected = extract_score(current.observation);

    if (detected < 0) {
        // No score visible – return neutral reward and warn once.
        std::cerr << "[vision] warning: score not visible in observation\n";
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

            // Standard luminance formula.
            uint8_t lum = static_cast<uint8_t>(
                0.299f * r + 0.587f * g + 0.114f * b);

            gray[static_cast<size_t>(row) * score_region_.width + col] = lum;
        }
    }
    return gray;
}

std::pair<int, float> VisionRewardSystem::match_digit(
    const std::vector<uint8_t>& patch) const {
    // Binarise the patch: pixels above the mean are foreground (1).
    float mean = 0.0f;
    for (auto v : patch) mean += v;
    mean /= static_cast<float>(patch.size());

    std::vector<uint8_t> binary(patch.size());
    for (size_t i = 0; i < patch.size(); ++i) {
        binary[i] = (patch[i] > static_cast<uint8_t>(mean)) ? 1 : 0;
    }

    int best_digit = -1;
    float best_score = 0.0f;
    constexpr float kAcceptThreshold = 0.65f;

    for (int d = 0; d < kNumDigits; ++d) {
        int matches = 0;
        const auto& tmpl = digit_templates_[d];
        for (size_t i = 0; i < binary.size(); ++i) {
            if (binary[i] == tmpl[i]) ++matches;
        }
        float score =
            static_cast<float>(matches) / static_cast<float>(binary.size());
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

int VisionRewardSystem::extract_score(
    const std::vector<uint8_t>& observation) const {
    if (observation.empty()) return -1;

    // Ensure the region fits within the observation.
    if (score_region_.width < kDigitWidth ||
        score_region_.height < kDigitHeight) {
        return -1;
    }

    std::vector<uint8_t> gray = crop_to_grayscale(observation);

    // Slide a kDigitWidth × kDigitHeight window across the cropped region
    // from left to right, stepping by kDigitWidth (non-overlapping).
    int max_digits = score_region_.width / kDigitWidth;
    int score = 0;
    bool found_any = false;

    for (int d = 0; d < max_digits; ++d) {
        int x_off = d * kDigitWidth;

        // Extract the patch.
        std::vector<uint8_t> patch(
            static_cast<size_t>(kDigitWidth) * kDigitHeight);
        for (int row = 0; row < kDigitHeight; ++row) {
            for (int col = 0; col < kDigitWidth; ++col) {
                size_t src =
                    static_cast<size_t>(row) * score_region_.width +
                    (x_off + col);
                patch[static_cast<size_t>(row) * kDigitWidth + col] =
                    gray[src];
            }
        }

        auto [digit, confidence] = match_digit(patch);
        if (digit >= 0) {
            score = score * 10 + digit;
            found_any = true;
        }
    }

    return found_any ? score : -1;
}

}  // namespace retro_ai
