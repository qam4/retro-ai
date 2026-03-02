#pragma once

#include "retro_ai/reward_system.hpp"
#include "retro_ai/rl_interface.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace retro_ai {

/// Rectangular region of the screen used for score detection.
struct ScreenRegion {
    int x;       ///< Left edge in pixels.
    int y;       ///< Top edge in pixels.
    int width;   ///< Region width in pixels.
    int height;  ///< Region height in pixels.
};

/// Vision-based reward: extracts score from screen pixels using template
/// matching against hardcoded digit bitmaps, then returns the delta between
/// the current and previous detected score.
class VisionRewardSystem : public RewardSystem {
public:
    /// Width of each digit template in pixels.
    static constexpr int kDigitWidth = 7;
    /// Height of each digit template in pixels.
    static constexpr int kDigitHeight = 10;
    /// Number of digit templates (0-9).
    static constexpr int kNumDigits = 10;

    /// Construct with a score region and the observation space dimensions.
    /// @param score_region  Screen area where the score is rendered.
    /// @param obs_space     Observation dimensions (used for pixel addressing).
    explicit VisionRewardSystem(const ScreenRegion& score_region,
                                const ObservationSpace& obs_space);

    float compute_reward(const StepResult& current,
                         const StepResult& previous) override;

    void reset() override;
    std::string name() const override;

    /// Load digit templates from a directory of image files.
    /// Currently a no-op; templates are hardcoded as static bitmaps.
    void load_digit_templates(const std::string& template_dir);

private:
    /// Extract the score visible in the configured screen region.
    /// @param observation  Flat RGB888 pixel buffer.
    /// @return Detected numeric score, or -1 if no digits matched.
    int extract_score(const std::vector<uint8_t>& observation) const;

    /// Convert a region of the observation to a grayscale intensity map.
    /// @param observation  Flat RGB888 pixel buffer.
    /// @return Grayscale pixels for the configured score region.
    std::vector<uint8_t> crop_to_grayscale(
        const std::vector<uint8_t>& observation) const;

    /// Match a single digit template against a grayscale patch.
    /// @param patch       Grayscale pixels (kDigitWidth * kDigitHeight).
    /// @return Best-matching digit (0-9) and its match score, or (-1, 0)
    ///         if no template exceeds the acceptance threshold.
    std::pair<int, float> match_digit(const std::vector<uint8_t>& patch) const;

    ScreenRegion score_region_;
    ObservationSpace obs_space_;
    int64_t previous_score_;
    bool has_previous_;

    /// Hardcoded 7×10 binary digit templates (1 = foreground, 0 = background).
    static const std::array<std::array<uint8_t, kDigitWidth * kDigitHeight>,
                            kNumDigits> digit_templates_;
};

}  // namespace retro_ai
