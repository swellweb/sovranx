#include "sovranx/speculative/acceptance.hpp"

#include <algorithm>

namespace sovranx::speculative {

std::vector<float> residual_distribution(const std::vector<float>& target_probs,
                                         const std::vector<float>& draft_probs) {
    std::vector<float> residual(target_probs.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < target_probs.size(); ++i) {
        const float q = i < draft_probs.size() ? draft_probs[i] : 0.0f;
        residual[i] = std::max(target_probs[i] - q, 0.0f);
        sum += residual[i];
    }
    if (sum <= 0.0f) return target_probs;  // q == p pointwise
    for (float& v : residual) v /= sum;
    return residual;
}

}  // namespace sovranx::speculative
