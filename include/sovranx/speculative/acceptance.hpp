#pragma once

#include <vector>

namespace sovranx::speculative {

// Pure speculative-sampling primitives (Leviathan/Chen scheme, the one
// DSpark builds on). Kept as free functions so the math is testable with
// hand-computed values.

// Accept a draft token given target probability p, draft probability q and
// a uniform draw r in [0, 1):
//   p >= q            -> always accept
//   otherwise         -> accept with probability p / q  (i.e. r < p / q)
// Formulated as r * q < p to avoid the division (q may be 0: never accept).
inline bool accept_token(float target_prob, float draft_prob, float r) {
    return r * draft_prob < target_prob;
}

// Distribution to sample the corrected token from after a rejection:
// normalized max(p - q, 0). Sampling rejections from the residual makes the
// overall output distribution exactly the target's. Falls back to p itself
// when the residual is empty (q == p pointwise).
std::vector<float> residual_distribution(const std::vector<float>& target_probs,
                                         const std::vector<float>& draft_probs);

}  // namespace sovranx::speculative
