#pragma once

#include <random>
#include <vector>

#include "sovrano/core/engine.hpp"
#include "sovrano/core/llama_backend.hpp"

namespace sovrano::core {

// Token sampler: repeat penalty -> temperature -> top-p (nucleus) -> draw.
// Pure w.r.t. the outside world (seeded RNG inside), so fully testable.
//
// Rules:
//   - repeat penalty (llama.cpp convention) on the last `repeat_last_n`
//     tokens of `recent`: positive logits are divided by the penalty,
//     negative ones multiplied.
//   - temperature <= 0 selects the argmax (greedy), no randomness.
//   - top-p keeps the smallest prefix of the probability-sorted vocab with
//     cumulative probability >= top_p (always at least one token).
class Sampler {
public:
    explicit Sampler(const GenerationConfig& cfg);

    // `logits` is taken by value: the sampler mutates its copy.
    // Throws EngineError on empty logits.
    TokenId sample(std::vector<float> logits,
                   const std::vector<TokenId>& recent);

private:
    GenerationConfig cfg_;
    std::mt19937 rng_;
};

}  // namespace sovrano::core
