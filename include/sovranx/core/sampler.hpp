#pragma once

#include <random>
#include <vector>

#include "sovranx/core/engine.hpp"
#include "sovranx/core/llama_backend.hpp"

namespace sovranx::core {

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

    // Full sampling distribution over the vocabulary after repeat penalty,
    // temperature, softmax and top-p: entries outside the nucleus are 0 and
    // the rest sum to 1. Greedy (temperature <= 0) yields a one-hot at the
    // argmax. `logits` is taken by value: the sampler mutates its copy.
    // Throws EngineError on empty logits.
    //
    // Speculative decoding relies on this being THE distribution tokens are
    // drawn from: draft q and target p must share these exact semantics.
    std::vector<float> distribution(std::vector<float> logits,
                                    const std::vector<TokenId>& recent);

    // Multinomial draw from a distribution produced by distribution().
    TokenId draw(const std::vector<float>& probs);

    // draw(distribution(logits, recent)) — the classic path.
    TokenId sample(std::vector<float> logits,
                   const std::vector<TokenId>& recent);

private:
    GenerationConfig cfg_;
    std::mt19937 rng_;
};

}  // namespace sovranx::core
