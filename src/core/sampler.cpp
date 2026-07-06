#include "sovrano/core/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sovrano::core {

Sampler::Sampler(const GenerationConfig& cfg)
    : cfg_(cfg), rng_(static_cast<std::mt19937::result_type>(cfg.seed)) {}

TokenId Sampler::sample(std::vector<float> logits,
                        const std::vector<TokenId>& recent) {
    if (logits.empty())
        throw EngineError("cannot sample from empty logits");

    const auto n_vocab = logits.size();

    // 1. Repeat penalty over the last repeat_last_n tokens.
    if (cfg_.repeat_penalty != 1.0f && cfg_.repeat_last_n > 0) {
        const std::size_t window =
            std::min(recent.size(), static_cast<std::size_t>(cfg_.repeat_last_n));
        for (std::size_t i = recent.size() - window; i < recent.size(); ++i) {
            const TokenId t = recent[i];
            if (t < 0 || static_cast<std::size_t>(t) >= n_vocab) continue;
            float& logit = logits[static_cast<std::size_t>(t)];
            logit = logit > 0.0f ? logit / cfg_.repeat_penalty
                                 : logit * cfg_.repeat_penalty;
        }
    }

    // 2. Greedy when temperature is off.
    const auto argmax = std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end()));
    if (cfg_.temperature <= 0.0f) return static_cast<TokenId>(argmax);

    // 3. Temperature, then softmax (max-subtracted for stability).
    const float max_logit = logits[static_cast<std::size_t>(argmax)];
    std::vector<float> probs(n_vocab);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / cfg_.temperature);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    // 4. Top-p nucleus: smallest probability-sorted prefix with cumulative
    // mass >= top_p (at least one token survives).
    std::vector<std::size_t> order(n_vocab);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });

    const float top_p = std::clamp(cfg_.top_p, 0.0f, 1.0f);
    std::size_t keep = 0;
    float cumulative = 0.0f;
    while (keep < n_vocab) {
        cumulative += probs[order[keep]];
        ++keep;
        if (cumulative >= top_p) break;
    }

    // 5. Draw from the renormalized nucleus.
    std::uniform_real_distribution<float> uniform(0.0f, cumulative);
    float r = uniform(rng_);
    for (std::size_t i = 0; i < keep; ++i) {
        r -= probs[order[i]];
        if (r <= 0.0f) return static_cast<TokenId>(order[i]);
    }
    return static_cast<TokenId>(order[keep - 1]);  // float round-off fallback
}

}  // namespace sovrano::core
