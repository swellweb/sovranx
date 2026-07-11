#include "sovranx/speculative/batch_verifier.hpp"

#include <utility>

#include "sovranx/speculative/acceptance.hpp"

namespace sovranx::speculative {

namespace {

// Multinomial draw driven by the shared acceptance RNG (kept separate from
// the Sampler's RNG so draft sampling and verification don't entangle).
TokenId draw_from(const std::vector<float>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float r = uniform(rng);
    std::size_t last_nonzero = 0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] <= 0.0f) continue;
        last_nonzero = i;
        r -= probs[i];
        if (r <= 0.0f) return static_cast<TokenId>(i);
    }
    return static_cast<TokenId>(last_nonzero);
}

}  // namespace

BatchVerifier::BatchVerifier(LlamaBackend& target_backend)
    : backend_(target_backend) {}

VerificationResult BatchVerifier::verify_batch(
    TokenId last_token, const DraftResult& draft, core::Sampler& sampler,
    std::mt19937& rng, const std::vector<TokenId>& recent) {
    if (draft.tokens.empty())
        throw core::EngineError("verify_batch called with an empty draft");

    // One batched target forward: position i's logits are the target
    // distribution for draft token i.
    std::vector<TokenId> batch{last_token};
    batch.insert(batch.end(), draft.tokens.begin(), draft.tokens.end() - 1);
    const auto all_logits = backend_.decode_batch(batch);

    VerificationResult result;
    std::vector<TokenId> history = recent;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for (std::size_t i = 0; i < draft.tokens.size(); ++i) {
        const TokenId d = draft.tokens[i];
        const auto target_probs = sampler.distribution(all_logits[i], history);
        const float p = target_probs[static_cast<std::size_t>(d)];
        const float q = draft.probs[i][static_cast<std::size_t>(d)];

        if (accept_token(p, q, uniform(rng))) {
            result.accepted_tokens.push_back(d);
            history.push_back(d);
            continue;
        }

        // First rejection: sample the corrected token from the residual so
        // the overall output distribution stays exactly the target's.
        result.has_correction = true;
        result.corrected_token =
            draw_from(residual_distribution(target_probs, draft.probs[i]), rng);
        break;
    }

    result.rejected_count = static_cast<int>(draft.tokens.size() -
                                             result.accepted_tokens.size());
    return result;
}

}  // namespace sovranx::speculative
