#pragma once

#include <random>
#include <vector>

#include "sovranx/core/engine.hpp"
#include "sovranx/core/llama_backend.hpp"
#include "sovranx/core/sampler.hpp"
#include "sovranx/speculative/draft_generator.hpp"

namespace sovranx::speculative {

struct VerificationResult {
    std::vector<TokenId> accepted_tokens;  // accepted prefix of the draft
    int rejected_count = 0;                // draft size - accepted
    TokenId corrected_token = -1;          // valid only if has_correction
    bool has_correction = false;           // true iff a rejection happened
};

// Verifies a draft with ONE batched forward pass of the target model.
//
// The batch is [last_token, draft[0..N-2]]: the logits at position i are
// the target distribution for draft token i. KV contract: on entry the
// target backend holds everything EXCEPT last_token; on exit it holds
// last_token plus the first N-1 draft tokens — the caller truncates back
// to the accepted prefix.
class BatchVerifier {
public:
    explicit BatchVerifier(LlamaBackend& target_backend);

    // `sampler` provides the target distribution semantics (must be
    // configured identically to the draft's); `rng` drives acceptance and
    // the residual draw. `recent` = history up to and including last_token.
    VerificationResult verify_batch(TokenId last_token,
                                    const DraftResult& draft,
                                    core::Sampler& sampler,
                                    std::mt19937& rng,
                                    const std::vector<TokenId>& recent);

private:
    LlamaBackend& backend_;
};

}  // namespace sovranx::speculative
