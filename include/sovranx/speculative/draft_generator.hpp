#pragma once

#include <vector>

#include "sovranx/core/engine.hpp"
#include "sovranx/core/llama_backend.hpp"
#include "sovranx/core/sampler.hpp"

namespace sovranx::speculative {

// Draft rollout result: candidate tokens plus the full draft distribution
// each one was sampled from (needed by acceptance sampling and by the
// residual on rejection).
struct DraftResult {
    std::vector<TokenId> tokens;
    std::vector<std::vector<float>> probs;  // probs[i] = q-dist of tokens[i]
};

// Rolls the small draft model forward `draft_tokens` steps.
//
// KV contract: on entry the draft backend holds every token EXCEPT
// `last_token`; on exit it additionally holds `last_token` and all drafted
// tokens but the final one (which the caller feeds back on the next
// iteration or discards via truncate_to).
class DraftGenerator {
public:
    struct Config {
        int draft_tokens = 16;
    };

    DraftGenerator(LlamaBackend& draft_backend, const Config& cfg);

    // `recent` = full token history (for the repeat penalty); `sampler`
    // must be the dedicated draft sampler so RNG state stays coherent
    // across iterations. `n_tokens` overrides cfg.draft_tokens when > 0
    // (adaptive drafting).
    DraftResult generate_draft(TokenId last_token,
                               const std::vector<TokenId>& recent,
                               core::Sampler& sampler,
                               int n_tokens = 0);

private:
    LlamaBackend& backend_;
    Config cfg_;
};

}  // namespace sovranx::speculative
