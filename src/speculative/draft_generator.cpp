#include "sovranx/speculative/draft_generator.hpp"

#include <utility>

namespace sovranx::speculative {

DraftGenerator::DraftGenerator(LlamaBackend& draft_backend, const Config& cfg)
    : backend_(draft_backend), cfg_(cfg) {}

DraftResult DraftGenerator::generate_draft(TokenId last_token,
                                           const std::vector<TokenId>& recent,
                                           core::Sampler& sampler,
                                           int n_tokens) {
    const int n = n_tokens > 0 ? n_tokens : cfg_.draft_tokens;
    const TokenId eos = backend_.eos_token();

    DraftResult result;
    std::vector<TokenId> history = recent;
    TokenId cur = last_token;

    for (int i = 0; i < n; ++i) {
        auto logits = backend_.decode_append({cur});
        auto probs = sampler.distribution(std::move(logits), history);
        const TokenId next = sampler.draw(probs);

        result.tokens.push_back(next);
        result.probs.push_back(std::move(probs));

        // No point drafting past EOS: the verifier either accepts it (and
        // generation stops) or rejects it (and everything after would be
        // thrown away anyway).
        if (next == eos) break;

        history.push_back(next);
        cur = next;
    }
    return result;
}

}  // namespace sovranx::speculative
