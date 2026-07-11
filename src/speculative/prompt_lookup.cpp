#include "sovranx/speculative/prompt_lookup.hpp"

#include <algorithm>

namespace sovranx::speculative {

PromptLookup::PromptLookup(const Config& cfg) : cfg_(cfg) {}

std::vector<TokenId> PromptLookup::find_draft(
    const std::vector<TokenId>& history, int n_draft) const {
    if (n_draft <= 0) return {};
    const auto len = history.size();

    const int max_n = std::max(cfg_.max_ngram, 1);
    const int min_n = std::clamp(cfg_.min_ngram, 1, max_n);

    for (int n = max_n; n >= min_n; --n) {
        const auto un = static_cast<std::size_t>(n);
        if (len < un + 1) continue;  // need the tail plus at least 1 token

        const TokenId* tail = history.data() + (len - un);

        // Most recent occurrence strictly BEFORE the tail (start < len-n
        // rules out the trailing self-match). The continuation runs from
        // right after the occurrence up to the end of the history — it may
        // overlap the tail: those are genuinely observed tokens.
        for (std::size_t start = len - un; start-- > 0;) {
            if (!std::equal(tail, tail + un, history.data() + start)) continue;

            const std::size_t cont = start + un;  // continuation begins here
            const std::size_t take = std::min(
                len - cont, static_cast<std::size_t>(n_draft));
            return {history.begin() + static_cast<long>(cont),
                    history.begin() + static_cast<long>(cont + take)};
        }
    }
    return {};
}

}  // namespace sovranx::speculative
