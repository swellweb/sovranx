#pragma once

#include <vector>

#include "sovranx/core/llama_backend.hpp"

namespace sovranx::speculative {

// Draft-model-free speculation: candidate continuations are found by
// n-gram lookup in the tokens already seen (prompt + generated). When the
// model is quoting or rephrasing its input — audits, rewrites, structured
// output — the continuation after a repeated n-gram is highly predictable,
// so acceptance is high and the draft costs NOTHING (no second model).
//
// Algorithm (prompt-lookup decoding): take the last `n` tokens of the
// history (n from max_ngram down to min_ngram); find the most recent
// earlier occurrence of that n-gram; propose the tokens that followed it.
class PromptLookup {
public:
    struct Config {
        int max_ngram = 3;  // longest pattern tried first
        int min_ngram = 1;  // shortest fallback
    };

    explicit PromptLookup(const Config& cfg);

    // Up to `n_draft` proposed continuation tokens (empty when the tail
    // n-grams never occurred before). The proposal is deterministic — the
    // matching acceptance distribution is a one-hot per token.
    std::vector<TokenId> find_draft(const std::vector<TokenId>& history,
                                    int n_draft) const;

private:
    Config cfg_;
};

}  // namespace sovranx::speculative
