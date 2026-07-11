// Isolated tests for PromptLookup: pure token-sequence logic, every
// expectation derived by hand from the n-gram rule.

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "sovranx/speculative/prompt_lookup.hpp"

using sovranx::TokenId;
using sovranx::speculative::PromptLookup;

namespace {
using Tokens = std::vector<TokenId>;
}

TEST_CASE("proposes the continuation of the most recent matching n-gram") {
    PromptLookup lookup({/*max_ngram=*/2, /*min_ngram=*/1});

    // History ...{2,3}... ends with {2,3}: the earlier occurrence at index
    // 1 was followed by {4, 2, 3} -> propose up to n_draft of those.
    const Tokens history{1, 2, 3, 4, 2, 3};

    CHECK(lookup.find_draft(history, 1) == Tokens{4});
    CHECK(lookup.find_draft(history, 2) == Tokens{4, 2});
    CHECK(lookup.find_draft(history, 8) == Tokens{4, 2, 3});  // clipped at end
}

TEST_CASE("prefers the longest n-gram, falls back to shorter ones") {
    PromptLookup lookup({/*max_ngram=*/3, /*min_ngram=*/1});

    // Tail {9, 1}: the 2-gram never occurred before, but the 1-gram {1}
    // did (index 1), followed by {9}.
    const Tokens history{5, 1, 9, 1};
    CHECK(lookup.find_draft(history, 4) == Tokens{9, 1});

    // Tail 3-gram {2,3,4} occurred at index 0 followed by {7}: the longer
    // match wins even though shorter tails also match elsewhere.
    const Tokens h2{2, 3, 4, 7, 3, 4, 8, 2, 3, 4};
    CHECK(lookup.find_draft(h2, 1) == Tokens{7});
}

TEST_CASE("most recent occurrence wins among equals") {
    PromptLookup lookup({/*max_ngram=*/1, /*min_ngram=*/1});

    // Token 2 occurs at indices 0 (followed by 5) and 2 (followed by 9):
    // the LATER occurrence is the better predictor.
    const Tokens history{2, 5, 2, 9, 2};
    CHECK(lookup.find_draft(history, 1) == Tokens{9});
}

TEST_CASE("returns empty when nothing matches or history is too short") {
    PromptLookup lookup({/*max_ngram=*/3, /*min_ngram=*/1});

    CHECK(lookup.find_draft({}, 4).empty());
    CHECK(lookup.find_draft({7}, 4).empty());          // no earlier tokens
    CHECK(lookup.find_draft({1, 2, 3, 4}, 4).empty()); // all distinct
}

TEST_CASE("the trailing self-match is never used") {
    PromptLookup lookup({/*max_ngram=*/2, /*min_ngram=*/1});

    // Tail {3,4}: its only "occurrence" is the tail itself. The 1-gram {4}
    // likewise only occurs at the very end -> empty.
    CHECK(lookup.find_draft({3, 4}, 2).empty());

    // {4} also occurs earlier here, followed by {3, 4} (the continuation
    // may run into the tail: those are observed tokens).
    CHECK(lookup.find_draft({4, 3, 4}, 2) == Tokens{3, 4});
    CHECK(lookup.find_draft({4, 3, 4}, 1) == Tokens{3});
}

TEST_CASE("n_draft zero or negative yields empty") {
    PromptLookup lookup({2, 1});
    CHECK(lookup.find_draft({1, 2, 1, 2}, 0).empty());
    CHECK(lookup.find_draft({1, 2, 1, 2}, -3).empty());
}
