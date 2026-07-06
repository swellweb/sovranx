// Isolated tests for sovrano::core::Sampler. Pure unit: hand-built logits,
// expected outcomes derived from the sampling rules, no llama.cpp.

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <vector>

#include "sovrano/core/sampler.hpp"

using sovrano::TokenId;
using sovrano::core::EngineError;
using sovrano::core::GenerationConfig;
using sovrano::core::Sampler;

namespace {

GenerationConfig greedy() {
    GenerationConfig cfg;
    cfg.temperature = 0.0f;
    cfg.repeat_penalty = 1.0f;  // neutral unless a test opts in
    return cfg;
}

}  // namespace

TEST_CASE("greedy picks the argmax") {
    Sampler s(greedy());
    // argmax of {0.1, 3.0, 0.5, 2.9} is index 1.
    CHECK(s.sample({0.1f, 3.0f, 0.5f, 2.9f}, {}) == 1);
}

TEST_CASE("empty logits throw") {
    Sampler s(greedy());
    CHECK_THROWS_AS(s.sample({}, {}), EngineError);
}

TEST_CASE("repeat penalty divides positive logits") {
    auto cfg = greedy();
    cfg.repeat_penalty = 2.0f;
    cfg.repeat_last_n = 64;
    Sampler s(cfg);

    // Token 0 (logit 2.0) was seen recently: 2.0 / 2.0 = 1.0 < 1.9,
    // so greedy now prefers token 1.
    CHECK(s.sample({2.0f, 1.9f}, {0}) == 1);

    // Without the token in `recent`, argmax stays 0.
    Sampler s2(cfg);
    CHECK(s2.sample({2.0f, 1.9f}, {}) == 0);
}

TEST_CASE("repeat penalty multiplies negative logits") {
    auto cfg = greedy();
    cfg.repeat_penalty = 1.2f;
    Sampler s(cfg);

    // Token 1 (logit -0.45) seen recently: -0.45 * 1.2 = -0.54 < -0.5,
    // so token 0 wins despite starting lower.
    CHECK(s.sample({-0.5f, -0.45f}, {1}) == 0);
}

TEST_CASE("repeat penalty only looks at the last repeat_last_n tokens") {
    auto cfg = greedy();
    cfg.repeat_penalty = 2.0f;
    cfg.repeat_last_n = 2;
    Sampler s(cfg);

    // Token 0 appears in `recent`, but outside the last-2 window {5, 6},
    // so its logit is NOT penalized and it stays the argmax.
    CHECK(s.sample({2.0f, 1.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                   {0, 5, 6}) == 0);

    // Inside the window it is penalized: 2.0/2.0 = 1.0 < 1.9 -> token 1.
    Sampler s2(cfg);
    CHECK(s2.sample({2.0f, 1.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                    {5, 6, 0}) == 1);
}

TEST_CASE("tiny top_p reduces to argmax even with temperature") {
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 1e-6f;  // nucleus keeps only the most probable token
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    for (int i = 0; i < 50; ++i)
        CHECK(s.sample({0.0f, 5.0f, 1.0f}, {}) == 1);
}

TEST_CASE("dominant logit wins regardless of randomness") {
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.95f;
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    // softmax({20, 0, 0}) puts ~1.0 on token 0; with top_p 0.95 the
    // nucleus is {0} alone.
    for (int i = 0; i < 50; ++i)
        CHECK(s.sample({20.0f, 0.0f, 0.0f}, {}) == 0);
}

TEST_CASE("same seed reproduces the same sequence of draws") {
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 1.0f;
    cfg.repeat_penalty = 1.0f;
    cfg.seed = 1234;

    const std::vector<float> logits{1.0f, 1.1f, 0.9f, 1.05f};

    Sampler a(cfg), b(cfg);
    for (int i = 0; i < 100; ++i)
        CHECK(a.sample(logits, {}) == b.sample(logits, {}));
}

TEST_CASE("stochastic draws stay within the vocabulary") {
    GenerationConfig cfg;
    cfg.temperature = 1.5f;
    cfg.top_p = 1.0f;
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    const std::vector<float> logits{1.0f, 1.0f, 1.0f, 1.0f};
    std::set<TokenId> seen;
    for (int i = 0; i < 200; ++i) {
        const TokenId t = s.sample(logits, {});
        REQUIRE(t >= 0);
        REQUIRE(t < 4);
        seen.insert(t);
    }
    // Uniform logits with full nucleus: more than one token must appear.
    CHECK(seen.size() > 1);
}
