// Isolated tests for sovranx::core::Sampler. Pure unit: hand-built logits,
// expected outcomes derived from the sampling rules, no llama.cpp.

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>
#include <vector>

#include "sovranx/core/sampler.hpp"

using Catch::Matchers::WithinAbs;

using sovranx::TokenId;
using sovranx::core::EngineError;
using sovranx::core::GenerationConfig;
using sovranx::core::Sampler;

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

TEST_CASE("distribution: softmax with temperature 1 and full nucleus") {
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 1.0f;
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    // softmax({ln 2, 0, 0}) = {2, 1, 1} / 4 = {0.5, 0.25, 0.25}.
    const auto d = s.distribution({std::log(2.0f), 0.0f, 0.0f}, {});

    REQUIRE(d.size() == 3);
    CHECK_THAT(d[0], WithinAbs(0.50, 1e-5));
    CHECK_THAT(d[1], WithinAbs(0.25, 1e-5));
    CHECK_THAT(d[2], WithinAbs(0.25, 1e-5));
}

TEST_CASE("distribution: top_p zeroes the tail and renormalizes") {
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.7f;
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    // Base probs {0.5, 0.25, 0.25}; sorted cum: 0.5 < 0.7, then 0.75 >= 0.7
    // -> keep two tokens, renormalize to {2/3, 1/3, 0}.
    const auto d = s.distribution({std::log(2.0f), 0.0f, 0.0f}, {});

    CHECK_THAT(d[0], WithinAbs(2.0 / 3.0, 1e-5));
    CHECK_THAT(d[1] + d[2], WithinAbs(1.0 / 3.0, 1e-5));
    CHECK((d[1] == 0.0f || d[2] == 0.0f));  // exactly one of the ties survives
}

TEST_CASE("distribution: nucleus on a large flat vocab keeps ceil(p*n)") {
    // Exercises the wide-nucleus path (vocab far beyond any top-K
    // prefilter): 1000 equal logits, top_p 0.95 -> the nucleus is the
    // smallest prefix reaching 0.95, i.e. 950 tokens of 1/1000 each,
    // renormalized to (1/1000)/0.95.
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.95f;
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    const auto d = s.distribution(std::vector<float>(1000, 0.0f), {});

    REQUIRE(d.size() == 1000);
    std::size_t kept = 0;
    double sum = 0.0;
    for (const float p : d) {
        if (p > 0.0f) {
            ++kept;
            // 1e-5: the float cumulative over 950 addends drifts ~2e-6.
            CHECK_THAT(p, WithinAbs(0.001 / 0.95, 1e-5));
        }
        sum += p;
    }
    // Exact math keeps 950; the float cumulative may undershoot 0.95 by
    // ~3e-7 and legitimately take one extra token.
    CHECK((kept == 950 || kept == 951));
    CHECK_THAT(sum, WithinAbs(1.0, 1e-4));
}

TEST_CASE("distribution: nucleus on a large peaked vocab keeps the peak") {
    // The dominant token alone exceeds top_p: nucleus of exactly 1 out
    // of 1000, probability renormalized to 1.
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.9f;
    cfg.repeat_penalty = 1.0f;
    Sampler s(cfg);

    std::vector<float> logits(1000, 0.0f);
    logits[123] = 20.0f;  // e^20 dwarfs 999 ones: p > 0.999
    const auto d = s.distribution(std::move(logits), {});

    REQUIRE(d.size() == 1000);
    CHECK_THAT(d[123], WithinAbs(1.0, 1e-5));
    for (std::size_t i = 0; i < d.size(); ++i)
        if (i != 123) CHECK(d[i] == 0.0f);
}

TEST_CASE("distribution: greedy is a one-hot at the argmax") {
    Sampler s(greedy());

    const auto d = s.distribution({0.1f, 3.0f, 0.5f}, {});

    REQUIRE(d.size() == 3);
    CHECK(d[0] == 0.0f);
    CHECK(d[1] == 1.0f);
    CHECK(d[2] == 0.0f);
}

TEST_CASE("draw from a one-hot always returns that token") {
    GenerationConfig cfg;
    cfg.temperature = 1.0f;
    Sampler s(cfg);

    for (int i = 0; i < 20; ++i)
        CHECK(s.draw({0.0f, 0.0f, 1.0f, 0.0f}) == 2);
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
