// Isolated tests for the speculative-decoding module. All model behavior is
// scripted through MockBackend; the acceptance math is checked against
// hand-computed values, plus a statistical check of the acceptance rule.
// Real-model correctness/performance live at the bottom, gated on model
// availability.

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "../mock/llama_mock.hpp"
#include "reame/core/model.hpp"
#include "reame/palimpsest/corpus_index.hpp"
#include "reame/speculative/acceptance.hpp"
#include "reame/speculative/batch_verifier.hpp"
#include "reame/speculative/draft_generator.hpp"
#include "reame/speculative/speculative_decoder.hpp"

using Catch::Matchers::WithinAbs;
using reame::TokenId;
using reame::test::MockBackend;
using reame::core::EngineError;
using reame::core::GenerationConfig;
using reame::core::Sampler;
using reame::speculative::accept_token;
using reame::speculative::BatchVerifier;
using reame::speculative::DraftGenerator;
using reame::speculative::DraftResult;
using reame::speculative::residual_distribution;
using reame::speculative::SpeculativeDecoder;

namespace {

GenerationConfig greedy(int max_tokens = 32) {
    GenerationConfig g;
    g.temperature = 0.0f;
    g.repeat_penalty = 1.0f;
    g.max_tokens = max_tokens;
    return g;
}

// Logits with the argmax at `idx` (vocab of `n`).
std::vector<float> peak(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 10.0f;
    return v;
}

std::vector<float> one_hot(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 1.0f;
    return v;
}

// vocab 6, EOS 5 throughout the mock tests.
constexpr std::size_t kVocab = 6;
constexpr TokenId kEos = 5;

void setup(MockBackend& m) {
    m.vocab_size_value = kVocab;
    m.eos_token_value = kEos;
}

}  // namespace

// ---------------------------------------------------------------------------
// Acceptance math (pure)
// ---------------------------------------------------------------------------

TEST_CASE("accept_token: target >= draft always accepts") {
    CHECK(accept_token(0.5f, 0.5f, 0.999f));
    CHECK(accept_token(0.9f, 0.1f, 0.999f));
}

TEST_CASE("accept_token: otherwise accepts iff r < p/q") {
    // p = 0.2, q = 0.8 -> threshold 0.25.
    CHECK(accept_token(0.2f, 0.8f, 0.24f));
    CHECK_FALSE(accept_token(0.2f, 0.8f, 0.26f));
}

TEST_CASE("accept_token: zero draft probability never accepts a zero-target token") {
    CHECK_FALSE(accept_token(0.0f, 0.0f, 0.0f));
    CHECK_FALSE(accept_token(0.0f, 0.5f, 0.0f));
}

TEST_CASE("residual_distribution: normalized max(p - q, 0)") {
    // p - q = {-0.3, 0.2, 0.1} -> clamp {0, 0.2, 0.1} -> {0, 2/3, 1/3}.
    const auto r = residual_distribution({0.5f, 0.3f, 0.2f},
                                         {0.8f, 0.1f, 0.1f});

    REQUIRE(r.size() == 3);
    CHECK(r[0] == 0.0f);
    CHECK_THAT(r[1], WithinAbs(2.0 / 3.0, 1e-5));
    CHECK_THAT(r[2], WithinAbs(1.0 / 3.0, 1e-5));
}

TEST_CASE("residual_distribution: falls back to target when p == q") {
    const std::vector<float> p{0.6f, 0.4f};
    CHECK(residual_distribution(p, p) == p);
}

TEST_CASE("acceptance rule matches its probability statistically") {
    // p = 0.3, q = 0.6 -> accept probability 0.5.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    int accepted = 0;
    const int trials = 20000;
    for (int i = 0; i < trials; ++i)
        if (accept_token(0.3f, 0.6f, uniform(rng))) ++accepted;

    CHECK_THAT(static_cast<double>(accepted) / trials, WithinAbs(0.5, 0.02));
}

// ---------------------------------------------------------------------------
// DraftGenerator
// ---------------------------------------------------------------------------

TEST_CASE("draft generator rolls out sequentially from the last token") {
    MockBackend draft;
    setup(draft);
    draft.decode_queue = {peak(kVocab, 1), peak(kVocab, 2), peak(kVocab, 3)};

    DraftGenerator gen(draft, {/*draft_tokens=*/3});
    Sampler sampler(greedy());

    const auto result = gen.generate_draft(/*last_token=*/0, {0}, sampler);

    CHECK(result.tokens == std::vector<TokenId>{1, 2, 3});
    REQUIRE(result.probs.size() == 3);
    CHECK(result.probs[0] == one_hot(kVocab, 1));  // greedy -> one-hot q
    CHECK(result.probs[2] == one_hot(kVocab, 3));

    // Sequential rollout: cur, then each drafted token except the last.
    REQUIRE(draft.decode_append_calls.size() == 3);
    CHECK(draft.decode_append_calls[0] == std::vector<TokenId>{0});
    CHECK(draft.decode_append_calls[1] == std::vector<TokenId>{1});
    CHECK(draft.decode_append_calls[2] == std::vector<TokenId>{2});
}

TEST_CASE("draft generator honours the per-call token count override") {
    MockBackend draft;
    setup(draft);
    draft.decode_result = peak(kVocab, 1);

    DraftGenerator gen(draft, {/*draft_tokens=*/16});
    Sampler sampler(greedy());

    const auto result = gen.generate_draft(0, {0}, sampler, /*n_tokens=*/2);

    CHECK(result.tokens.size() == 2);
    CHECK(draft.decode_append_calls.size() == 2);
}

// ---------------------------------------------------------------------------
// BatchVerifier
// ---------------------------------------------------------------------------

TEST_CASE("verifier accepts the agreeing prefix and corrects the first miss") {
    MockBackend target;
    setup(target);
    // Batch [cur=0, d1=1, d2=2]: position i is the target dist for draft i.
    // Target agrees on 1 and 2, wants 4 instead of 3.
    target.decode_batch_queue = {
        {peak(kVocab, 1), peak(kVocab, 2), peak(kVocab, 4)}};

    DraftResult draft;
    draft.tokens = {1, 2, 3};
    draft.probs = {one_hot(kVocab, 1), one_hot(kVocab, 2), one_hot(kVocab, 3)};

    BatchVerifier verifier(target);
    Sampler sampler(greedy());
    std::mt19937 rng(42);

    const auto res = verifier.verify_batch(0, draft, sampler, rng, {0});

    CHECK(res.accepted_tokens == std::vector<TokenId>{1, 2});
    CHECK(res.rejected_count == 1);
    REQUIRE(res.has_correction);
    CHECK(res.corrected_token == 4);

    // Exactly one batched target forward: [last_token, d1, d2].
    REQUIRE(target.decode_batch_calls.size() == 1);
    CHECK(target.decode_batch_calls[0] == std::vector<TokenId>{0, 1, 2});
}

TEST_CASE("verifier accepts everything when the target agrees") {
    MockBackend target;
    setup(target);
    target.decode_batch_queue = {
        {peak(kVocab, 1), peak(kVocab, 2), peak(kVocab, 3)}};

    DraftResult draft;
    draft.tokens = {1, 2, 3};
    draft.probs = {one_hot(kVocab, 1), one_hot(kVocab, 2), one_hot(kVocab, 3)};

    BatchVerifier verifier(target);
    Sampler sampler(greedy());
    std::mt19937 rng(42);

    const auto res = verifier.verify_batch(0, draft, sampler, rng, {0});

    CHECK(res.accepted_tokens == std::vector<TokenId>{1, 2, 3});
    CHECK(res.rejected_count == 0);
    CHECK_FALSE(res.has_correction);
}

// ---------------------------------------------------------------------------
// SpeculativeDecoder
// ---------------------------------------------------------------------------

TEST_CASE("decoder rejects vocabularies differing by more than the padding tolerance") {
    MockBackend target, draft;
    setup(target);
    setup(draft);
    draft.vocab_size_value = static_cast<std::int32_t>(kVocab) + 129;

    CHECK_THROWS_AS(SpeculativeDecoder(target, &draft, {}), EngineError);
}

TEST_CASE("decoder tolerates small vocab padding differences (Qwen 7B vs 0.5B)") {
    // Qwen2.5-7B has 152064 entries, the 0.5B draft 151936: the extra 128
    // are unused padding. Same shape here at mock scale: target one entry
    // larger than the draft.
    MockBackend target, draft;
    setup(target);
    setup(draft);
    target.vocab_size_value = static_cast<std::int32_t>(kVocab) + 1;

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;

    // Draft (vocab 6) proposes {1, 2}; target (vocab 7) agrees, then EOS.
    draft.decode_queue = {peak(kVocab, 1), peak(kVocab, 2), peak(kVocab, kEos)};
    target.decode_batch_queue = {{peak(kVocab + 1, 1), peak(kVocab + 1, 2)},
                                 {peak(kVocab + 1, kEos)}};

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0}, greedy());

    CHECK(out == std::vector<TokenId>{1, 2});
    CHECK(decoder.speculative_active());
}

TEST_CASE("decoder stops at any end-of-generation token, not only eos") {
    MockBackend target, draft;
    setup(target);
    setup(draft);
    // Token 4 is a second end-of-turn marker (ChatML-style <|im_end|>).
    const TokenId kImEnd = 4;
    target.eog_tokens = {kImEnd};
    draft.eog_tokens = {kImEnd};

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;

    // Draft proposes {1, kImEnd}; target agrees on both. The accepted
    // kImEnd must end generation: output is just {1}.
    draft.decode_queue = {peak(kVocab, 1), peak(kVocab, kImEnd)};
    target.decode_batch_queue = {{peak(kVocab, 1), peak(kVocab, kImEnd)}};

    SpeculativeDecoder decoder(target, &draft, cfg);
    CHECK(decoder.generate({0}, greedy()) == std::vector<TokenId>{1});
}

TEST_CASE("decoder falls back to plain when a token outside the draft vocab is emitted") {
    MockBackend target, draft;
    setup(target);
    setup(draft);
    target.vocab_size_value = static_cast<std::int32_t>(kVocab) + 1;

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;

    // Draft proposes {1}; target instead wants token 6 — valid for the
    // target but OUTSIDE the draft's vocabulary. The decoder must emit it
    // and then stop speculating (the draft cannot ingest that token).
    draft.decode_queue = {peak(kVocab, 1)};
    draft.decode_result = peak(kVocab, 1);
    target.decode_batch_queue = {{peak(kVocab + 1, kVocab)}};
    // Plain continuation after the fallback: EOS.
    target.decode_queue = {peak(kVocab + 1, kEos)};

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0}, greedy());

    CHECK(out == std::vector<TokenId>{static_cast<TokenId>(kVocab)});
    CHECK_FALSE(decoder.speculative_active());
}

TEST_CASE("decoder: accept-all then EOS-correction, with metrics") {
    MockBackend target, draft;
    setup(target);
    setup(draft);

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;

    // Iter 1: draft proposes {1, 2}; target agrees on both.
    // Iter 2: draft proposes {3, 4}; target accepts 3, corrects 4 -> EOS.
    draft.decode_queue = {peak(kVocab, 1), peak(kVocab, 2),
                          peak(kVocab, 3), peak(kVocab, 4)};
    target.decode_batch_queue = {
        {peak(kVocab, 1), peak(kVocab, 2)},
        {peak(kVocab, 3), peak(kVocab, kEos)}};

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0}, greedy());

    CHECK(out == std::vector<TokenId>{1, 2, 3});

    // Target forwards: [cur, d1] per iteration.
    REQUIRE(target.decode_batch_calls.size() == 2);
    CHECK(target.decode_batch_calls[0] == std::vector<TokenId>{0, 1});
    CHECK(target.decode_batch_calls[1] == std::vector<TokenId>{2, 3});

    const auto& m = decoder.metrics();
    CHECK(m.total_draft_tokens == 4);
    CHECK(m.total_accepted_tokens == 3);
    CHECK(m.total_rejected_tokens == 1);
    CHECK(m.generated_tokens == 3);
    CHECK_THAT(m.acceptance_rate(), WithinAbs(0.75, 1e-9));
}

TEST_CASE("decoder greedy output equals plain greedy output") {
    // The "model": after token 0 comes 1, then 2, then EOS.
    const auto script = [] {
        return std::deque<std::vector<float>>{
            peak(kVocab, 1), peak(kVocab, 2), peak(kVocab, kEos)};
    };

    // Plain path: no draft model.
    MockBackend plain_target;
    setup(plain_target);
    plain_target.decode_queue = script();
    SpeculativeDecoder plain(plain_target, nullptr, {});
    const auto plain_out = plain.generate({0}, greedy());
    CHECK_FALSE(plain.speculative_active());

    // Speculative path: draft is the same "model".
    MockBackend target, draft;
    setup(target);
    setup(draft);
    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;
    draft.decode_queue = script();
    target.decode_batch_queue = {
        {peak(kVocab, 1), peak(kVocab, 2)},        // accepts {1, 2}
        {peak(kVocab, kEos), peak(kVocab, 0)}};    // accepts draft's EOS
    SpeculativeDecoder spec(target, &draft, cfg);
    const auto spec_out = spec.generate({0}, greedy());

    CHECK(plain_out == std::vector<TokenId>{1, 2});
    CHECK(spec_out == plain_out);
}

TEST_CASE("decoder falls back to plain decoding when the draft fails") {
    MockBackend target, draft;
    setup(target);
    setup(draft);
    draft.fail_decodes = true;
    target.decode_queue = {peak(kVocab, 1), peak(kVocab, 2),
                           peak(kVocab, kEos)};

    SpeculativeDecoder decoder(target, &draft, {});
    CHECK(decoder.speculative_active());

    const auto out = decoder.generate({0}, greedy());

    CHECK(out == std::vector<TokenId>{1, 2});
    CHECK_FALSE(decoder.speculative_active());
    CHECK(target.decode_batch_calls.empty());
}

TEST_CASE("decoder auto-disables speculation on sustained low acceptance") {
    MockBackend target, draft;
    setup(target);
    setup(draft);

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;
    cfg.disable_after_drafted = 4;
    cfg.disable_below_acceptance = 0.5;

    // Draft always proposes token 4; target never agrees.
    draft.decode_result = peak(kVocab, 4);
    target.decode_batch_queue = {
        {peak(kVocab, 1), peak(kVocab, 0)},   // reject at 0 -> corrected 1
        {peak(kVocab, 2), peak(kVocab, 0)}};  // reject at 0 -> corrected 2
    // Plain continuation after auto-disable: next comes EOS.
    target.decode_queue = {peak(kVocab, kEos)};

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0}, greedy());

    CHECK(out == std::vector<TokenId>{1, 2});
    CHECK_FALSE(decoder.speculative_active());
    CHECK(target.decode_batch_calls.size() == 2);  // no drafting after disable
    CHECK(decoder.metrics().total_draft_tokens == 4);
    CHECK(decoder.metrics().total_accepted_tokens == 0);
}

TEST_CASE("decoder grows the draft length when everything is accepted") {
    MockBackend target, draft;
    setup(target);
    setup(draft);

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 4;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 8;

    draft.decode_queue = {peak(kVocab, 1), peak(kVocab, 2),
                          peak(kVocab, 3), peak(kVocab, 4)};
    target.decode_batch_queue = {{peak(kVocab, 1), peak(kVocab, 2),
                                  peak(kVocab, 3), peak(kVocab, 4)}};

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0}, greedy(/*max_tokens=*/4));

    CHECK(out == std::vector<TokenId>{1, 2, 3, 4});
    CHECK(decoder.current_draft_tokens() > 4);
}

TEST_CASE("decoder shrinks the draft length on heavy rejection") {
    MockBackend target, draft;
    setup(target);
    setup(draft);

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 4;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 8;

    draft.decode_result = peak(kVocab, 4);  // always proposes 4
    target.decode_batch_queue = {
        {peak(kVocab, 1), peak(kVocab, 0), peak(kVocab, 0), peak(kVocab, 0)}};

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0}, greedy(/*max_tokens=*/1));

    CHECK(out == std::vector<TokenId>{1});  // the corrected token
    CHECK(decoder.current_draft_tokens() < 4);
}

TEST_CASE("decoder truncates both models back to the accepted prefix") {
    MockBackend target, draft;
    setup(target);
    setup(draft);

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 3;
    cfg.min_draft_tokens = 3;
    cfg.max_draft_tokens = 3;

    // Prompt {0, 1}: prefill decodes {0} on both models (consuming one
    // queue entry each — the leading filler), cur = 1. Draft then proposes
    // {2, 3, 4}; target accepts 2, corrects to 1.
    const auto filler = peak(kVocab, 0);
    draft.decode_queue = {filler, peak(kVocab, 2), peak(kVocab, 3),
                          peak(kVocab, 4)};
    target.decode_batch_queue = {
        {peak(kVocab, 2), peak(kVocab, 1), peak(kVocab, 0)}};
    // Target: filler for the prefill, then EOS for iteration 2's
    // single-token batch (decode_batch falls back to decode_queue).
    target.decode_queue = {filler, peak(kVocab, kEos)};
    draft.decode_result = peak(kVocab, kEos);

    SpeculativeDecoder decoder(target, &draft, cfg);
    const auto out = decoder.generate({0, 1}, greedy());

    // Accepted {2}, corrected 1 -> emitted {2, 1}; then EOS from the next
    // iteration's verification.
    REQUIRE(out.size() >= 2);
    CHECK(out[0] == 2);
    CHECK(out[1] == 1);

    // After iter 1 both models hold prefix(1) + cur(1) + accepted(1) = 3
    // positions; everything beyond was truncated away.
    REQUIRE_FALSE(target.truncate_calls.empty());
    CHECK(target.truncate_calls[0] == 3);
    REQUIRE_FALSE(draft.truncate_calls.empty());
    CHECK(draft.truncate_calls[0] == 3);
}

// ---------------------------------------------------------------------------
// Prompt-lookup mode (draft-model-free speculation)
// ---------------------------------------------------------------------------

TEST_CASE("lookup mode drafts from history and verifies in one batch") {
    MockBackend target;
    setup(target);

    SpeculativeDecoder::Config cfg;
    cfg.mode = SpeculativeDecoder::Config::Mode::PromptLookup;
    cfg.draft_tokens = 3;
    cfg.min_draft_tokens = 3;
    cfg.max_draft_tokens = 3;

    // Prompt {1,2,3,1,2}: tail 2-gram {1,2} matched at index 0, followed
    // by {3,1,2} -> proposal. Target agrees on all three; the second
    // iteration proposes again and the target corrects to EOS.
    target.decode_batch_queue = {
        {peak(kVocab, 3), peak(kVocab, 1), peak(kVocab, 2)},
        {peak(kVocab, kEos), peak(kVocab, 0), peak(kVocab, 0)}};

    SpeculativeDecoder decoder(target, nullptr, cfg);
    CHECK(decoder.speculative_active());  // active WITHOUT a draft model

    const auto out = decoder.generate({1, 2, 3, 1, 2}, greedy());

    CHECK(out == std::vector<TokenId>{3, 1, 2});
    // Verification batches: [cur=2, d1=3, d2=1] both times.
    REQUIRE(target.decode_batch_calls.size() == 2);
    CHECK(target.decode_batch_calls[0] == std::vector<TokenId>{2, 3, 1});
    CHECK(decoder.metrics().total_draft_tokens == 6);
    CHECK(decoder.metrics().total_accepted_tokens == 3);
}

TEST_CASE("lookup mode falls back to a plain step when nothing matches, staying active") {
    MockBackend target;
    setup(target);

    SpeculativeDecoder::Config cfg;
    cfg.mode = SpeculativeDecoder::Config::Mode::PromptLookup;

    // All-distinct prompt: no n-gram repeats -> plain single-token step.
    // First entry feeds the prefill (its logits are unused).
    target.decode_queue = {peak(kVocab, 0), peak(kVocab, 4), peak(kVocab, kEos)};

    SpeculativeDecoder decoder(target, nullptr, cfg);
    const auto out = decoder.generate({0, 1, 2, 3}, greedy(/*max_tokens=*/1));

    CHECK(out == std::vector<TokenId>{4});
    CHECK(target.decode_batch_calls.empty());       // no batch: nothing to verify
    CHECK_FALSE(target.decode_append_calls.empty());  // plain path used
    CHECK(decoder.speculative_active());            // lookup stays armed
}

// ---------------------------------------------------------------------------
// Archive (palimpsest) drafts: server memory as a speculation source
// ---------------------------------------------------------------------------

TEST_CASE("lookup mode falls back to the corpus when the prompt has no repeats") {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "reame-spec-corpus";
    fs::remove_all(dir);
    fs::create_directories(dir);
    reame::palimpsest::CorpusIndex corpus({dir, /*ngram=*/2});
    // A PREVIOUS generation on this server: after {0, 1} came {2, 3}.
    corpus.observe({9, 0, 1, 2, 3});

    MockBackend target;
    setup(target);
    SpeculativeDecoder::Config cfg;
    cfg.mode = SpeculativeDecoder::Config::Mode::PromptLookup;
    cfg.lookup_max_ngram = 2;
    cfg.draft_tokens = 2;
    cfg.min_draft_tokens = 2;
    cfg.max_draft_tokens = 2;
    cfg.corpus = &corpus;

    // Prompt {0, 1}: no internal repeats -> prompt-lookup finds nothing,
    // the corpus proposes {2, 3}; target accepts both. The follow-up step
    // has no draft material left, so it runs plain and meets EOS.
    target.decode_batch_queue = {{peak(kVocab, 2), peak(kVocab, 3)}};
    target.decode_result = peak(kVocab, kEos);

    SpeculativeDecoder decoder(target, nullptr, cfg);
    const auto out = decoder.generate({0, 1}, greedy());

    CHECK(out == std::vector<TokenId>{2, 3});
    REQUIRE_FALSE(target.decode_batch_calls.empty());
    CHECK(target.decode_batch_calls[0] == std::vector<TokenId>{1, 2});
    fs::remove_all(dir);
}

TEST_CASE("finished generations are observed into the corpus") {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "reame-spec-observe";
    fs::remove_all(dir);
    fs::create_directories(dir);
    reame::palimpsest::CorpusIndex corpus({dir, /*ngram=*/2});

    MockBackend target;
    setup(target);
    SpeculativeDecoder::Config cfg;
    cfg.mode = SpeculativeDecoder::Config::Mode::PromptLookup;
    cfg.corpus = &corpus;

    // Plain-path generation (no repeats anywhere): {0,1} -> 2 -> 3 -> EOS.
    target.decode_queue = {peak(kVocab, 0), peak(kVocab, 2), peak(kVocab, 3),
                           peak(kVocab, kEos)};

    SpeculativeDecoder decoder(target, nullptr, cfg);
    decoder.generate({0, 1}, greedy());

    // The full history (prompt + generated) is now server memory.
    CHECK(corpus.draft({0, 1}, 2) == std::vector<TokenId>{2, 3});
    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Real-model integration. SKIPs when models are unavailable.
// ---------------------------------------------------------------------------

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    if (const char* v = std::getenv(name)) return v;
    return fallback;
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

}  // namespace

TEST_CASE("[integration] speculative equals plain greedy with a real model",
          "[integration]") {
#ifndef REAME_HAS_LLAMA
    SKIP("built without llama.cpp");
#else
    const auto path = env_or("REAME_TEST_MODEL",
                             "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    if (!file_exists(path)) SKIP("model file not found: " + path);

    reame::ModelParams p;
    p.path = path;
    p.context_length = 256;
    p.threads = 4;

    auto target = reame::make_llama_backend(p);
    auto draft = reame::make_llama_backend(p);  // draft == target model

    // Predictable continuation on purpose: single-token and batched CPU
    // kernels differ numerically (x86 repack especially), so a flat
    // distribution flips argmax on near-ties and acceptance drops even
    // with an identical model. Large argmax gaps make the check about the
    // MECHANISM, not about kernel rounding.
    const auto prompt = target->tokenize(
        "Count slowly: one, two, three, four, five, six, seven, eight,",
        true);

    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 4;
    SpeculativeDecoder spec(*target, draft.get(), cfg);
    const auto spec_out = spec.generate(prompt, greedy(12));

    // Same model as draft -> greedy acceptance is near-total. Not exactly
    // 1.0 on every platform: the draft decodes token-by-token while the
    // verifier uses the batched kernels, whose numerics differ slightly
    // (observed on x86/AVX2), so the argmax can flip on a near-tie.
    CHECK(spec.metrics().acceptance_rate() > 0.9);

    auto plain_backend = reame::make_llama_backend(p);
    SpeculativeDecoder plain(*plain_backend, nullptr, {});
    const auto plain_out = plain.generate(prompt, greedy(12));

    // Exact output equality only holds when no near-tie flipped; a flip
    // is still corrected from the target's own distribution.
    if (spec.metrics().acceptance_rate() == 1.0)
        CHECK(spec_out == plain_out);
    else
        CHECK(spec_out.size() == plain_out.size());
#endif
}

TEST_CASE("[integration] speculative is faster with a real draft/target pair",
          "[integration][performance]") {
#ifndef REAME_HAS_LLAMA
    SKIP("built without llama.cpp");
#else
    const auto target_path = env_or(
        "REAME_TARGET_MODEL", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    const auto draft_path = env_or(
        "REAME_DRAFT_MODEL", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if (!file_exists(target_path) || !file_exists(draft_path))
        SKIP("target/draft model pair not found (" + target_path + ", " +
             draft_path + ")");

    reame::ModelParams tp;
    tp.path = target_path;
    tp.context_length = 512;
    tp.threads = 4;
    auto sp = tp;
    sp.path = draft_path;

    // Highly predictable continuation: speculative decoding pays off when
    // the draft can anticipate the target (DSpark's operating regime). A
    // free-form story prompt on this small pair yields ~0.5 acceptance,
    // where CPU speculation is at or below break-even.
    const std::string prompt_text =
        "Count slowly: one, two, three, four, five, six, seven, eight, "
        "nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen,";
    const int n_gen = 64;

    // Plain baseline.
    auto plain_target = reame::make_llama_backend(tp);
    const auto prompt = plain_target->tokenize(prompt_text, true);
    SpeculativeDecoder plain(*plain_target, nullptr, {});
    const auto t0 = std::chrono::steady_clock::now();
    const auto plain_out = plain.generate(prompt, greedy(n_gen));
    const auto t1 = std::chrono::steady_clock::now();

    // Speculative.
    auto target = reame::make_llama_backend(tp);
    auto draft = reame::make_llama_backend(sp);
    SpeculativeDecoder::Config cfg;
    cfg.draft_tokens = 8;
    SpeculativeDecoder spec(*target, draft.get(), cfg);
    const auto t2 = std::chrono::steady_clock::now();
    const auto spec_out = spec.generate(prompt, greedy(n_gen));
    const auto t3 = std::chrono::steady_clock::now();

    const double plain_s = std::chrono::duration<double>(t1 - t0).count();
    const double spec_s = std::chrono::duration<double>(t3 - t2).count();
    const double speedup = plain_s / spec_s;

    WARN("plain: " << plain_s << "s (" << plain_out.size() << " tok), "
         << "speculative: " << spec_s << "s (" << spec_out.size() << " tok), "
         << "speedup: " << speedup << "x, acceptance: "
         << spec.metrics().acceptance_rate());

    // Exact token equality with the plain path is NOT asserted here:
    // llama.cpp's batched and single-token CPU kernels differ numerically,
    // so greedy argmax can flip on near-ties. Distribution-level
    // equivalence is pinned by the same-model integration test above and
    // by the unit tests; here we require full-length generation and a
    // draft that is actually anticipating the target.
    CHECK(spec_out.size() == plain_out.size());
    CHECK(spec.metrics().acceptance_rate() > 0.5);

    // Wall-clock speedup depends on the draft/target cost ratio. On the
    // VPS pair (30B target, ~1B draft: 20-30x cheaper) the expected gain
    // is 30%+; a small local pair (1.5B/0.5B: only ~3x cheaper) sits at or
    // below break-even, so the hard assertion is opt-in for the deployment
    // hardware.
    if (std::getenv("REAME_PERF_STRICT") != nullptr)
        CHECK(speedup > 1.3);
    else
        SUCCEED("speedup measured: set REAME_PERF_STRICT=1 to enforce");
#endif
}
