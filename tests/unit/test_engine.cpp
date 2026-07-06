// Isolated tests for sovrano::core::SovranoEngine over MockBackend.
// Generation is driven with scripted logits (decode_queue) and greedy
// sampling so every expected output is derived by hand.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../mock/llama_mock.hpp"
#include "sovrano/core/engine.hpp"
#include "sovrano/speculative/speculative_decoder.hpp"

using sovrano::TokenId;
using sovrano::test::MockBackend;
using sovrano::core::EngineError;
using sovrano::core::GenerationConfig;
using sovrano::core::SovranoEngine;

namespace {

SovranoEngine::Config valid_config() {
    SovranoEngine::Config c;
    c.model_path = "/models/test.gguf";
    c.n_ctx = 2048;
    c.n_threads = 4;
    return c;
}

GenerationConfig greedy(int max_tokens = 16) {
    GenerationConfig g;
    g.temperature = 0.0f;
    g.repeat_penalty = 1.0f;
    g.max_tokens = max_tokens;
    return g;
}

std::pair<SovranoEngine, MockBackend*> make_engine(
    const SovranoEngine::Config& cfg = valid_config()) {
    auto backend = std::make_unique<MockBackend>();
    MockBackend* raw = backend.get();
    return {SovranoEngine(cfg, std::move(backend)), raw};
}

// Logits with the argmax at `idx` (vocab of `n`).
std::vector<float> peak(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 10.0f;
    return v;
}

// Standard scripted mock: prompt -> {1, 2}, vocab 5, EOS 4,
// steps: token 3 ("foo"), token 0 ("bar"), then EOS.
void script_foobar(MockBackend* mock) {
    mock->vocab_size_value = 5;
    mock->eos_token_value = 4;
    mock->tokenize_result = {1, 2};
    mock->piece_map = {{3, "foo"}, {0, "bar"}};
    mock->decode_queue = {peak(5, 3), peak(5, 0), peak(5, 4)};
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_CASE("engine constructor rejects invalid config") {
    auto cfg = valid_config();

    SECTION("empty model path") { cfg.model_path.clear(); }
    SECTION("non-positive n_ctx") { cfg.n_ctx = 0; }
    SECTION("non-positive n_threads") { cfg.n_threads = 0; }

    CHECK_THROWS_AS(SovranoEngine(cfg, std::make_unique<MockBackend>()),
                    EngineError);
}

TEST_CASE("engine constructor rejects null backend") {
    CHECK_THROWS_AS(SovranoEngine(valid_config(), nullptr), EngineError);
}

TEST_CASE("context_size and vocab_size come from the backend") {
    auto [engine, mock] = make_engine();
    mock->vocab_size_value = 5;
    mock->context_length_value = 128;

    CHECK(engine.vocab_size() == 5);
    CHECK(engine.context_size() == 128);
}

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------

TEST_CASE("generate: prefill once, append per token, stop at EOS") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    const auto out = engine.generate("hi", greedy());

    CHECK(out == "foobar");
    // One fresh prefill with the prompt tokens...
    REQUIRE(mock->decode_calls.size() == 1);
    CHECK(mock->decode_calls[0] == std::vector<TokenId>{1, 2});
    // ...then one append per accepted token (EOS is never appended).
    REQUIRE(mock->decode_append_calls.size() == 2);
    CHECK(mock->decode_append_calls[0] == std::vector<TokenId>{3});
    CHECK(mock->decode_append_calls[1] == std::vector<TokenId>{0});
}

TEST_CASE("generate honours max_tokens") {
    auto [engine, mock] = make_engine();
    mock->vocab_size_value = 5;
    mock->eos_token_value = 4;
    mock->tokenize_result = {1};
    mock->piece_map = {{3, "x"}};
    mock->decode_result = peak(5, 3);  // argmax 3 forever

    const auto out = engine.generate("hi", greedy(/*max_tokens=*/3));

    CHECK(out == "xxx");
    // After the 3rd token the loop must stop WITHOUT another decode.
    CHECK(mock->decode_append_calls.size() == 2);
}

TEST_CASE("generate with echo_prompt prepends the prompt") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    auto g = greedy();
    g.echo_prompt = true;

    CHECK(engine.generate("hi", g) == "hifoobar");
}

TEST_CASE("generate stops when the context is full") {
    auto [engine, mock] = make_engine();
    mock->vocab_size_value = 5;
    mock->eos_token_value = 4;
    mock->context_length_value = 3;
    mock->tokenize_result = {1, 2};  // prompt already uses 2 of 3 slots
    mock->piece_map = {{3, "x"}};
    mock->decode_result = peak(5, 3);

    const auto out = engine.generate("hi", greedy(/*max_tokens=*/100));

    // Only one token fits; no decode_append after the context filled up.
    CHECK(out == "x");
    CHECK(mock->decode_append_calls.empty());
}

TEST_CASE("generate rejects a prompt longer than the context") {
    auto [engine, mock] = make_engine();
    mock->context_length_value = 2;
    mock->tokenize_result = {1, 2, 3};

    CHECK_THROWS_AS(engine.generate("too long", greedy()), EngineError);
    CHECK(mock->decode_calls.empty());
}

TEST_CASE("generate rejects a prompt that tokenizes to nothing") {
    auto [engine, mock] = make_engine();
    mock->tokenize_result = {};

    CHECK_THROWS_AS(engine.generate("", greedy()), EngineError);
}

// ---------------------------------------------------------------------------
// generate_stream
// ---------------------------------------------------------------------------

TEST_CASE("generate_stream delivers pieces and respects a false return") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    std::vector<std::string> pieces;
    engine.generate_stream(
        "hi",
        [&](const std::string& p) {
            pieces.push_back(p);
            return pieces.size() < 1;  // stop after the first piece
        },
        greedy());

    CHECK(pieces == std::vector<std::string>{"foo"});
    // Stopped by the callback: no further decode.
    CHECK(mock->decode_append_calls.empty());
}

TEST_CASE("generate_stream with echo_prompt emits the prompt first") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    auto g = greedy();
    g.echo_prompt = true;

    std::vector<std::string> pieces;
    engine.generate_stream(
        "hi", [&](const std::string& p) { pieces.push_back(p); return true; },
        g);

    CHECK(pieces == std::vector<std::string>{"hi", "foo", "bar"});
}

// ---------------------------------------------------------------------------
// Speculative integration
// ---------------------------------------------------------------------------

namespace {

std::vector<float> spec_peak(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 10.0f;
    return v;
}

// Target/draft pair scripted so speculative greedy emits "ab" then stops:
// prompt -> {0}; draft proposes {1, 2} (accepted), then EOS.
void script_spec(MockBackend* target, MockBackend* draft) {
    for (MockBackend* m : {target, draft}) {
        m->vocab_size_value = 6;
        m->eos_token_value = 5;
        m->tokenize_result = {0};
    }
    target->piece_map = {{1, "a"}, {2, "b"}};
    draft->decode_queue = {spec_peak(6, 1), spec_peak(6, 2), spec_peak(6, 5)};
    target->decode_batch_queue = {{spec_peak(6, 1), spec_peak(6, 2)},
                                  {spec_peak(6, 5)}};
}

}  // namespace

TEST_CASE("engine with a draft backend generates through the speculative decoder") {
    auto cfg = valid_config();
    cfg.draft_tokens = 2;
    auto target = std::make_unique<MockBackend>();
    auto draft = std::make_unique<MockBackend>();
    MockBackend* target_raw = target.get();
    MockBackend* draft_raw = draft.get();
    script_spec(target_raw, draft_raw);

    SovranoEngine engine(cfg, std::move(target), std::move(draft));

    CHECK(engine.generate("hi", greedy()) == "ab");
    CHECK_FALSE(target_raw->decode_batch_calls.empty());  // speculative path
    REQUIRE(engine.speculative_metrics() != nullptr);
    // Drafted {1, 2} then {EOS}; all three pass verification (the accepted
    // EOS ends generation and is never emitted).
    CHECK(engine.speculative_metrics()->total_accepted_tokens == 3);
    CHECK(engine.speculative_metrics()->generated_tokens == 2);
}

TEST_CASE("engine ignores the draft backend when use_speculative is off") {
    auto cfg = valid_config();
    cfg.use_speculative = false;
    auto target = std::make_unique<MockBackend>();
    auto draft = std::make_unique<MockBackend>();
    MockBackend* target_raw = target.get();
    script_foobar(target_raw);

    SovranoEngine engine(cfg, std::move(target), std::move(draft));

    CHECK(engine.generate("hi", greedy()) == "foobar");
    CHECK(target_raw->decode_batch_calls.empty());  // classic path
    CHECK(engine.speculative_metrics() == nullptr);
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

TEST_CASE("sessions: ids are unique, save/load round-trips the context") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    const auto s1 = engine.create_session();
    const auto s2 = engine.create_session();
    CHECK(s1 != s2);

    engine.generate("hi", greedy());  // context: {1, 2, 3, 0}
    engine.save_session(s1);

    mock->decode_calls.clear();
    engine.load_session(s1);

    // load re-prefills the model with the saved tokens.
    REQUIRE(mock->decode_calls.size() == 1);
    CHECK(mock->decode_calls[0] == std::vector<TokenId>{1, 2, 3, 0});
}

TEST_CASE("sessions: unknown or deleted ids throw") {
    auto [engine, mock] = make_engine();

    CHECK_THROWS_AS(engine.save_session("nope"), EngineError);
    CHECK_THROWS_AS(engine.load_session("nope"), EngineError);
    CHECK_THROWS_AS(engine.delete_session("nope"), EngineError);

    const auto id = engine.create_session();
    engine.save_session(id);
    engine.delete_session(id);
    CHECK_THROWS_AS(engine.load_session(id), EngineError);
}

// ---------------------------------------------------------------------------
// Prompt cache
// ---------------------------------------------------------------------------

namespace {

struct CacheTempDir {
    std::filesystem::path path;
    CacheTempDir() {
        path = std::filesystem::temp_directory_path() /
               ("sovrano-engine-cache-" + std::to_string(counter++));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~CacheTempDir() { std::filesystem::remove_all(path); }
    static int counter;
};
int CacheTempDir::counter = 0;

// Prompt tokenizes to {1, 2}; the model then emits "tok" (3) and EOS (4).
// `with_prefill_logits` adds the entry consumed by the cold prompt prefill.
void script_cached(MockBackend* m, bool with_prefill_logits) {
    m->vocab_size_value = 5;
    m->eos_token_value = 4;
    m->tokenize_result = {1, 2};
    m->piece_map = {{3, "tok"}};
    if (with_prefill_logits) m->decode_queue.push_back(peak(5, 0));  // filler
    m->decode_queue.push_back(peak(5, 3));
    m->decode_queue.push_back(peak(5, 4));
}

}  // namespace

TEST_CASE("prompt cache: cold run snapshots the prefix, warm run skips prefill") {
    CacheTempDir dir;
    auto cfg = valid_config();
    cfg.cache_dir = dir.path.string();

    // Cold: prefix {1} decoded, state stored, then generation.
    {
        auto backend = std::make_unique<MockBackend>();
        MockBackend* mock = backend.get();
        script_cached(mock, /*with_prefill_logits=*/true);
        mock->state_data_result = {'S', '1'};

        SovranoEngine engine(cfg, std::move(backend));
        CHECK(engine.generate("hi", greedy()) == "tok");

        CHECK(mock->reset_calls == 1);
        CHECK(mock->state_data_calls == 1);       // snapshot taken
        CHECK(mock->decode_calls.empty());        // split prefill, no decode()
        REQUIRE(mock->decode_append_calls.size() == 3);
        CHECK(mock->decode_append_calls[0] == std::vector<TokenId>{1});
        CHECK(mock->decode_append_calls[1] == std::vector<TokenId>{2});
    }

    // Warm: fresh engine + backend, same cache dir -> state restored, the
    // prefix is never decoded again.
    {
        auto backend = std::make_unique<MockBackend>();
        MockBackend* mock = backend.get();
        script_cached(mock, /*with_prefill_logits=*/false);

        SovranoEngine engine(cfg, std::move(backend));
        CHECK(engine.generate("hi", greedy()) == "tok");

        REQUIRE(mock->set_state_calls.size() == 1);
        CHECK(mock->set_state_calls[0].first == std::vector<char>{'S', '1'});
        CHECK(mock->set_state_calls[0].second == 1);  // prefix length
        CHECK(mock->state_data_calls == 0);           // hit: nothing stored
        REQUIRE(mock->decode_append_calls.size() == 2);
        CHECK(mock->decode_append_calls[0] == std::vector<TokenId>{2});
        CHECK(mock->decode_append_calls[1] == std::vector<TokenId>{3});
    }
}

TEST_CASE("sessions: with a cache dir, load restores the snapshot instead of re-prefilling") {
    CacheTempDir dir;
    auto cfg = valid_config();
    cfg.cache_dir = dir.path.string();

    auto backend = std::make_unique<MockBackend>();
    MockBackend* mock = backend.get();
    script_cached(mock, true);
    mock->state_data_result = {'K', 'V'};

    SovranoEngine engine(cfg, std::move(backend));
    engine.generate("hi", greedy());  // context: {1, 2, 3}

    const auto id = engine.create_session();
    engine.save_session(id);

    mock->decode_calls.clear();
    mock->set_state_calls.clear();
    engine.load_session(id);

    // Snapshot restored; no re-prefill decode.
    REQUIRE(mock->set_state_calls.size() == 1);
    CHECK(mock->set_state_calls[0].first == std::vector<char>{'K', 'V'});
    CHECK(mock->set_state_calls[0].second == 3);
    CHECK(mock->decode_calls.empty());
}

TEST_CASE("without a cache dir the classic prefill path is unchanged") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    CHECK(engine.generate("hi", greedy()) == "foobar");
    CHECK(mock->state_data_calls == 0);
    CHECK(mock->set_state_calls.empty());
    REQUIRE(mock->decode_calls.size() == 1);  // single full-prompt prefill
}

// ---------------------------------------------------------------------------
// Integration (real llama.cpp + TinyLlama). SKIPs when unavailable.
// ---------------------------------------------------------------------------

namespace {

std::string integration_model_path() {
    if (const char* env = std::getenv("SOVRANO_TEST_MODEL")) return env;
    return "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

}  // namespace

TEST_CASE("[integration] prompt cache reproduces the cold output and hits disk",
          "[integration]") {
#ifndef SOVRANO_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    CacheTempDir dir;
    SovranoEngine::Config cfg;
    cfg.model_path = path;
    cfg.n_ctx = 256;
    cfg.n_threads = 4;
    cfg.cache_dir = dir.path.string();

    const std::string prompt =
        "The lighthouse keeper counted the ships as they passed: first";

    std::string cold, warm;
    {
        SovranoEngine engine(cfg);
        cold = engine.generate(prompt, greedy(8));
    }
    // A snapshot landed on disk.
    CHECK(std::filesystem::directory_iterator(dir.path) !=
          std::filesystem::directory_iterator{});
    {
        SovranoEngine engine(cfg);  // fresh process-equivalent, same cache
        warm = engine.generate(prompt, greedy(8));
    }

    CHECK(!cold.empty());
    // Same split-prefill code path cold and warm -> identical numerics.
    CHECK(warm == cold);
#endif
}

TEST_CASE("[integration] engine generates deterministic greedy text",
          "[integration]") {
#ifndef SOVRANO_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    SovranoEngine::Config cfg;
    cfg.model_path = path;
    cfg.n_ctx = 256;
    cfg.n_threads = 4;
    SovranoEngine engine(cfg);

    const auto out1 = engine.generate("The capital of Italy is", greedy(8));
    CHECK(!out1.empty());

    // Streaming pieces must preserve word boundaries: 8 tokens of English
    // continuation contain at least one space. (Regression: detokenizing
    // token-by-token via llama_detokenize strips leading spaces.)
    CHECK(out1.find(' ') != std::string::npos);

    // Greedy is deterministic: a second run must match exactly.
    const auto out2 = engine.generate("The capital of Italy is", greedy(8));
    CHECK(out1 == out2);
#endif
}
