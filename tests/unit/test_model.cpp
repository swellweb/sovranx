// Isolated tests for sovrano::LlamaModel and ModelParams::from_config.
// Every llama.cpp dependency is replaced by MockBackend; the only test that
// touches the real backend is the integration section at the bottom, which
// SKIPs when no model file is available.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <utility>

#include "../mock/llama_mock.hpp"
#include "sovrano/core/model.hpp"
#include "sovrano/utils/config.hpp"

using sovrano::Config;
using sovrano::LlamaModel;
using sovrano::ModelError;
using sovrano::ModelParams;
using sovrano::TokenId;
using sovrano::test::MockBackend;

namespace {

ModelParams valid_params() {
    ModelParams p;
    p.path = "/models/test.gguf";
    p.context_length = 2048;
    p.threads = 4;
    return p;
}

// Builds a LlamaModel over a mock, returning the raw mock pointer for
// assertions (ownership goes to the model).
std::pair<LlamaModel, MockBackend*> make_mock_model(const ModelParams& p) {
    auto backend = std::make_unique<MockBackend>();
    MockBackend* raw = backend.get();
    return {LlamaModel(p, std::move(backend)), raw};
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelParams::from_config
// ---------------------------------------------------------------------------

TEST_CASE("from_config maps config keys") {
    const auto cfg = Config::parse_string(
        "[model]\n"
        "path = /opt/models/m.gguf\n"
        "context_length = 8192\n"
        "threads = 14\n"
        "[memory]\n"
        "use_mmap = false\n"
        "use_mlock = true\n"
        "kv_cache_type = q8_0\n");

    const auto p = ModelParams::from_config(cfg);

    CHECK(p.path == "/opt/models/m.gguf");
    CHECK(p.context_length == 8192);
    CHECK(p.threads == 14);
    CHECK(p.use_mmap == false);
    CHECK(p.use_mlock == true);
    CHECK(p.kv_cache_type == "q8_0");
}

TEST_CASE("from_config defaults kv_cache_type to f16 and validates it") {
    const auto cfg = Config::parse_string("[model]\npath = /m.gguf\n");
    CHECK(ModelParams::from_config(cfg).kv_cache_type == "f16");

    const auto bad = Config::parse_string(
        "[model]\npath = /m.gguf\n[memory]\nkv_cache_type = q2_banana\n");
    CHECK_THROWS_AS(ModelParams::from_config(bad), ModelError);
}

TEST_CASE("from_config applies defaults for optional keys") {
    const auto cfg = Config::parse_string("[model]\npath = /m.gguf\n");

    const auto p = ModelParams::from_config(cfg);

    CHECK(p.context_length == 4096);
    CHECK(p.threads > 0);  // hardware-dependent, but always positive
    CHECK(p.use_mmap == true);
    CHECK(p.use_mlock == false);
}

TEST_CASE("from_config requires model.path") {
    const auto cfg = Config::parse_string("[model]\ncontext_length = 512\n");
    CHECK_THROWS_AS(ModelParams::from_config(cfg), ModelError);
}

// ---------------------------------------------------------------------------
// LlamaModel construction / validation
// ---------------------------------------------------------------------------

TEST_CASE("constructor rejects invalid params") {
    auto p = valid_params();

    SECTION("empty path") { p.path.clear(); }
    SECTION("non-positive context") { p.context_length = 0; }
    SECTION("negative context") { p.context_length = -1; }
    SECTION("non-positive threads") { p.threads = 0; }

    CHECK_THROWS_AS(LlamaModel(p, std::make_unique<MockBackend>()), ModelError);
}

TEST_CASE("constructor rejects null injected backend") {
    CHECK_THROWS_AS(LlamaModel(valid_params(), nullptr), ModelError);
}

TEST_CASE("exposes backend vocab size and context length") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->vocab_size_value = 32000;
    mock->context_length_value = 2048;

    CHECK(model.vocab_size() == 32000);
    CHECK(model.context_length() == 2048);
}

// ---------------------------------------------------------------------------
// tokenize / detokenize
// ---------------------------------------------------------------------------

TEST_CASE("tokenize delegates text and add_special flag") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->tokenize_result = {1, 15043, 3186};

    const auto tokens = model.tokenize("hello world", /*add_special=*/true);

    CHECK(tokens == std::vector<TokenId>{1, 15043, 3186});
    REQUIRE(mock->tokenize_calls.size() == 1);
    CHECK(mock->tokenize_calls[0].first == "hello world");
    CHECK(mock->tokenize_calls[0].second == true);

    model.tokenize("x", /*add_special=*/false);
    CHECK(mock->tokenize_calls[1].second == false);
}

TEST_CASE("detokenize delegates valid tokens") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->detokenize_result = "hello world";

    const auto text = model.detokenize({1, 15043, 3186});

    CHECK(text == "hello world");
    REQUIRE(mock->detokenize_calls.size() == 1);
    CHECK(mock->detokenize_calls[0] == std::vector<TokenId>{1, 15043, 3186});
}

TEST_CASE("detokenize of empty vector is empty string, no backend call") {
    auto [model, mock] = make_mock_model(valid_params());

    CHECK(model.detokenize({}) == "");
    CHECK(mock->detokenize_calls.empty());
}

TEST_CASE("detokenize rejects out-of-range tokens") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->vocab_size_value = 100;

    CHECK_THROWS_AS(model.detokenize({0, 100}), ModelError);  // == vocab_size
    CHECK_THROWS_AS(model.detokenize({-1}), ModelError);
    CHECK(mock->detokenize_calls.empty());
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

TEST_CASE("forward returns last-token logits from backend") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->vocab_size_value = 4;
    mock->decode_result = {0.1f, 0.2f, 0.3f, 0.4f};

    const auto logits = model.forward({1, 2, 3});

    CHECK(logits == std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
    REQUIRE(mock->decode_calls.size() == 1);
    CHECK(mock->decode_calls[0] == std::vector<TokenId>{1, 2, 3});
}

TEST_CASE("forward rejects empty input") {
    auto [model, mock] = make_mock_model(valid_params());
    CHECK_THROWS_AS(model.forward({}), ModelError);
    CHECK(mock->decode_calls.empty());
}

TEST_CASE("forward rejects input longer than context") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->context_length_value = 4;

    CHECK_THROWS_AS(model.forward({1, 2, 3, 4, 5}), ModelError);
    CHECK(mock->decode_calls.empty());

    // Exactly at the limit is allowed.
    mock->vocab_size_value = 1;
    mock->decode_result = {0.5f};
    CHECK_NOTHROW(model.forward({1, 2, 3, 4}));
}

TEST_CASE("forward rejects logits/vocab size mismatch from backend") {
    auto [model, mock] = make_mock_model(valid_params());
    mock->vocab_size_value = 4;
    mock->decode_result = {0.1f, 0.2f};  // wrong size

    CHECK_THROWS_AS(model.forward({1}), ModelError);
}

// ---------------------------------------------------------------------------
// Integration (real llama.cpp + real model file). SKIPs when unavailable.
// Model path: $SOVRANO_TEST_MODEL or models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
// (see scripts/download_models.sh).
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

TEST_CASE("[integration] multi-sequence decode matches sequential decodes",
          "[integration]") {
#ifndef SOVRANO_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    ModelParams p;
    p.path = path;
    p.context_length = 512;
    p.threads = 4;

    // Reference: two prompts decoded sequentially, single-sequence.
    auto ref = sovrano::make_llama_backend(p);
    const auto prompt_a = ref->tokenize("The capital of Italy is", true);
    const auto prompt_b = ref->tokenize("Water boils at a temperature of", true);
    const auto ref_a = ref->decode(prompt_a);
    const auto ref_b = ref->decode(prompt_b);

    // Same two prompts as ONE interleaved multi-seq batch.
    auto pp = p;
    pp.n_seq_max = 2;
    auto multi = sovrano::make_llama_backend(pp);
    const auto outs = multi->decode_seqs({{0, prompt_a, 0}, {1, prompt_b, 0}});

    REQUIRE(outs.size() == 2);
    // Same argmax as the sequential reference (bitwise logits may differ
    // slightly across batch layouts; the chosen token must not).
    const auto argmax = [](const std::vector<float>& v) {
        return std::distance(v.begin(),
                             std::max_element(v.begin(), v.end()));
    };
    CHECK(argmax(outs[0]) == argmax(ref_a));
    CHECK(argmax(outs[1]) == argmax(ref_b));

    // Sequences continue independently: one more token each, then drop
    // seq 0 and keep decoding seq 1.
    const auto ta = static_cast<TokenId>(argmax(outs[0]));
    const auto tb = static_cast<TokenId>(argmax(outs[1]));
    const auto next = multi->decode_seqs(
        {{0, {ta}, static_cast<std::uint32_t>(prompt_a.size())},
         {1, {tb}, static_cast<std::uint32_t>(prompt_b.size())}});
    REQUIRE(next.size() == 2);

    multi->clear_seq(0);
    const auto tb2 = static_cast<TokenId>(argmax(next[1]));
    const auto solo = multi->decode_seqs(
        {{1, {tb2}, static_cast<std::uint32_t>(prompt_b.size()) + 1}});
    REQUIRE(solo.size() == 1);
    REQUIRE(solo[0].size() == static_cast<std::size_t>(multi->vocab_size()));
#endif
}

TEST_CASE("[integration] real model: tokenize round-trip and forward pass",
          "[integration]") {
#ifndef SOVRANO_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    ModelParams p;
    p.path = path;
    p.context_length = 512;
    p.threads = 4;

    LlamaModel model(p);

    CHECK(model.vocab_size() > 0);
    CHECK(model.context_length() == 512);

    const auto tokens = model.tokenize("The capital of Italy is", true);
    REQUIRE(tokens.size() >= 2);

    // Round-trip must contain the original text (BOS handling may add
    // leading markup, so check containment, not equality).
    const auto text = model.detokenize(tokens);
    CHECK(text.find("The capital of Italy is") != std::string::npos);

    const auto logits = model.forward(tokens);
    REQUIRE(logits.size() == static_cast<std::size_t>(model.vocab_size()));

    // Sanity: logits must not be all-zero/NaN.
    bool any_nonzero = false;
    for (float v : logits) {
        REQUIRE(v == v);  // not NaN
        if (v != 0.0f) any_nonzero = true;
    }
    CHECK(any_nonzero);
#endif
}
