// Isolated tests for reame::core::ReameEngine over MockBackend.
// Generation is driven with scripted logits (decode_queue) and greedy
// sampling so every expected output is derived by hand.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../mock/llama_mock.hpp"
#include "reame/cache/cache_manager.hpp"
#include "reame/core/engine.hpp"
#include "reame/speculative/speculative_decoder.hpp"

using reame::TokenId;
using reame::test::MockBackend;
using reame::core::EngineError;
using reame::core::GenerationConfig;
using reame::core::ReameEngine;

namespace {

ReameEngine::Config valid_config() {
    ReameEngine::Config c;
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

std::pair<ReameEngine, MockBackend*> make_engine(
    const ReameEngine::Config& cfg = valid_config()) {
    auto backend = std::make_unique<MockBackend>();
    MockBackend* raw = backend.get();
    return {ReameEngine(cfg, std::move(backend)), raw};
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
    SECTION("invalid kv_cache_type") { cfg.kv_cache_type = "q2_banana"; }

    CHECK_THROWS_AS(ReameEngine(cfg, std::make_unique<MockBackend>()),
                    EngineError);
}

TEST_CASE("engine constructor rejects null backend") {
    CHECK_THROWS_AS(ReameEngine(valid_config(), nullptr), EngineError);
}

TEST_CASE("count_tokens returns the tokenizer's token count") {
    auto [engine, mock] = make_engine();
    mock->tokenize_result = {1, 2, 3};

    CHECK(engine.count_tokens("some text") == 3);
    REQUIRE(mock->tokenize_calls.size() == 1);
    CHECK(mock->tokenize_calls[0].first == "some text");
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

TEST_CASE("format_chat delegates to the model's own template") {
    auto [engine, mock] = make_engine();
    CHECK(engine.format_chat("hello") == "<U>hello</U><A>");
    REQUIRE(mock->format_chat_calls.size() == 1);
    CHECK(mock->format_chat_calls[0] == "hello");
}

TEST_CASE("format_chat passes through when the model has no template") {
    auto [engine, mock] = make_engine();
    mock->chat_template_empty = true;
    // Template-less model: raw completion is the correct behavior.
    CHECK(engine.format_chat("hello") == "hello");
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

    ReameEngine engine(cfg, std::move(target), std::move(draft));

    CHECK(engine.generate("hi", greedy()) == "ab");
    CHECK_FALSE(target_raw->decode_batch_calls.empty());  // speculative path
    REQUIRE(engine.speculative_metrics() != nullptr);
    // Drafted {1, 2} then {EOS}; all three pass verification (the accepted
    // EOS ends generation and is never emitted).
    CHECK(engine.speculative_metrics()->total_accepted_tokens == 3);
    CHECK(engine.speculative_metrics()->generated_tokens == 2);
}

TEST_CASE("engine with prompt lookup speculates without a draft backend") {
    auto cfg = valid_config();
    cfg.use_prompt_lookup = true;
    cfg.draft_tokens = 2;

    auto target = std::make_unique<MockBackend>();
    MockBackend* mock = target.get();
    mock->vocab_size_value = 6;
    mock->eos_token_value = 5;
    // Prompt repeats: {1,2,1,2} -> lookup proposes {1,2}; target accepts
    // then wants EOS.
    mock->tokenize_result = {1, 2, 1, 2};
    mock->piece_map = {{1, "x"}, {2, "y"}};
    mock->decode_queue = {spec_peak(6, 0)};  // prefill filler
    mock->decode_batch_queue = {{spec_peak(6, 1), spec_peak(6, 2)},
                                {spec_peak(6, 5)}};

    ReameEngine engine(cfg, std::move(target));

    CHECK(engine.generate("rep", greedy()) == "xy");
    CHECK_FALSE(mock->decode_batch_calls.empty());  // speculating...
    REQUIRE(engine.speculative_metrics() != nullptr);
    CHECK(engine.speculative_metrics()->total_accepted_tokens >= 2);
}

TEST_CASE("engine ignores the draft backend when use_speculative is off") {
    auto cfg = valid_config();
    cfg.use_speculative = false;
    auto target = std::make_unique<MockBackend>();
    auto draft = std::make_unique<MockBackend>();
    MockBackend* target_raw = target.get();
    script_foobar(target_raw);

    ReameEngine engine(cfg, std::move(target), std::move(draft));

    CHECK(engine.generate("hi", greedy()) == "foobar");
    CHECK(target_raw->decode_batch_calls.empty());  // classic path
    CHECK(engine.speculative_metrics() == nullptr);
}

// ---------------------------------------------------------------------------
// Parallel (interleaved multi-user) mode
// ---------------------------------------------------------------------------

TEST_CASE("parallel engine serves two concurrent generations") {
    auto cfg = valid_config();
    cfg.n_parallel = 2;

    auto backend = std::make_unique<MockBackend>();
    MockBackend* mock = backend.get();
    mock->vocab_size_value = 6;
    mock->eos_token_value = 5;
    mock->tokenize_result = {1, 2};
    mock->piece_map = {{3, "x"}, {4, "y"}};
    // Every sequence — however the two requests interleave, and even when
    // one reuses the other's freed slot — runs the same fresh script:
    // "x", "y", then EOS. seq_template re-seeds a slot on reuse.
    mock->seq_template = {peak(6, 3), peak(6, 4), peak(6, 5)};
    mock->seq_decode_queues[0] = mock->seq_template;
    mock->seq_decode_queues[1] = mock->seq_template;

    ReameEngine engine(cfg, std::move(backend));
    CHECK(engine.parallel_capable());

    std::string out1, out2;
    std::exception_ptr e1, e2;
    std::thread t1([&] {
        try { out1 = engine.generate("hi", greedy()); }
        catch (...) { e1 = std::current_exception(); }
    });
    std::thread t2([&] {
        try { out2 = engine.generate("hi", greedy()); }
        catch (...) { e2 = std::current_exception(); }
    });
    t1.join();
    t2.join();
    if (e1) std::rethrow_exception(e1);
    if (e2) std::rethrow_exception(e2);

    CHECK(out1 == "xy");
    CHECK(out2 == "xy");
    CHECK_FALSE(mock->decode_seqs_calls.empty());
}

TEST_CASE("conclave: parallel attempts elect a consensus answer") {
    auto cfg = valid_config();
    cfg.n_parallel = 3;

    auto backend = std::make_unique<MockBackend>();
    MockBackend* mock = backend.get();
    mock->vocab_size_value = 6;
    mock->eos_token_value = 5;
    mock->tokenize_result = {1, 2};
    mock->piece_map = {{3, "x"}, {4, "y"}};
    // Every attempt (and every reused slot) follows the same script:
    // "x", "y", EOS — with peaked logits the mild conclave temperature
    // still samples the peak, so all attempts answer "xy".
    mock->seq_template = {peak(6, 3), peak(6, 4), peak(6, 5)};
    for (int s = 0; s < 3; ++s) mock->seq_decode_queues[s] = mock->seq_template;

    ReameEngine engine(cfg, std::move(backend));
    CHECK(engine.generate_best("hi", greedy(), 3) == "xy");
    CHECK_FALSE(mock->decode_seqs_calls.empty());  // attempts interleaved
}

TEST_CASE("conclave: n=1 degenerates to a plain generate") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);
    CHECK(engine.generate_best("hi", greedy(), 1) == "foobar");
}

TEST_CASE("conclave: early consensus skips the remaining attempts") {
    // Sequential conclave (n_parallel=1), n=3. The first two attempts both
    // answer "8": an absolute majority (2 of 3) — the third attempt must
    // never run. The script holds exactly two attempts; a third would fall
    // back to decode_result ("Z") and betray itself in decode_calls.
    auto [engine, mock] = make_engine();
    mock->vocab_size_value = 6;
    mock->eos_token_value = 5;
    mock->tokenize_result = {1, 2};
    mock->piece_map = {{3, "8"}, {0, "Z"}};
    mock->decode_queue = {peak(6, 3), peak(6, 5),   // attempt 0: "8", EOS
                          peak(6, 3), peak(6, 5)};  // attempt 1: "8", EOS
    mock->decode_result = peak(6, 0);

    CHECK(engine.generate_best("hi", greedy(), 3) == "8");
    // One prefill per attempt: exactly 2 attempts ran.
    CHECK(mock->decode_calls.size() == 2);
}

TEST_CASE("conclave: reports how many candidates agreed") {
    // Sequential conclave, n=3: attempts answer "8", "8", (stopped early
    // at 2 votes = absolute majority of 3) -> votes 2. A split conclave
    // ("7", "8", "9") reports 1: no two candidates agreed.
    auto [engine, mock] = make_engine();
    mock->vocab_size_value = 6;
    mock->eos_token_value = 5;
    mock->tokenize_result = {1, 2};
    mock->piece_map = {{2, "7"}, {3, "8"}, {4, "9"}};
    mock->decode_queue = {peak(6, 3), peak(6, 5),   // "8"
                          peak(6, 3), peak(6, 5)};  // "8" -> majority
    int votes = 0;
    CHECK(engine.generate_best("hi", greedy(), 3, &votes) == "8");
    CHECK(votes == 2);

    mock->decode_queue = {peak(6, 2), peak(6, 5),   // "7"
                          peak(6, 3), peak(6, 5),   // "8"
                          peak(6, 4), peak(6, 5)};  // "9"
    CHECK(engine.generate_best("hi", greedy(), 3, &votes) == "7");
    CHECK(votes == 1);
}

TEST_CASE("conclave: consensus reached in parallel stops the straggler") {
    auto cfg = valid_config();
    cfg.n_parallel = 3;

    auto backend = std::make_unique<MockBackend>();
    MockBackend* mock = backend.get();
    mock->vocab_size_value = 6;
    mock->eos_token_value = 5;
    mock->tokenize_result = {1, 2};
    mock->piece_map = {{3, "8"}, {4, "9"}};
    // Two slots answer "8" in one step; the third rambles "9" for 200
    // steps. Majority (2 of 3) lands after the fast pair finishes: the
    // straggler must be cut off long before draining its script. Steps
    // are paced (as with a real model) so the verdict lands between
    // steps, not after the whole script has drained at mock speed. Slot
    // reuse (a late attempt landing on a freed slot) re-seeds from the
    // fast template, which only reinforces the majority.
    mock->decode_seqs_delay_ms = 5;
    mock->seq_template = {peak(6, 3), peak(6, 5)};
    mock->seq_decode_queues[0] = mock->seq_template;
    mock->seq_decode_queues[1] = mock->seq_template;
    std::deque<std::vector<float>> slow(600, peak(6, 4));
    slow.push_back(peak(6, 5));
    mock->seq_decode_queues[2] = slow;

    ReameEngine engine(cfg, std::move(backend));
    auto gen = greedy(/*max_tokens=*/700);
    CHECK(engine.generate_best("hi", gen, 3) == "8");
    // Early stop: a run to the end of the straggler's script would take
    // 600+ interleaved steps. The generous bound absorbs thread-scheduling
    // starvation on tiny CI runners while still proving the cutoff.
    CHECK(mock->decode_seqs_calls.size() < 300);
}

TEST_CASE("parallel mode rejects incompatible feature combinations") {
    auto cfg = valid_config();
    cfg.n_parallel = 2;

    SECTION("with a draft model") {
        CHECK_THROWS_AS(ReameEngine(cfg, std::make_unique<MockBackend>(),
                                      std::make_unique<MockBackend>()),
                        EngineError);
    }
    SECTION("with prompt lookup") {
        cfg.use_prompt_lookup = true;
        CHECK_THROWS_AS(ReameEngine(cfg, std::make_unique<MockBackend>()),
                        EngineError);
    }
    SECTION("with the disk cache") {
        cfg.cache_dir = "/tmp/somewhere";
        CHECK_THROWS_AS(ReameEngine(cfg, std::make_unique<MockBackend>()),
                        EngineError);
    }
}

TEST_CASE("single-parallel engine is not parallel capable") {
    auto [engine, mock] = make_engine();
    (void)mock;
    CHECK_FALSE(engine.parallel_capable());
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
               ("reame-engine-cache-" + std::to_string(counter++));
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

        ReameEngine engine(cfg, std::move(backend));
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

        ReameEngine engine(cfg, std::move(backend));
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

    ReameEngine engine(cfg, std::move(backend));
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

TEST_CASE("engine exposes cache stats when the cache is enabled") {
    CacheTempDir dir;
    auto cfg = valid_config();
    cfg.cache_dir = dir.path.string();

    auto backend = std::make_unique<MockBackend>();
    MockBackend* mock = backend.get();
    script_cached(mock, true);

    ReameEngine engine(cfg, std::move(backend));
    CHECK(engine.cache_stats() != nullptr);

    engine.generate("hi", greedy());
    CHECK(engine.cache_stats()->stores >= 1);  // prefix snapshotted
}

TEST_CASE("without a cache dir cache stats are null and prefill unchanged") {
    auto [engine, mock] = make_engine();
    script_foobar(mock);

    CHECK(engine.cache_stats() == nullptr);
    CHECK(engine.generate("hi", greedy()) == "foobar");
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
// Fail-fast model file checks (wants_draft_backend / missing_model_file_error)
// ---------------------------------------------------------------------------

namespace {

// Creates an empty file so std::filesystem::exists() is true.
void touch(const std::filesystem::path& p) {
    std::ofstream(p).flush();
}

}  // namespace

TEST_CASE("wants_draft_backend requires speculation, model mode and a path") {
    ReameEngine::Config c = valid_config();

    // Default: no draft path -> no draft backend.
    CHECK_FALSE(reame::core::wants_draft_backend(c));

    // Speculation on, mode = model, path set -> draft backend wanted.
    c.draft_model_path = "models/draft.gguf";
    c.use_speculative = true;
    c.use_prompt_lookup = false;
    CHECK(reame::core::wants_draft_backend(c));

    // mode = lookup drafts from n-grams: the second model must NOT load
    // even when a (stale) draft_model_path is still set in the config.
    c.use_prompt_lookup = true;
    CHECK_FALSE(reame::core::wants_draft_backend(c));

    // Speculation off ignores the draft path entirely.
    c.use_prompt_lookup = false;
    c.use_speculative = false;
    CHECK_FALSE(reame::core::wants_draft_backend(c));
}

TEST_CASE("missing_model_file_error names the missing main model") {
    CacheTempDir dir;
    ReameEngine::Config c = valid_config();
    c.model_path = (dir.path / "nope.gguf").string();

    const auto err = reame::core::missing_model_file_error(c);
    CHECK(err.find(c.model_path) != std::string::npos);
    // The hint that bit the first HN user: relative paths resolve from the
    // working directory, not from the config file's location.
    CHECK(err.find("working directory") != std::string::npos);
}

TEST_CASE("missing_model_file_error is empty when no draft is wanted") {
    CacheTempDir dir;
    ReameEngine::Config c = valid_config();
    c.model_path = (dir.path / "model.gguf").string();
    touch(c.model_path);

    SECTION("no draft path") {
        CHECK(reame::core::missing_model_file_error(c).empty());
    }
    SECTION("stale draft path but mode = lookup") {
        c.draft_model_path = (dir.path / "missing-draft.gguf").string();
        c.use_prompt_lookup = true;
        CHECK(reame::core::missing_model_file_error(c).empty());
    }
    SECTION("stale draft path but speculation disabled") {
        c.draft_model_path = (dir.path / "missing-draft.gguf").string();
        c.use_speculative = false;
        CHECK(reame::core::missing_model_file_error(c).empty());
    }
}

TEST_CASE("missing_model_file_error flags a wanted-but-missing draft model") {
    CacheTempDir dir;
    ReameEngine::Config c = valid_config();
    c.model_path = (dir.path / "model.gguf").string();
    touch(c.model_path);
    c.draft_model_path = (dir.path / "draft.gguf").string();
    c.use_speculative = true;
    c.use_prompt_lookup = false;

    const auto err = reame::core::missing_model_file_error(c);
    CHECK(err.find(c.draft_model_path) != std::string::npos);
    // Must point at the no-second-model way out.
    CHECK(err.find("mode = lookup") != std::string::npos);

    // Both files present -> no error.
    touch(c.draft_model_path);
    CHECK(reame::core::missing_model_file_error(c).empty());
}

// ---------------------------------------------------------------------------
// Integration (real llama.cpp + TinyLlama). SKIPs when unavailable.
// ---------------------------------------------------------------------------

namespace {

std::string integration_model_path() {
    if (const char* env = std::getenv("REAME_TEST_MODEL")) return env;
    return "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

}  // namespace

TEST_CASE("[integration] prompt cache reproduces the cold output and hits disk",
          "[integration]") {
#ifndef REAME_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    CacheTempDir dir;
    ReameEngine::Config cfg;
    cfg.model_path = path;
    cfg.n_ctx = 256;
    cfg.n_threads = 4;
    cfg.cache_dir = dir.path.string();

    const std::string prompt =
        "The lighthouse keeper counted the ships as they passed: first";

    std::string cold, warm;
    {
        ReameEngine engine(cfg);
        cold = engine.generate(prompt, greedy(8));
    }
    // A snapshot landed on disk.
    CHECK(std::filesystem::directory_iterator(dir.path) !=
          std::filesystem::directory_iterator{});
    {
        ReameEngine engine(cfg);  // fresh process-equivalent, same cache
        warm = engine.generate(prompt, greedy(8));
    }

    CHECK(!cold.empty());
    // Same split-prefill code path cold and warm -> identical numerics.
    CHECK(warm == cold);
#endif
}

TEST_CASE("[integration] parallel engine: 3 users cost far less than 3x one",
          "[integration][parbench]") {
#ifndef REAME_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    const auto run = [&](int n_parallel, int n_requests) {
        ReameEngine::Config cfg;
        cfg.model_path = path;
        cfg.n_ctx = 1024;  // total budget shared across sequences
        cfg.n_threads = 4;
        cfg.n_parallel = n_parallel;
        cfg.use_speculative = false;
        ReameEngine engine(cfg);

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> outs(static_cast<std::size_t>(n_requests));
        if (n_parallel == 1) {
            // Baseline: a non-parallel engine is not thread-safe — the
            // three requests run strictly one after another.
            for (int i = 0; i < n_requests; ++i)
                outs[static_cast<std::size_t>(i)] = engine.generate(
                    "The capital of Italy is", greedy(24));
        } else {
            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errs(
                static_cast<std::size_t>(n_requests));
            for (int i = 0; i < n_requests; ++i)
                threads.emplace_back([&, i] {
                    try {
                        outs[static_cast<std::size_t>(i)] = engine.generate(
                            "The capital of Italy is", greedy(24));
                    } catch (...) {
                        errs[static_cast<std::size_t>(i)] =
                            std::current_exception();
                    }
                });
            for (auto& t : threads) t.join();
            for (const auto& e : errs)
                if (e) std::rethrow_exception(e);
        }
        for (const auto& o : outs) REQUIRE(!o.empty());
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    };

    const double serial = run(/*n_parallel=*/1, /*n_requests=*/3);
    const double parallel = run(/*n_parallel=*/3, /*n_requests=*/3);

    WARN("3 requests serial: " << serial << "s, interleaved: " << parallel
                               << "s, ratio: " << serial / parallel << "x");
    // Interleaving must beat full serialization by a clear margin.
    CHECK(parallel < serial * 0.75);
#endif
}

TEST_CASE("[integration] engine generates deterministic greedy text",
          "[integration]") {
#ifndef REAME_HAS_LLAMA
    SKIP("built without llama.cpp (submodule not initialized)");
#else
    const auto path = integration_model_path();
    if (!file_exists(path))
        SKIP("model file not found: " + path +
             " (run scripts/download_models.sh)");

    ReameEngine::Config cfg;
    cfg.model_path = path;
    cfg.n_ctx = 256;
    cfg.n_threads = 4;
    ReameEngine engine(cfg);

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
