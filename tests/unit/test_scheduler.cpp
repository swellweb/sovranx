// Isolated tests for the interleaved scheduler. step() is driven directly
// (no threads); the mock's per-sequence queues script each request's
// token trajectory, so every expectation is hand-derived.

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "../mock/llama_mock.hpp"
#include "reame/core/scheduler.hpp"

using reame::SeqSlice;
using reame::TokenId;
using reame::test::MockBackend;
using reame::core::EngineError;
using reame::core::GenerationConfig;
using reame::core::Scheduler;

namespace {

GenerationConfig greedy(int max_tokens = 16) {
    GenerationConfig g;
    g.temperature = 0.0f;
    g.repeat_penalty = 1.0f;
    g.max_tokens = max_tokens;
    return g;
}

std::vector<float> peak(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 10.0f;
    return v;
}

constexpr std::size_t kVocab = 6;
constexpr TokenId kEos = 5;

void setup(MockBackend& m) {
    m.vocab_size_value = kVocab;
    m.eos_token_value = kEos;
    m.context_length_value = 256;
}

struct Collector {
    std::vector<TokenId> tokens;
    Scheduler::TokenCallback cb() {
        return [this](TokenId t) {
            tokens.push_back(t);
            return true;
        };
    }
};

}  // namespace

TEST_CASE("a sequence stops at any end-of-generation token, not only eos") {
    MockBackend backend;
    setup(backend);
    // Token 4 is a second end-of-turn marker (ChatML-style <|im_end|>):
    // the sequence must stop there, before the scripted token 2.
    backend.eog_tokens = {4};
    backend.seq_decode_queues[0] = {peak(kVocab, 1), peak(kVocab, 4),
                                    peak(kVocab, 2)};

    Scheduler sched(backend, {/*n_parallel=*/1});
    Collector a;
    const auto ra = sched.submit({10}, greedy(), a.cb());
    sched.run_until_idle();

    CHECK(a.tokens == std::vector<TokenId>{1});
    CHECK(sched.finished(ra));
    CHECK(sched.error(ra) == nullptr);
}

TEST_CASE("two requests interleave in shared batches and both complete") {
    MockBackend backend;
    setup(backend);
    // seq 0 (request A): tokens 1, 2, then EOS.  seq 1 (B): token 3, EOS.
    backend.seq_decode_queues[0] = {peak(kVocab, 1), peak(kVocab, 2),
                                    peak(kVocab, kEos)};
    backend.seq_decode_queues[1] = {peak(kVocab, 3), peak(kVocab, kEos)};

    Scheduler sched(backend, {/*n_parallel=*/2});
    Collector a, b;
    const auto ra = sched.submit({10, 11}, greedy(), a.cb());
    const auto rb = sched.submit({20}, greedy(), b.cb());

    sched.run_until_idle();

    CHECK(a.tokens == std::vector<TokenId>{1, 2});
    CHECK(b.tokens == std::vector<TokenId>{3});
    CHECK(sched.finished(ra));
    CHECK(sched.finished(rb));
    CHECK(sched.error(ra) == nullptr);

    // First batch: both prefills together, at position 0 of their seqs.
    REQUIRE_FALSE(backend.decode_seqs_calls.empty());
    const auto& first = backend.decode_seqs_calls[0];
    REQUIRE(first.size() == 2);
    CHECK(first[0].tokens == std::vector<TokenId>{10, 11});
    CHECK(first[0].pos_start == 0);
    CHECK(first[1].tokens == std::vector<TokenId>{20});
    CHECK(first[1].seq_id != first[0].seq_id);

    // Second batch: one generated token per request, positions advanced.
    const auto& second = backend.decode_seqs_calls[1];
    REQUIRE(second.size() == 2);
    CHECK(second[0].tokens == std::vector<TokenId>{1});
    CHECK(second[0].pos_start == 2);
    CHECK(second[1].tokens == std::vector<TokenId>{3});
    CHECK(second[1].pos_start == 1);

    // Both sequences were released.
    CHECK(backend.clear_seq_calls.size() == 2);
}

TEST_CASE("requests beyond n_parallel wait and run after a slot frees") {
    MockBackend backend;
    setup(backend);
    backend.seq_decode_queues[0] = {peak(kVocab, 1), peak(kVocab, kEos),
                                    // reused by the queued request:
                                    peak(kVocab, 2), peak(kVocab, kEos)};

    Scheduler sched(backend, {/*n_parallel=*/1});
    Collector a, b;
    sched.submit({10}, greedy(), a.cb());
    sched.submit({20}, greedy(), b.cb());

    CHECK(sched.step());          // prefill A only: one slice
    CHECK(backend.decode_seqs_calls[0].size() == 1);

    sched.run_until_idle();

    CHECK(a.tokens == std::vector<TokenId>{1});
    CHECK(b.tokens == std::vector<TokenId>{2});
}

TEST_CASE("a callback returning false ends that request but not the others") {
    MockBackend backend;
    setup(backend);
    backend.seq_decode_queues[0] = {peak(kVocab, 1), peak(kVocab, 2),
                                    peak(kVocab, 3), peak(kVocab, kEos)};
    backend.seq_decode_queues[1] = {peak(kVocab, 4), peak(kVocab, kEos)};

    Scheduler sched(backend, {2});
    Collector b;
    std::vector<TokenId> a_tokens;
    const auto ra = sched.submit({10}, greedy(), [&](TokenId t) {
        a_tokens.push_back(t);
        return false;  // stop A immediately
    });
    const auto rb = sched.submit({20}, greedy(), b.cb());

    sched.run_until_idle();

    CHECK(a_tokens == std::vector<TokenId>{1});
    CHECK(b.tokens == std::vector<TokenId>{4});
    CHECK(sched.finished(ra));
    CHECK(sched.finished(rb));
}

TEST_CASE("max_tokens is enforced per request") {
    MockBackend backend;
    setup(backend);
    backend.decode_result = peak(kVocab, 2);  // token 2 forever

    Scheduler sched(backend, {1});
    Collector a;
    sched.submit({10}, greedy(/*max_tokens=*/3), a.cb());
    sched.run_until_idle();

    CHECK(a.tokens == std::vector<TokenId>{2, 2, 2});
}

TEST_CASE("a prompt that can never fit is rejected at submit") {
    MockBackend backend;
    setup(backend);
    backend.context_length_value = 4;

    Scheduler sched(backend, {2});
    CHECK_THROWS_AS(
        sched.submit({1, 2, 3, 4, 5}, greedy(), [](TokenId) { return true; }),
        EngineError);
}

TEST_CASE("a backend failure fails in-flight requests, scheduler survives") {
    MockBackend backend;
    setup(backend);

    Scheduler sched(backend, {2});
    Collector a;
    const auto ra = sched.submit({10}, greedy(), a.cb());

    backend.fail_decodes = true;
    sched.run_until_idle();

    CHECK(sched.finished(ra));
    CHECK(sched.error(ra) != nullptr);

    // Recovered: a new request completes normally.
    backend.fail_decodes = false;
    backend.seq_decode_queues.clear();
    backend.seq_decode_queues[0] = {peak(kVocab, 1), peak(kVocab, kEos)};
    Collector c;
    const auto rc = sched.submit({30}, greedy(), c.cb());
    sched.run_until_idle();

    CHECK(sched.finished(rc));
    CHECK(sched.error(rc) == nullptr);
    CHECK(c.tokens == std::vector<TokenId>{1});
}

TEST_CASE("idle scheduler reports no work") {
    MockBackend backend;
    setup(backend);
    Scheduler sched(backend, {2});
    CHECK_FALSE(sched.step());
    CHECK(sched.active_count() == 0);
}

TEST_CASE("identical prompt clones the donor's KV instead of prefilling") {
    MockBackend backend;
    setup(backend);
    // A (seq 0): prompt {1,2,3} -> token 4, then EOS.
    // B (seq 1): same prompt -> token 1, then EOS.
    backend.seq_decode_queues[0] = {peak(kVocab, 4), peak(kVocab, kEos)};
    backend.seq_decode_queues[1] = {peak(kVocab, 1), peak(kVocab, kEos)};

    Scheduler sched(backend, {2});
    Collector ca, cb;
    const auto a = sched.submit({1, 2, 3}, greedy(), ca.cb());

    // Step 1: A prefills alone (3-token slice at pos 0).
    REQUIRE(sched.step());
    REQUIRE(backend.decode_seqs_calls.size() == 1);
    REQUIRE(backend.decode_seqs_calls[0].size() == 1);
    CHECK(backend.decode_seqs_calls[0][0].tokens.size() == 3);

    // B arrives with the SAME prompt: its KV must be cloned from A —
    // copy positions [0, 2) and decode only the LAST prompt token (which
    // produces the logits a prefill would have produced).
    const auto b = sched.submit({1, 2, 3}, greedy(), cb.cb());
    sched.run_until_idle();

    REQUIRE(backend.copy_seq_calls.size() == 1);
    CHECK(backend.copy_seq_calls[0].src == 0);
    CHECK(backend.copy_seq_calls[0].dst == 1);
    CHECK(backend.copy_seq_calls[0].n_tokens == 2);

    // No full prefill for B: every slice on seq 1 is a single token, and
    // its first slice sits at position 2 carrying prompt token 3.
    bool saw_b = false;
    for (const auto& call : backend.decode_seqs_calls)
        for (const auto& s : call)
            if (s.seq_id == 1) {
                CHECK(s.tokens.size() == 1);
                if (!saw_b) {
                    CHECK(s.pos_start == 2);
                    CHECK(s.tokens[0] == 3);
                    saw_b = true;
                }
            }
    CHECK(saw_b);

    CHECK(sched.finished(a));
    CHECK(sched.finished(b));
    CHECK(ca.tokens == std::vector<TokenId>{4});
    CHECK(cb.tokens == std::vector<TokenId>{1});
}

TEST_CASE("different prompt does not clone: full prefill") {
    MockBackend backend;
    setup(backend);
    backend.seq_decode_queues[0] = {peak(kVocab, 4), peak(kVocab, kEos)};
    backend.seq_decode_queues[1] = {peak(kVocab, 1), peak(kVocab, kEos)};

    Scheduler sched(backend, {2});
    Collector ca, cb;
    sched.submit({1, 2, 3}, greedy(), ca.cb());
    REQUIRE(sched.step());
    sched.submit({1, 2, 4}, greedy(), cb.cb());  // differs in the last token
    sched.run_until_idle();

    CHECK(backend.copy_seq_calls.empty());
    // B's first slice is its full 3-token prefill.
    bool saw_b_prefill = false;
    for (const auto& call : backend.decode_seqs_calls)
        for (const auto& s : call)
            if (s.seq_id == 1 && s.tokens.size() == 3 && s.pos_start == 0)
                saw_b_prefill = true;
    CHECK(saw_b_prefill);
}

TEST_CASE("same prompts submitted together: first prefills, second clones") {
    MockBackend backend;
    setup(backend);
    backend.seq_decode_queues[0] = {peak(kVocab, 4), peak(kVocab, kEos)};
    backend.seq_decode_queues[1] = {peak(kVocab, 1), peak(kVocab, kEos)};

    Scheduler sched(backend, {2});
    Collector ca, cb;
    const auto a = sched.submit({1, 2, 3}, greedy(), ca.cb());
    const auto b = sched.submit({1, 2, 3}, greedy(), cb.cb());
    sched.run_until_idle();

    // The second waits one step for the donor's prefill, then clones.
    REQUIRE(backend.copy_seq_calls.size() == 1);
    CHECK(backend.copy_seq_calls[0].n_tokens == 2);
    CHECK(sched.finished(a));
    CHECK(sched.finished(b));
    CHECK(ca.tokens == std::vector<TokenId>{4});
    CHECK(cb.tokens == std::vector<TokenId>{1});
}
