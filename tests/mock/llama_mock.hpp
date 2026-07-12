#pragma once

// Mock LlamaBackend for isolated LlamaModel tests: canned return values,
// recorded calls, no llama.cpp involved. The interface shape mirrors the
// real backend (llama_backend_real.cpp), which was written against the
// actual llama.h of the pinned submodule.

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "reame/core/llama_backend.hpp"

namespace reame::test {

class MockBackend : public LlamaBackend {
public:
    // Canned responses (set from the test before use).
    std::vector<TokenId> tokenize_result;
    std::string detokenize_result;
    std::vector<float> decode_result;
    // Scripted logits: decode/decode_append pop from here first, falling
    // back to decode_result when empty. Lets a test drive a whole
    // generation step by step.
    std::deque<std::vector<float>> decode_queue;
    // Scripted per-position logits for decode_batch. When empty, the batch
    // result is synthesized from decode_queue/decode_result (one entry per
    // input token).
    std::deque<std::vector<std::vector<float>>> decode_batch_queue;
    // When set, every decode/decode_append/decode_batch throws (fallback
    // tests).
    bool fail_decodes = false;
    // Pacing for decode_seqs (ms per call): emulates real per-step model
    // latency where cross-thread signals land between steps.
    int decode_seqs_delay_ms = 0;
    // Per-token piece for token_piece (generation streaming); falls back
    // to detokenize_result when the token is not mapped.
    std::map<TokenId, std::string> piece_map;
    std::int32_t vocab_size_value = 32000;
    std::uint32_t context_length_value = 2048;
    TokenId eos_token_value = 2;
    // Extra end-of-generation tokens beyond eos_token_value (ChatML models
    // mark <|im_end|> as EOG alongside the vocab eos).
    std::set<TokenId> eog_tokens;
    // False emulates a recurrent/hybrid model (Qwen3.5, Mamba): the state
    // cannot be rolled back, so truncate_to would fail at runtime.
    bool supports_rollback_value = true;

    // Recorded calls.
    std::vector<std::pair<std::string, bool>> tokenize_calls;
    std::vector<std::vector<TokenId>> detokenize_calls;
    std::vector<TokenId> token_piece_calls;
    std::vector<std::vector<TokenId>> decode_calls;
    std::vector<std::vector<TokenId>> decode_append_calls;
    std::vector<std::vector<TokenId>> decode_batch_calls;
    std::vector<std::vector<SeqSlice>> decode_seqs_calls;
    std::vector<std::int32_t> clear_seq_calls;
    struct CopySeqCall {
        std::int32_t src;
        std::int32_t dst;
        std::uint32_t n_tokens;
    };
    std::vector<CopySeqCall> copy_seq_calls;
    std::vector<std::string> format_chat_calls;
    std::vector<std::vector<ChatMessage>> format_chat_messages_calls;
    // When true the mock emulates a template-less model: format_chat
    // passes the message through untouched (multi-message: plain
    // role-prefixed fallback, mirroring the real backend).
    bool chat_template_empty = false;
    // Per-sequence scripted logits for decode_seqs (falls back to
    // decode_result when a seq's queue is empty). When a seq slot is
    // reused by a new request (clear_seq), its queue is re-seeded from
    // seq_template — modelling that each sequence is independent, exactly
    // as the real backend behaves.
    std::map<std::int32_t, std::deque<std::vector<float>>> seq_decode_queues;
    std::deque<std::vector<float>> seq_template;
    std::vector<std::uint32_t> truncate_calls;
    int reset_calls = 0;
    // State snapshot support (cache tests).
    std::vector<char> state_data_result = {'s', 't', 'a', 't', 'e'};
    int state_data_calls = 0;
    std::vector<std::pair<std::vector<char>, std::uint32_t>> set_state_calls;

    std::vector<TokenId> tokenize(const std::string& text,
                                  bool add_special) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        tokenize_calls.emplace_back(text, add_special);
        return tokenize_result;
    }

    std::string detokenize(const std::vector<TokenId>& tokens) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        detokenize_calls.push_back(tokens);
        return detokenize_result;
    }

    std::string token_piece(TokenId token) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        token_piece_calls.push_back(token);
        const auto it = piece_map.find(token);
        return it != piece_map.end() ? it->second : detokenize_result;
    }

    std::vector<float> decode(const std::vector<TokenId>& tokens) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        maybe_fail();
        decode_calls.push_back(tokens);
        n_past_value = static_cast<std::uint32_t>(tokens.size());
        return next_logits();
    }

    std::vector<float> decode_append(const std::vector<TokenId>& tokens) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        maybe_fail();
        decode_append_calls.push_back(tokens);
        n_past_value += static_cast<std::uint32_t>(tokens.size());
        return next_logits();
    }

    std::vector<std::vector<float>> decode_batch(
        const std::vector<TokenId>& tokens) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        maybe_fail();
        decode_batch_calls.push_back(tokens);
        n_past_value += static_cast<std::uint32_t>(tokens.size());
        if (!decode_batch_queue.empty()) {
            auto v = std::move(decode_batch_queue.front());
            decode_batch_queue.pop_front();
            return v;
        }
        std::vector<std::vector<float>> out;
        for (std::size_t i = 0; i < tokens.size(); ++i)
            out.push_back(next_logits());
        return out;
    }

    std::vector<std::vector<float>> decode_seqs(
        const std::vector<SeqSlice>& slices) override {
        // Optional per-step pacing OUTSIDE the lock: a real model spends
        // ~100ms per interleaved step, so cross-thread signals (e.g. the
        // conclave's early-consensus stop) land within a step or two. At
        // mock speed (ns/step) those signals would race the whole script.
        if (decode_seqs_delay_ms > 0)
            std::this_thread::sleep_for(
                std::chrono::milliseconds(decode_seqs_delay_ms));
        std::lock_guard<std::mutex> lock(mock_mutex_);
        maybe_fail();
        decode_seqs_calls.push_back(slices);
        std::vector<std::vector<float>> out;
        for (const auto& s : slices) {
            auto& q = seq_decode_queues[s.seq_id];
            if (!q.empty()) {
                out.push_back(std::move(q.front()));
                q.pop_front();
            } else {
                out.push_back(decode_result);
            }
        }
        return out;
    }

    void clear_seq(std::int32_t seq_id) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        clear_seq_calls.push_back(seq_id);
        if (!seq_template.empty()) seq_decode_queues[seq_id] = seq_template;
    }

    std::string format_chat(const std::string& user_message) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        format_chat_calls.push_back(user_message);
        if (chat_template_empty) return user_message;
        return "<U>" + user_message + "</U><A>";
    }

    std::string format_chat(
        const std::vector<ChatMessage>& messages) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        format_chat_messages_calls.push_back(messages);
        std::string out;
        if (chat_template_empty) {
            // Mirrors the real backend's template-less fallback.
            for (const auto& m : messages)
                out += m.role + ": " + m.content + "\n";
            return out + "assistant:";
        }
        for (const auto& m : messages) out += "<M:" + m.role + ">" + m.content;
        return out + "<A>";
    }

    void copy_seq(std::int32_t src, std::int32_t dst,
                  std::uint32_t n_tokens) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        copy_seq_calls.push_back({src, dst, n_tokens});
    }

    void truncate_to(std::uint32_t n_tokens) override {
        std::lock_guard<std::mutex> lock(mock_mutex_);
        truncate_calls.push_back(n_tokens);
        n_past_value = n_tokens;
    }

    void reset() override {
        ++reset_calls;
        n_past_value = 0;
    }

    std::vector<char> state_data() override {
        ++state_data_calls;
        return state_data_result;
    }

    void set_state(const std::vector<char>& data,
                   std::uint32_t n_past) override {
        set_state_calls.emplace_back(data, n_past);
        n_past_value = n_past;
    }

    std::uint32_t n_past() const override { return n_past_value; }

    std::int32_t vocab_size() const override { return vocab_size_value; }
    std::uint32_t context_length() const override { return context_length_value; }
    TokenId eos_token() const override { return eos_token_value; }
    bool is_eog(TokenId token) const override {
        return token == eos_token_value || eog_tokens.count(token) > 0;
    }
    bool supports_rollback() const override {
        return supports_rollback_value;
    }

private:
    void maybe_fail() {
        if (fail_decodes) throw std::runtime_error("mock decode failure");
    }

    std::vector<float> next_logits() {
        if (decode_queue.empty()) return decode_result;
        auto v = std::move(decode_queue.front());
        decode_queue.pop_front();
        return v;
    }

    std::uint32_t n_past_value = 0;
    // Concurrent engine tests (parallel mode) hit the mock from several
    // threads; the real tokenizer is thread-safe, so the mock must be too.
    mutable std::mutex mock_mutex_;
};

}  // namespace reame::test
