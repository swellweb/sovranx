#pragma once

// Mock LlamaBackend for isolated LlamaModel tests: canned return values,
// recorded calls, no llama.cpp involved. The interface shape mirrors the
// real backend (llama_backend_real.cpp), which was written against the
// actual llama.h of the pinned submodule.

#include <cstdint>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sovrano/core/llama_backend.hpp"

namespace sovrano::test {

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
    // Per-token piece for token_piece (generation streaming); falls back
    // to detokenize_result when the token is not mapped.
    std::map<TokenId, std::string> piece_map;
    std::int32_t vocab_size_value = 32000;
    std::uint32_t context_length_value = 2048;
    TokenId eos_token_value = 2;

    // Recorded calls.
    std::vector<std::pair<std::string, bool>> tokenize_calls;
    std::vector<std::vector<TokenId>> detokenize_calls;
    std::vector<TokenId> token_piece_calls;
    std::vector<std::vector<TokenId>> decode_calls;
    std::vector<std::vector<TokenId>> decode_append_calls;
    std::vector<std::vector<TokenId>> decode_batch_calls;
    std::vector<std::uint32_t> truncate_calls;
    int reset_calls = 0;
    // State snapshot support (cache tests).
    std::vector<char> state_data_result = {'s', 't', 'a', 't', 'e'};
    int state_data_calls = 0;
    std::vector<std::pair<std::vector<char>, std::uint32_t>> set_state_calls;

    std::vector<TokenId> tokenize(const std::string& text,
                                  bool add_special) override {
        tokenize_calls.emplace_back(text, add_special);
        return tokenize_result;
    }

    std::string detokenize(const std::vector<TokenId>& tokens) override {
        detokenize_calls.push_back(tokens);
        return detokenize_result;
    }

    std::string token_piece(TokenId token) override {
        token_piece_calls.push_back(token);
        const auto it = piece_map.find(token);
        return it != piece_map.end() ? it->second : detokenize_result;
    }

    std::vector<float> decode(const std::vector<TokenId>& tokens) override {
        maybe_fail();
        decode_calls.push_back(tokens);
        n_past_value = static_cast<std::uint32_t>(tokens.size());
        return next_logits();
    }

    std::vector<float> decode_append(const std::vector<TokenId>& tokens) override {
        maybe_fail();
        decode_append_calls.push_back(tokens);
        n_past_value += static_cast<std::uint32_t>(tokens.size());
        return next_logits();
    }

    std::vector<std::vector<float>> decode_batch(
        const std::vector<TokenId>& tokens) override {
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

    void truncate_to(std::uint32_t n_tokens) override {
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
};

}  // namespace sovrano::test
