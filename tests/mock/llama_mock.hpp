#pragma once

// Mock LlamaBackend for isolated LlamaModel tests: canned return values,
// recorded calls, no llama.cpp involved. The interface shape mirrors the
// real backend (llama_backend_real.cpp), which was written against the
// actual llama.h of the pinned submodule.

#include <cstdint>
#include <deque>
#include <map>
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
        decode_calls.push_back(tokens);
        return next_logits();
    }

    std::vector<float> decode_append(const std::vector<TokenId>& tokens) override {
        decode_append_calls.push_back(tokens);
        return next_logits();
    }

    std::int32_t vocab_size() const override { return vocab_size_value; }
    std::uint32_t context_length() const override { return context_length_value; }
    TokenId eos_token() const override { return eos_token_value; }

private:
    std::vector<float> next_logits() {
        if (decode_queue.empty()) return decode_result;
        auto v = std::move(decode_queue.front());
        decode_queue.pop_front();
        return v;
    }
};

}  // namespace sovrano::test
