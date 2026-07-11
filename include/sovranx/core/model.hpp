#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sovranx/core/llama_backend.hpp"

namespace sovranx {

class Config;

class ModelError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ModelParams {
    std::string path;
    std::int32_t context_length = 4096;
    std::int32_t threads = 0;  // 0 is invalid; from_config fills a default
    bool use_mmap = true;
    bool use_mlock = false;
    // KV-cache element type: f16 (default) | q8_0 | q4_0. Quantizing the
    // cache halves/quarters its RAM at negligible quality cost — decisive
    // on low-RAM hosts.
    std::string kv_cache_type = "f16";
    // Max concurrent sequences the context supports (decode_seqs);
    // context_length is the TOTAL KV budget shared across them.
    std::int32_t n_seq_max = 1;
    // Physical batch size for prompt processing. 0 = llama default (512).
    // Larger values process long prompts in fewer, bigger matmuls — better
    // SIMD utilization on CPU, faster cold prefill.
    std::int32_t n_ubatch = 0;

    // Reads: model.path (required), model.context_length, model.threads
    // (default: hardware concurrency), memory.use_mmap, memory.use_mlock,
    // memory.kv_cache_type. Throws ModelError if model.path is missing or
    // a value is invalid.
    static ModelParams from_config(const Config& cfg);
};

// High-level wrapper around a GGUF model. Owns validation and error
// translation; the llama.cpp specifics live behind LlamaBackend (pimpl,
// so this header never includes llama.h).
class LlamaModel {
public:
    // Production: loads the model via the real llama.cpp backend.
    explicit LlamaModel(const ModelParams& params);

    // Test seam: inject a backend (mock). Still validates `params`.
    LlamaModel(const ModelParams& params, std::unique_ptr<LlamaBackend> backend);

    ~LlamaModel();
    LlamaModel(LlamaModel&&) noexcept;
    LlamaModel& operator=(LlamaModel&&) noexcept;
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    std::vector<TokenId> tokenize(const std::string& text,
                                  bool add_special = true) const;

    // Throws ModelError if any token is outside [0, vocab_size).
    std::string detokenize(const std::vector<TokenId>& tokens) const;

    // Forward pass over a fresh sequence; returns logits of the last token.
    // Throws ModelError on empty input, input longer than the context, or
    // a backend logits/vocab size mismatch.
    std::vector<float> forward(const std::vector<TokenId>& tokens);

    std::int32_t vocab_size() const;
    std::uint32_t context_length() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sovranx
