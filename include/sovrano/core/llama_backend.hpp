#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sovrano {

using TokenId = std::int32_t;

struct ModelParams;

// Seam between LlamaModel and llama.cpp. The real implementation
// (llama_backend_real.cpp) talks to the llama C API; tests inject a mock.
// Implementations own the underlying model/context and release them on
// destruction.
class LlamaBackend {
public:
    virtual ~LlamaBackend() = default;

    virtual std::vector<TokenId> tokenize(const std::string& text,
                                          bool add_special) = 0;
    virtual std::string detokenize(const std::vector<TokenId>& tokens) = 0;

    // Renders a single token as its text piece, preserving leading spaces.
    // This is what streaming must use: llama_detokenize on a 1-token slice
    // strips the leading space marker.
    virtual std::string token_piece(TokenId token) = 0;

    // Runs a full forward pass over `tokens` (fresh sequence, KV cache
    // cleared) and returns the logits of the last token (size == vocab_size).
    virtual std::vector<float> decode(const std::vector<TokenId>& tokens) = 0;

    // Continues the current sequence (KV cache kept) with `tokens` and
    // returns the logits of the last token. Used by the generation loop to
    // avoid re-prefilling on every step.
    virtual std::vector<float> decode_append(const std::vector<TokenId>& tokens) = 0;

    virtual std::int32_t vocab_size() const = 0;
    virtual std::uint32_t context_length() const = 0;
    virtual TokenId eos_token() const = 0;
};

// Factory for the real llama.cpp backend. Throws ModelError if the model
// cannot be loaded, or if the binary was built without llama.cpp
// (submodule not initialized).
std::unique_ptr<LlamaBackend> make_llama_backend(const ModelParams& params);

}  // namespace sovrano
