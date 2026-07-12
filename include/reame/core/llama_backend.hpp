#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace reame {

using TokenId = std::int32_t;

struct ModelParams;

// One sequence's contribution to a multi-sequence batch.
struct SeqSlice {
    std::int32_t seq_id = 0;
    std::vector<TokenId> tokens;  // tokens to decode for this sequence
    std::uint32_t pos_start = 0;  // position of tokens[0] within the seq
};

// One turn of an OpenAI-style conversation ("system"/"user"/"assistant").
struct ChatMessage {
    std::string role;
    std::string content;
};

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

    // Continues the current sequence with `tokens` and returns the logits
    // of EVERY position (result[i] predicts the token after tokens[i]).
    // This is the single batched forward pass speculative verification
    // needs.
    virtual std::vector<std::vector<float>> decode_batch(
        const std::vector<TokenId>& tokens) = 0;

    // Rolls the sequence back to its first `n_tokens` positions (drops the
    // KV entries of everything after). Used to discard rejected draft
    // tokens.
    virtual void truncate_to(std::uint32_t n_tokens) = 0;

    // ---- Multi-sequence (interleaved multi-user) support ----------------
    // Decodes several sequences' tokens in ONE forward pass: the model
    // weights are read once for the whole batch, which on memory-bound
    // CPUs makes N concurrent generations cost far less than N sequential
    // ones. Returns the last-token logits of each slice, in input order.
    // Requires the context to be created with n_seq_max > 1
    // (ModelParams::n_seq_max).
    virtual std::vector<std::vector<float>> decode_seqs(
        const std::vector<SeqSlice>& slices) = 0;

    // Formats a single user message with the model's own chat template
    // (GGUF metadata tokenizer.chat_template), assistant turn opened.
    // Returns the message unchanged when the model ships no template —
    // raw completion is then the correct behavior.
    virtual std::string format_chat(const std::string& user_message) = 0;

    // Formats a whole conversation with the model's own chat template,
    // assistant turn opened. Template-less models fall back to plain
    // role-prefixed turns ("role: content\n...assistant:").
    virtual std::string format_chat(
        const std::vector<ChatMessage>& messages) = 0;

    // Drops one sequence's KV cache entirely (request finished).
    virtual void clear_seq(std::int32_t seq_id) = 0;

    // Copies the first n_tokens KV positions of `src` into `dst` (shared
    // prefill: identical prompts pay one prefill, the clones are nearly
    // free). `dst` must be empty.
    virtual void copy_seq(std::int32_t src, std::int32_t dst,
                          std::uint32_t n_tokens) = 0;

    // Clears the sequence entirely (KV cache + position counter).
    virtual void reset() = 0;

    // Full model state snapshot (RNG, logits, KV cache) as an opaque blob.
    virtual std::vector<char> state_data() = 0;

    // Restores a snapshot produced by state_data(). The blob does not
    // carry the wrapper's position counter, so callers must say how many
    // sequence positions it represents.
    virtual void set_state(const std::vector<char>& data,
                           std::uint32_t n_past) = 0;

    // Number of positions currently stored in the sequence.
    virtual std::uint32_t n_past() const = 0;

    virtual std::int32_t vocab_size() const = 0;
    virtual std::uint32_t context_length() const = 0;
    virtual TokenId eos_token() const = 0;

    // True when `token` ends generation. Broader than `token ==
    // eos_token()`: chat models mark their end-of-turn control token
    // (e.g. ChatML's <|im_end|>) as EOG without it being the vocab eos.
    virtual bool is_eog(TokenId token) const = 0;
};

// Factory for the real llama.cpp backend. Throws ModelError if the model
// cannot be loaded, or if the binary was built without llama.cpp
// (submodule not initialized).
std::unique_ptr<LlamaBackend> make_llama_backend(const ModelParams& params);

}  // namespace reame
