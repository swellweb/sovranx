// Real llama.cpp backend. This is the ONLY translation unit that includes
// llama.h; it is compiled only when the submodule is present
// (REAME_HAS_LLAMA). CPU-only by design: n_gpu_layers = 0.

#include <llama.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "reame/core/llama_backend.hpp"
#include "reame/core/model.hpp"

namespace reame {

namespace {

void ensure_backend_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        llama_backend_init();
        // Keep llama.cpp's own logging quiet; Reame has its own logger.
        llama_log_set(
            [](ggml_log_level level, const char* text, void*) {
                if (level >= GGML_LOG_LEVEL_ERROR || std::getenv("REAME_LLAMA_VERBOSE")) std::fputs(text, stderr);
            },
            nullptr);
    });
}

class RealLlamaBackend final : public LlamaBackend {
public:
    explicit RealLlamaBackend(const ModelParams& params) {
        ensure_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;  // CPU-only engine
        mparams.use_mmap = params.use_mmap;
        mparams.use_mlock = params.use_mlock;

        model_ = llama_model_load_from_file(params.path.c_str(), mparams);
        if (model_ == nullptr)
            throw ModelError("failed to load GGUF model: " + params.path);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(params.context_length);
        cparams.n_batch = static_cast<uint32_t>(params.context_length);
        if (params.n_ubatch > 0)
            cparams.n_ubatch = static_cast<uint32_t>(params.n_ubatch);
        cparams.n_threads = params.threads;
        cparams.n_threads_batch = params.threads;
        if (params.kv_cache_type == "q8_0") {
            cparams.type_k = GGML_TYPE_Q8_0;
            cparams.type_v = GGML_TYPE_Q8_0;
        } else if (params.kv_cache_type == "q4_0") {
            cparams.type_k = GGML_TYPE_Q4_0;
            cparams.type_v = GGML_TYPE_Q4_0;
        }  // f16 = llama default
        // A quantized V cache requires flash attention; AUTO enables it
        // where the build supports it and errors clearly otherwise.
        if (params.kv_cache_type != "f16")
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        if (params.n_seq_max > 1) {
            cparams.n_seq_max = static_cast<uint32_t>(params.n_seq_max);
            // One shared KV buffer for all sequences (context_length is a
            // TOTAL budget): required for freely mixed multi-sequence
            // batches — the per-stream default asserts on them.
            cparams.kv_unified = true;
        }

        ctx_ = llama_init_from_model(model_, cparams);
        if (ctx_ == nullptr) {
            llama_model_free(model_);
            throw ModelError("failed to create llama context for: " + params.path);
        }

        vocab_ = llama_model_get_vocab(model_);
    }

    ~RealLlamaBackend() override {
        if (ctx_ != nullptr) llama_free(ctx_);
        if (model_ != nullptr) llama_model_free(model_);
    }

    RealLlamaBackend(const RealLlamaBackend&) = delete;
    RealLlamaBackend& operator=(const RealLlamaBackend&) = delete;

    std::vector<TokenId> tokenize(const std::string& text,
                                  bool add_special) override {
        const auto text_len = static_cast<int32_t>(text.size());

        // Two-pass: first call reports the required size as a negative count.
        // parse_special=true: chat templates carry control tokens as text
        // ("<|im_start|>", "</s>") and they must map to their single control
        // id — spelled out as plain text the model mimics the tags and never
        // emits a real EOS.
        int32_t n = llama_tokenize(vocab_, text.c_str(), text_len,
                                   nullptr, 0, add_special,
                                   /*parse_special=*/true);
        if (n == 0) return {};
        if (n == INT32_MIN)
            throw ModelError("tokenization overflow for input of " +
                             std::to_string(text.size()) + " bytes");
        if (n < 0) n = -n;

        std::vector<TokenId> tokens(static_cast<std::size_t>(n));
        const int32_t written =
            llama_tokenize(vocab_, text.c_str(), text_len, tokens.data(), n,
                           add_special, /*parse_special=*/true);
        if (written < 0)
            throw ModelError("tokenization failed");
        tokens.resize(static_cast<std::size_t>(written));
        return tokens;
    }

    std::string detokenize(const std::vector<TokenId>& tokens) override {
        const auto n_tokens = static_cast<int32_t>(tokens.size());

        int32_t n = llama_detokenize(vocab_, tokens.data(), n_tokens,
                                     nullptr, 0, /*remove_special=*/false,
                                     /*unparse_special=*/false);
        if (n == 0) return {};
        if (n < 0) n = -n;

        std::string text(static_cast<std::size_t>(n), '\0');
        const int32_t written =
            llama_detokenize(vocab_, tokens.data(), n_tokens, text.data(), n,
                             /*remove_special=*/false,
                             /*unparse_special=*/false);
        if (written < 0)
            throw ModelError("detokenization failed");
        text.resize(static_cast<std::size_t>(written));
        return text;
    }

    std::string token_piece(TokenId token) override {
        char buf[128];
        int32_t n = llama_token_to_piece(vocab_, token, buf, sizeof(buf),
                                         /*lstrip=*/0, /*special=*/false);
        if (n >= 0) return std::string(buf, static_cast<std::size_t>(n));

        // Piece longer than the stack buffer: retry with the exact size.
        std::string piece(static_cast<std::size_t>(-n), '\0');
        n = llama_token_to_piece(vocab_, token, piece.data(),
                                 static_cast<int32_t>(piece.size()),
                                 /*lstrip=*/0, /*special=*/false);
        if (n < 0)
            throw ModelError("token_to_piece failed for token " +
                             std::to_string(token));
        piece.resize(static_cast<std::size_t>(n));
        return piece;
    }

    std::vector<float> decode(const std::vector<TokenId>& tokens) override {
        reset();
        return decode_impl(tokens, /*all_logits=*/false).back();
    }

    std::vector<float> decode_append(const std::vector<TokenId>& tokens) override {
        return decode_impl(tokens, /*all_logits=*/false).back();
    }

    std::vector<std::vector<float>> decode_batch(
        const std::vector<TokenId>& tokens) override {
        return decode_impl(tokens, /*all_logits=*/true);
    }

    std::vector<std::vector<float>> decode_seqs(
        const std::vector<SeqSlice>& slices) override {
        std::int32_t total = 0;
        for (const auto& s : slices)
            total += static_cast<std::int32_t>(s.tokens.size());
        if (total == 0) throw ModelError("decode_seqs with no tokens");

        llama_batch batch = llama_batch_init(total, /*embd=*/0,
                                             /*n_seq_max=*/1);
        batch.n_tokens = total;
        std::vector<std::int32_t> slice_end(slices.size());
        std::int32_t i = 0;
        for (std::size_t si = 0; si < slices.size(); ++si) {
            const auto& s = slices[si];
            for (std::size_t t = 0; t < s.tokens.size(); ++t, ++i) {
                batch.token[i] = s.tokens[t];
                batch.pos[i] =
                    static_cast<llama_pos>(s.pos_start + t);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = s.seq_id;
                batch.logits[i] =
                    static_cast<int8_t>(t + 1 == s.tokens.size());
            }
            slice_end[si] = i - 1;
        }

        const int32_t rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0)
            throw ModelError("llama_decode (multi-seq) failed with status " +
                             std::to_string(rc));

        const auto n_vocab = static_cast<std::size_t>(vocab_size());
        std::vector<std::vector<float>> out;
        out.reserve(slices.size());
        for (const auto end : slice_end) {
            const float* logits = llama_get_logits_ith(ctx_, end);
            if (logits == nullptr)
                throw ModelError("no logits at multi-seq batch position " +
                                 std::to_string(end));
            out.emplace_back(logits, logits + n_vocab);
        }
        return out;
    }

    std::string format_chat(const std::string& user_message) override {
        // Template from the GGUF metadata; template-less models get raw
        // completion (the message unchanged).
        const char* tmpl = llama_model_chat_template(model_, nullptr);
        if (tmpl == nullptr) return user_message;

        llama_chat_message msg{"user", user_message.c_str()};
        std::vector<char> buf(user_message.size() + 512);
        std::int32_t n = llama_chat_apply_template(
            tmpl, &msg, 1, /*add_assistant=*/true, buf.data(),
            static_cast<std::int32_t>(buf.size()));
        if (n < 0) return user_message;  // unsupported template: raw
        if (static_cast<std::size_t>(n) > buf.size()) {
            buf.resize(static_cast<std::size_t>(n));
            n = llama_chat_apply_template(
                tmpl, &msg, 1, true, buf.data(),
                static_cast<std::int32_t>(buf.size()));
            if (n < 0) return user_message;
        }
        return std::string(buf.data(), static_cast<std::size_t>(n));
    }

    std::string format_chat(
        const std::vector<ChatMessage>& messages) override {
        // Template-less models (and unsupported templates) fall back to
        // plain role-prefixed turns — the best a raw completion model can
        // do with a conversation.
        const auto fallback = [&messages] {
            std::string out;
            for (const auto& m : messages)
                out += m.role + ": " + m.content + "\n";
            return out + "assistant:";
        };
        const char* tmpl = llama_model_chat_template(model_, nullptr);
        if (tmpl == nullptr) return fallback();

        std::vector<llama_chat_message> msgs;
        std::size_t text_size = 0;
        msgs.reserve(messages.size());
        for (const auto& m : messages) {
            msgs.push_back({m.role.c_str(), m.content.c_str()});
            text_size += m.role.size() + m.content.size();
        }
        std::vector<char> buf(text_size + 1024);
        std::int32_t n = llama_chat_apply_template(
            tmpl, msgs.data(), msgs.size(), /*add_assistant=*/true,
            buf.data(), static_cast<std::int32_t>(buf.size()));
        if (n < 0) return fallback();
        if (static_cast<std::size_t>(n) > buf.size()) {
            buf.resize(static_cast<std::size_t>(n));
            n = llama_chat_apply_template(
                tmpl, msgs.data(), msgs.size(), true, buf.data(),
                static_cast<std::int32_t>(buf.size()));
            if (n < 0) return fallback();
        }
        return std::string(buf.data(), static_cast<std::size_t>(n));
    }

    void copy_seq(std::int32_t src, std::int32_t dst,
                  std::uint32_t n_tokens) override {
        llama_memory_seq_cp(llama_get_memory(ctx_), src, dst, 0,
                            static_cast<llama_pos>(n_tokens));
    }

    void clear_seq(std::int32_t seq_id) override {
        llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, -1, -1);
    }

    void truncate_to(std::uint32_t n_tokens) override {
        if (n_tokens >= n_past_) return;
        if (!llama_memory_seq_rm(llama_get_memory(ctx_), /*seq_id=*/0,
                                 static_cast<llama_pos>(n_tokens),
                                 /*p1=*/-1))
            throw ModelError("failed to truncate sequence to " +
                             std::to_string(n_tokens) + " tokens");
        n_past_ = n_tokens;
    }

    void reset() override {
        llama_memory_clear(llama_get_memory(ctx_), /*data=*/true);
        n_past_ = 0;
    }

    std::uint32_t n_past() const override { return n_past_; }

    std::vector<char> state_data() override {
        const std::size_t size = llama_state_get_size(ctx_);
        std::vector<char> data(size);
        const std::size_t written = llama_state_get_data(
            ctx_, reinterpret_cast<uint8_t*>(data.data()), size);
        if (written == 0)
            throw ModelError("llama_state_get_data failed");
        data.resize(written);
        return data;
    }

    void set_state(const std::vector<char>& data,
                   std::uint32_t n_past) override {
        const std::size_t read = llama_state_set_data(
            ctx_, reinterpret_cast<const uint8_t*>(data.data()), data.size());
        if (read == 0)
            throw ModelError("llama_state_set_data failed (incompatible or "
                             "truncated state blob)");
        n_past_ = n_past;
    }

    std::int32_t vocab_size() const override {
        return llama_vocab_n_tokens(vocab_);
    }

    std::uint32_t context_length() const override { return llama_n_ctx(ctx_); }

    TokenId eos_token() const override { return llama_vocab_eos(vocab_); }

    bool is_eog(TokenId token) const override {
        return llama_vocab_is_eog(vocab_, token);
    }

private:
    // Explicit positions and logits flags: speculative decoding truncates
    // the sequence, which llama's automatic position tracking does not
    // survive. Returns one logits vector per requested output position
    // (only the last one unless all_logits).
    std::vector<std::vector<float>> decode_impl(
        const std::vector<TokenId>& tokens, bool all_logits) {
        const auto n = static_cast<int32_t>(tokens.size());
        if (n == 0) throw ModelError("decode of zero tokens");

        llama_batch batch = llama_batch_init(n, /*embd=*/0, /*n_seq_max=*/1);
        batch.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            batch.token[i] = tokens[static_cast<std::size_t>(i)];
            batch.pos[i] = static_cast<llama_pos>(n_past_) + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = static_cast<int8_t>(all_logits || i == n - 1);
        }

        const int32_t rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0)
            throw ModelError("llama_decode failed with status " +
                             std::to_string(rc));
        n_past_ += static_cast<std::uint32_t>(n);

        const auto n_vocab = static_cast<std::size_t>(vocab_size());
        std::vector<std::vector<float>> out;
        for (int32_t i = all_logits ? 0 : n - 1; i < n; ++i) {
            const float* logits = llama_get_logits_ith(ctx_, i);
            if (logits == nullptr)
                throw ModelError("no logits returned at batch position " +
                                 std::to_string(i));
            out.emplace_back(logits, logits + n_vocab);
        }
        return out;
    }

    std::uint32_t n_past_ = 0;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
};

}  // namespace

std::unique_ptr<LlamaBackend> make_llama_backend(const ModelParams& params) {
    return std::make_unique<RealLlamaBackend>(params);
}

}  // namespace reame
