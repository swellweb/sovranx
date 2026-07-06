// Real llama.cpp backend. This is the ONLY translation unit that includes
// llama.h; it is compiled only when the submodule is present
// (SOVRANO_HAS_LLAMA). CPU-only by design: n_gpu_layers = 0.

#include <llama.h>

#include <cstdio>
#include <mutex>
#include <vector>

#include "sovrano/core/llama_backend.hpp"
#include "sovrano/core/model.hpp"

namespace sovrano {

namespace {

void ensure_backend_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        llama_backend_init();
        // Keep llama.cpp's own logging quiet; Sovrano has its own logger.
        llama_log_set(
            [](ggml_log_level level, const char* text, void*) {
                if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
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
        cparams.n_threads = params.threads;
        cparams.n_threads_batch = params.threads;

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
        int32_t n = llama_tokenize(vocab_, text.c_str(), text_len,
                                   nullptr, 0, add_special,
                                   /*parse_special=*/false);
        if (n == 0) return {};
        if (n == INT32_MIN)
            throw ModelError("tokenization overflow for input of " +
                             std::to_string(text.size()) + " bytes");
        if (n < 0) n = -n;

        std::vector<TokenId> tokens(static_cast<std::size_t>(n));
        const int32_t written =
            llama_tokenize(vocab_, text.c_str(), text_len, tokens.data(), n,
                           add_special, /*parse_special=*/false);
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
        // Fresh sequence: clear the KV cache from prior forwards.
        llama_memory_clear(llama_get_memory(ctx_), /*data=*/true);
        return decode_impl(tokens);
    }

    std::vector<float> decode_append(const std::vector<TokenId>& tokens) override {
        return decode_impl(tokens);  // KV cache kept, positions auto-tracked
    }

    std::int32_t vocab_size() const override {
        return llama_vocab_n_tokens(vocab_);
    }

    std::uint32_t context_length() const override { return llama_n_ctx(ctx_); }

    TokenId eos_token() const override { return llama_vocab_eos(vocab_); }

private:
    std::vector<float> decode_impl(const std::vector<TokenId>& tokens) {
        // llama_batch_get_one does not take ownership but wants mutable data.
        std::vector<TokenId> input = tokens;
        llama_batch batch =
            llama_batch_get_one(input.data(), static_cast<int32_t>(input.size()));

        const int32_t rc = llama_decode(ctx_, batch);
        if (rc != 0)
            throw ModelError("llama_decode failed with status " +
                             std::to_string(rc));

        const float* logits = llama_get_logits_ith(
            ctx_, static_cast<int32_t>(input.size()) - 1);
        if (logits == nullptr)
            throw ModelError("no logits returned for last token");

        const auto n_vocab = static_cast<std::size_t>(vocab_size());
        return std::vector<float>(logits, logits + n_vocab);
    }

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
};

}  // namespace

std::unique_ptr<LlamaBackend> make_llama_backend(const ModelParams& params) {
    return std::make_unique<RealLlamaBackend>(params);
}

}  // namespace sovrano
