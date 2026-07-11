#include "sovranx/core/model.hpp"

#include <thread>
#include <utility>

#include "sovranx/utils/config.hpp"

namespace sovranx {

namespace {

void validate(const ModelParams& p) {
    if (p.path.empty())
        throw ModelError("model path is empty");
    if (p.context_length <= 0)
        throw ModelError("context_length must be positive, got " +
                         std::to_string(p.context_length));
    if (p.threads <= 0)
        throw ModelError("threads must be positive, got " +
                         std::to_string(p.threads));
}

std::int32_t default_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<std::int32_t>(hw);
}

}  // namespace

ModelParams ModelParams::from_config(const Config& cfg) {
    ModelParams p;
    try {
        p.path = cfg.get_string("model.path");
    } catch (const ConfigError&) {
        throw ModelError("config key 'model.path' is required to load a model");
    }
    p.context_length =
        static_cast<std::int32_t>(cfg.get_int("model.context_length", 4096));
    p.threads = static_cast<std::int32_t>(
        cfg.get_int("model.threads", default_threads()));
    p.use_mmap = cfg.get_bool("memory.use_mmap", true);
    p.use_mlock = cfg.get_bool("memory.use_mlock", false);
    p.kv_cache_type = cfg.get_string("memory.kv_cache_type", "f16");
    if (p.kv_cache_type != "f16" && p.kv_cache_type != "q8_0" &&
        p.kv_cache_type != "q4_0")
        throw ModelError("memory.kv_cache_type must be f16, q8_0 or q4_0, "
                         "got '" + p.kv_cache_type + "'");
    return p;
}

struct LlamaModel::Impl {
    std::unique_ptr<LlamaBackend> backend;
};

LlamaModel::LlamaModel(const ModelParams& params)
    : LlamaModel(params, [&params] {
          validate(params);  // fail fast, before touching llama.cpp
          return make_llama_backend(params);
      }()) {}

LlamaModel::LlamaModel(const ModelParams& params,
                       std::unique_ptr<LlamaBackend> backend) {
    validate(params);
    if (backend == nullptr)
        throw ModelError("backend is null");
    impl_ = std::make_unique<Impl>();
    impl_->backend = std::move(backend);
}

LlamaModel::~LlamaModel() = default;
LlamaModel::LlamaModel(LlamaModel&&) noexcept = default;
LlamaModel& LlamaModel::operator=(LlamaModel&&) noexcept = default;

std::vector<TokenId> LlamaModel::tokenize(const std::string& text,
                                          bool add_special) const {
    return impl_->backend->tokenize(text, add_special);
}

std::string LlamaModel::detokenize(const std::vector<TokenId>& tokens) const {
    if (tokens.empty()) return {};
    const std::int32_t n_vocab = impl_->backend->vocab_size();
    for (const TokenId t : tokens) {
        if (t < 0 || t >= n_vocab)
            throw ModelError("token id " + std::to_string(t) +
                             " out of range [0, " + std::to_string(n_vocab) + ")");
    }
    return impl_->backend->detokenize(tokens);
}

std::vector<float> LlamaModel::forward(const std::vector<TokenId>& tokens) {
    if (tokens.empty())
        throw ModelError("forward pass requires at least one token");
    const auto n_ctx = impl_->backend->context_length();
    if (tokens.size() > n_ctx)
        throw ModelError("input of " + std::to_string(tokens.size()) +
                         " tokens exceeds context length " +
                         std::to_string(n_ctx));

    auto logits = impl_->backend->decode(tokens);

    const auto n_vocab = static_cast<std::size_t>(impl_->backend->vocab_size());
    if (logits.size() != n_vocab)
        throw ModelError("backend returned " + std::to_string(logits.size()) +
                         " logits, expected vocab size " +
                         std::to_string(n_vocab));
    return logits;
}

std::int32_t LlamaModel::vocab_size() const {
    return impl_->backend->vocab_size();
}

std::uint32_t LlamaModel::context_length() const {
    return impl_->backend->context_length();
}

}  // namespace sovranx
