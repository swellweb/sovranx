#include "sovrano/core/engine.hpp"

#include <map>
#include <utility>
#include <vector>

#include "sovrano/cache/cache_manager.hpp"
#include "sovrano/core/model.hpp"
#include "sovrano/core/sampler.hpp"
#include "sovrano/speculative/speculative_decoder.hpp"

namespace sovrano::core {

namespace {

void validate(const SovranoEngine::Config& c) {
    if (c.model_path.empty())
        throw EngineError("model_path is empty");
    if (c.n_ctx <= 0)
        throw EngineError("n_ctx must be positive, got " +
                          std::to_string(c.n_ctx));
    if (c.n_threads <= 0)
        throw EngineError("n_threads must be positive, got " +
                          std::to_string(c.n_threads));
}

ModelParams to_model_params(const SovranoEngine::Config& c) {
    ModelParams p;
    p.path = c.model_path;
    p.context_length = c.n_ctx;
    p.threads = c.n_threads;
    p.use_mmap = c.use_mmap;
    p.use_mlock = c.use_mlock;
    return p;
}

}  // namespace

struct SovranoEngine::Impl {
    std::unique_ptr<LlamaBackend> backend;
    std::unique_ptr<LlamaBackend> draft_backend;
    std::unique_ptr<speculative::SpeculativeDecoder> decoder;
    std::unique_ptr<cache::CacheManager> cache;
    std::string model_tag;  // discriminates cache entries across models
    // Tokens currently represented in the KV cache (prompt + generated of
    // the last generate/load_session).
    std::vector<TokenId> context_tokens;
    // Sessions: named snapshots of context_tokens.
    std::map<std::string, std::vector<TokenId>> sessions;
    std::uint64_t next_session_id = 1;
};

SovranoEngine::SovranoEngine(const Config& config)
    : SovranoEngine(
          config,
          [&config] {
              validate(config);  // fail fast, before touching llama.cpp
              return make_llama_backend(to_model_params(config));
          }(),
          [&config]() -> std::unique_ptr<LlamaBackend> {
              if (config.draft_model_path.empty() || !config.use_speculative)
                  return nullptr;
              auto params = to_model_params(config);
              params.path = config.draft_model_path;
              return make_llama_backend(params);
          }()) {}

SovranoEngine::SovranoEngine(const Config& config,
                             std::unique_ptr<LlamaBackend> backend)
    : SovranoEngine(config, std::move(backend), nullptr) {}

SovranoEngine::SovranoEngine(const Config& config,
                             std::unique_ptr<LlamaBackend> backend,
                             std::unique_ptr<LlamaBackend> draft_backend) {
    validate(config);
    if (backend == nullptr)
        throw EngineError("backend is null");
    pimpl_ = std::make_unique<Impl>();
    pimpl_->backend = std::move(backend);
    pimpl_->model_tag = config.model_path;
    if (!config.cache_dir.empty()) {
        cache::CacheManager::Config cc;
        cc.directory = config.cache_dir;
        cc.max_bytes = config.cache_max_mb * 1024ull * 1024ull;
        cc.compress = config.cache_compress;
        pimpl_->cache = std::make_unique<cache::CacheManager>(cc);
    }
    if (draft_backend != nullptr && config.use_speculative) {
        pimpl_->draft_backend = std::move(draft_backend);
        speculative::SpeculativeDecoder::Config dc;
        dc.draft_tokens = config.draft_tokens;
        pimpl_->decoder = std::make_unique<speculative::SpeculativeDecoder>(
            *pimpl_->backend, pimpl_->draft_backend.get(), dc);
    }
}

SovranoEngine::~SovranoEngine() = default;
SovranoEngine::SovranoEngine(SovranoEngine&&) noexcept = default;
SovranoEngine& SovranoEngine::operator=(SovranoEngine&&) noexcept = default;

std::string SovranoEngine::generate(const std::string& prompt,
                                    const GenerationConfig& gen_config) {
    std::string out;
    generate_stream(
        prompt,
        [&out](const std::string& piece) {
            out += piece;
            return true;
        },
        gen_config);
    return out;
}

void SovranoEngine::generate_stream(
    const std::string& prompt,
    std::function<bool(const std::string& token)> callback,
    const GenerationConfig& gen_config) {
    if (!callback)
        throw EngineError("callback is null");

    LlamaBackend& backend = *pimpl_->backend;

    std::vector<TokenId> tokens = backend.tokenize(prompt, /*add_special=*/true);
    if (tokens.empty())
        throw EngineError("prompt tokenized to zero tokens");

    const auto n_ctx = static_cast<std::size_t>(backend.context_length());
    if (tokens.size() > n_ctx)
        throw EngineError("prompt of " + std::to_string(tokens.size()) +
                          " tokens exceeds context length " +
                          std::to_string(n_ctx));

    if (gen_config.echo_prompt && !callback(prompt)) return;

    if (pimpl_->decoder != nullptr) {
        pimpl_->decoder->generate_stream(
            tokens,
            [&](TokenId t) {
                tokens.push_back(t);
                return callback(backend.token_piece(t));
            },
            gen_config);
        pimpl_->context_tokens = std::move(tokens);
        return;
    }

    Sampler sampler(gen_config);
    const TokenId eos = backend.eos_token();

    // Prefill; from here `tokens` mirrors the KV cache content.
    std::vector<float> logits;
    if (pimpl_->cache != nullptr) {
        // Split prefill: the snapshot covers the prompt minus its last
        // token; that token is always decoded fresh, so the sampling
        // logits come from a real forward pass in both cold and warm runs.
        const std::vector<TokenId> prefix(tokens.begin(), tokens.end() - 1);
        if (prefix.empty()) {
            backend.reset();
        } else {
            const auto key =
                cache::CacheManager::make_key(pimpl_->model_tag, prefix);
            if (!pimpl_->cache->load_state(key, backend)) {
                backend.reset();
                backend.decode_append(prefix);
                pimpl_->cache->store_state(
                    key, backend, static_cast<std::uint32_t>(prefix.size()));
            }
        }
        logits = backend.decode_append({tokens.back()});
    } else {
        logits = backend.decode(tokens);
    }

    for (int produced = 0; produced < gen_config.max_tokens; ++produced) {
        const TokenId next = sampler.sample(std::move(logits), tokens);
        if (next == eos) break;

        const std::string piece = backend.token_piece(next);
        if (!callback(piece)) break;
        tokens.push_back(next);

        if (tokens.size() >= n_ctx) break;              // context full
        if (produced + 1 >= gen_config.max_tokens) break;  // budget spent

        logits = backend.decode_append({next});
    }

    pimpl_->context_tokens = std::move(tokens);
}

std::string SovranoEngine::create_session() {
    const std::string id = "sess-" + std::to_string(pimpl_->next_session_id++);
    pimpl_->sessions.emplace(id, std::vector<TokenId>{});
    return id;
}

namespace {

std::string session_cache_key(const std::string& model_tag,
                              const std::string& session_id,
                              const std::vector<TokenId>& tokens) {
    return cache::CacheManager::make_key(model_tag + "/session/" + session_id,
                                         tokens);
}

}  // namespace

void SovranoEngine::save_session(const std::string& session_id) {
    const auto it = pimpl_->sessions.find(session_id);
    if (it == pimpl_->sessions.end())
        throw EngineError("unknown session: " + session_id);
    it->second = pimpl_->context_tokens;
    if (pimpl_->cache != nullptr && !it->second.empty())
        pimpl_->cache->store_state(
            session_cache_key(pimpl_->model_tag, session_id, it->second),
            *pimpl_->backend, static_cast<std::uint32_t>(it->second.size()));
}

void SovranoEngine::load_session(const std::string& session_id) {
    const auto it = pimpl_->sessions.find(session_id);
    if (it == pimpl_->sessions.end())
        throw EngineError("unknown session: " + session_id);
    if (!it->second.empty()) {
        bool restored = false;
        if (pimpl_->cache != nullptr)
            restored = pimpl_->cache->load_state(
                session_cache_key(pimpl_->model_tag, session_id, it->second),
                *pimpl_->backend);
        if (!restored)
            pimpl_->backend->decode(it->second);  // re-prefill the KV cache
    }
    pimpl_->context_tokens = it->second;
}

void SovranoEngine::delete_session(const std::string& session_id) {
    if (pimpl_->sessions.erase(session_id) == 0)
        throw EngineError("unknown session: " + session_id);
}

int SovranoEngine::context_size() const {
    return static_cast<int>(pimpl_->backend->context_length());
}

int SovranoEngine::vocab_size() const {
    return pimpl_->backend->vocab_size();
}

const speculative::SpeculativeMetrics* SovranoEngine::speculative_metrics()
    const {
    return pimpl_->decoder == nullptr ? nullptr : &pimpl_->decoder->metrics();
}

}  // namespace sovrano::core
