#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "sovrano/core/llama_backend.hpp"

namespace sovrano::speculative {
struct SpeculativeMetrics;
}

namespace sovrano::core {

class EngineError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct GenerationConfig {
    int max_tokens = 512;
    float temperature = 0.7f;    // <= 0 means greedy (argmax)
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    int seed = 42;
    bool echo_prompt = false;
};

// Base generation engine: tokenize -> prefill -> sequential decode with
// sampling. No speculative decoding yet (Step 4) and sessions are
// in-memory snapshots of the context tokens (persistent cache is Step 5).
class SovranoEngine {
public:
    struct Config {
        std::string model_path;
        // When set (and use_speculative), a draft model is loaded and
        // generation goes through the speculative decoder. Both models
        // share n_threads (they run strictly in turn, never concurrently,
        // so the CPU pool is effectively shared).
        std::string draft_model_path;
        int n_ctx = 8192;
        int n_threads = 18;
        bool use_mmap = true;
        bool use_mlock = false;
        bool use_speculative = true;
        int draft_tokens = 16;  // starting speculative draft length
        // KV-cache persistence (DwarfStar4-style). Empty = disabled. When
        // set, prompt prefixes are snapshotted to disk and reused across
        // engines/processes, and sessions restore from disk instead of
        // re-prefilling. Applies to the classic path (the speculative
        // decoder manages its own prefill).
        std::string cache_dir;
        std::uint64_t cache_max_mb = 512;  // 0 = unlimited
        bool cache_compress = true;
        bool verbose = false;
    };

    // Production: loads the model(s) through the real llama.cpp backend.
    explicit SovranoEngine(const Config& config);

    // Test seam: inject a target backend (mock). Still validates `config`.
    SovranoEngine(const Config& config, std::unique_ptr<LlamaBackend> backend);

    // Test seam: inject target + draft backends (speculative decoding).
    SovranoEngine(const Config& config, std::unique_ptr<LlamaBackend> backend,
                  std::unique_ptr<LlamaBackend> draft_backend);

    ~SovranoEngine();
    SovranoEngine(SovranoEngine&&) noexcept;
    SovranoEngine& operator=(SovranoEngine&&) noexcept;
    SovranoEngine(const SovranoEngine&) = delete;
    SovranoEngine& operator=(const SovranoEngine&) = delete;

    std::string generate(const std::string& prompt,
                         const GenerationConfig& gen_config = {});

    // Streaming: `callback` receives each decoded piece; returning false
    // stops generation. With echo_prompt the prompt is emitted first as a
    // single piece.
    void generate_stream(const std::string& prompt,
                         std::function<bool(const std::string& token)> callback,
                         const GenerationConfig& gen_config = {});

    // Sessions: named snapshots of the current context tokens. load_session
    // re-prefills the model with the saved tokens. Unknown ids throw.
    std::string create_session();
    void save_session(const std::string& session_id);
    void load_session(const std::string& session_id);
    void delete_session(const std::string& session_id);

    int context_size() const;
    int vocab_size() const;

    // Non-null only when the speculative decoder is in use.
    const speculative::SpeculativeMetrics* speculative_metrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace sovrano::core
