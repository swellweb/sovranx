#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "reame/core/llama_backend.hpp"

namespace reame::speculative {
struct SpeculativeMetrics;
}

namespace reame::cache {
struct CacheStats;
}

namespace reame::core {

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
class ReameEngine {
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
        // KV-cache element type: f16 | q8_0 | q4_0 (see ModelParams).
        std::string kv_cache_type = "f16";
        // Physical prompt-processing batch (0 = default 512).
        int n_ubatch = 0;
        bool use_speculative = true;
        // Speculate from n-gram matches in the prompt/history instead of a
        // draft model (zero draft cost; no draft_model_path needed).
        bool use_prompt_lookup = false;
        int draft_tokens = 16;  // starting speculative draft length
        // KV-cache persistence (DwarfStar4-style). Empty = disabled. When
        // set, prompt prefixes are snapshotted to disk and reused across
        // engines/processes, and sessions restore from disk instead of
        // re-prefilling. Applies to the classic path (the speculative
        // decoder manages its own prefill).
        std::string cache_dir;
        std::uint64_t cache_max_mb = 512;  // 0 = unlimited
        bool cache_compress = true;
        // Prefix snapshots are taken every this many tokens: smaller =
        // more cross-prompt hits, more disk writes.
        int cache_block_tokens = 256;
        // Interleaved multi-user serving: N concurrent generations share
        // each read of the model weights (context_length becomes the
        // TOTAL KV budget). Mutually exclusive with speculative decoding
        // and the disk cache for now — the constructor rejects the combo.
        int n_parallel = 1;
        bool verbose = false;
    };

    // Production: loads the model(s) through the real llama.cpp backend.
    explicit ReameEngine(const Config& config);

    // Test seam: inject a target backend (mock). Still validates `config`.
    ReameEngine(const Config& config, std::unique_ptr<LlamaBackend> backend);

    // Test seam: inject target + draft backends (speculative decoding).
    ReameEngine(const Config& config, std::unique_ptr<LlamaBackend> backend,
                  std::unique_ptr<LlamaBackend> draft_backend);

    ~ReameEngine();
    ReameEngine(ReameEngine&&) noexcept;
    ReameEngine& operator=(ReameEngine&&) noexcept;
    ReameEngine(const ReameEngine&) = delete;
    ReameEngine& operator=(const ReameEngine&) = delete;

    std::string generate(const std::string& prompt,
                         const GenerationConfig& gen_config = {});

    // The Conclave: n attempts at the same prompt (attempt 0 is the
    // untouched anchor, explorers shift seed and heat up) and the
    // consensus answer wins. With n_parallel >= n the attempts share
    // weight reads in interleaved batches; with n_parallel == 1 they run
    // sequentially (correct, slower). When one final number reaches an
    // absolute majority the verdict is early: stragglers are stopped.
    // `consensus_votes`, when non-null, receives how many candidates
    // agreed on the winning final number (1 = no agreement — a caller
    // can escalate: retry with a reasoning prompt only when the conclave
    // is split).
    std::string generate_best(const std::string& prompt,
                              const GenerationConfig& gen_config, int n,
                              int* consensus_votes = nullptr);

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

    // Wraps a user message in the model's own chat template (from the
    // GGUF metadata); returns it unchanged for template-less models.
    std::string format_chat(const std::string& user_message) const;

    int context_size() const;
    int vocab_size() const;
    // Token count of `text` under the model's tokenizer (for API usage
    // accounting).
    int count_tokens(const std::string& text) const;

    // Non-null only when the speculative decoder is in use.
    const speculative::SpeculativeMetrics* speculative_metrics() const;

    // Non-null only when the disk cache is enabled (cache_dir set).
    const cache::CacheStats* cache_stats() const;

    // True when n_parallel > 1: generate/generate_stream may be called
    // concurrently from multiple threads and requests are interleaved.
    bool parallel_capable() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// True when the config calls for loading a second (draft) model: speculation
// on, mode = model (not prompt-lookup) and a draft_model_path set. Lookup
// mode drafts from n-grams and must never load the second model, even when
// a stale draft_model_path is still present in the config.
bool wants_draft_backend(const ReameEngine::Config& config);

// Fail-fast filesystem check for every model file the config will load
// (main model, plus the draft model when wants_draft_backend). Returns ""
// when they all exist, otherwise a human-readable error naming the missing
// file and how to fix it — so the CLI can fail before llama.cpp surfaces a
// bare "No such file or directory".
std::string missing_model_file_error(const ReameEngine::Config& config);

}  // namespace reame::core
