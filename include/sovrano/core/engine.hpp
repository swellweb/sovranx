#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "sovrano/core/llama_backend.hpp"

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
        std::string draft_model_path;  // unused until Step 4
        int n_ctx = 8192;
        int n_threads = 18;
        bool use_mmap = true;
        bool use_mlock = false;
        bool verbose = false;
    };

    // Production: loads the model through the real llama.cpp backend.
    explicit SovranoEngine(const Config& config);

    // Test seam: inject a backend (mock). Still validates `config`.
    SovranoEngine(const Config& config, std::unique_ptr<LlamaBackend> backend);

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

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace sovrano::core
