#include "sovrano/core/autoconfig.hpp"

#include <algorithm>
#include <cctype>

namespace sovrano::core {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

}  // namespace

const std::vector<ModelSpec>& model_catalog() {
    // Curated CPU-friendly picks. Kept small on purpose.
    static const std::vector<ModelSpec> catalog = {
        {"tinyllama",
         "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
         "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
         "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 2048,
         "1.1B · ~670 MB · fast, basic quality"},
        {"qwen2.5-0.5b",
         "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/"
         "main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
         "qwen2.5-0.5b-instruct-q4_k_m.gguf", 4096,
         "0.5B · ~400 MB · very fast, tiny"},
        {"qwen2.5-1.5b",
         "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/"
         "main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
         "qwen2.5-1.5b-instruct-q4_k_m.gguf", 4096,
         "1.5B · ~1 GB · good default on a laptop"},
        {"qwen2.5-3b",
         "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/"
         "main/qwen2.5-3b-instruct-q4_k_m.gguf",
         "qwen2.5-3b-instruct-q4_k_m.gguf", 4096,
         "3B · ~2 GB · stronger, still light"},
    };
    return catalog;
}

std::optional<ModelSpec> resolve_model(const std::string& token) {
    const std::string t = lower(token);

    // Exact alias first.
    for (const auto& m : model_catalog())
        if (lower(m.name) == t) return m;

    // Unambiguous case-insensitive prefix.
    const ModelSpec* match = nullptr;
    for (const auto& m : model_catalog()) {
        if (lower(m.name).rfind(t, 0) == 0) {
            if (match != nullptr) return std::nullopt;  // ambiguous
            match = &m;
        }
    }
    if (match != nullptr) return *match;
    return std::nullopt;
}

int auto_threads(unsigned hw) {
    if (hw == 0) return 1;
    if (hw <= 4) return static_cast<int>(hw);
    return static_cast<int>(hw) - 1;  // leave one for the OS/UI
}

SovranoEngine::Config auto_config(const std::string& model_path,
                                  const std::string& home, unsigned hw,
                                  int ctx) {
    SovranoEngine::Config c;
    c.model_path = model_path;
    c.n_threads = auto_threads(hw);
    c.n_ctx = ctx > 0 ? ctx : 4096;
    c.kv_cache_type = "q8_0";       // halve context RAM by default
    c.use_speculative = true;
    c.use_prompt_lookup = true;     // free speedup, needs no draft model

    std::string base;
    if (!home.empty()) {
        base = home + "/.sovrano";
    } else {
        // No home: sit next to the model, but never at the filesystem
        // root (a model at "/m.gguf" must not push the cache into "/").
        const auto slash = model_path.rfind('/');
        const std::string dir =
            slash == std::string::npos ? "" : model_path.substr(0, slash);
        base = (dir.empty() ? "." : dir) + "/.sovrano";
    }
    c.cache_dir = base + "/cache";
    return c;
}

}  // namespace sovrano::core
