#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sovranx/core/engine.hpp"

namespace sovranx::core {

// Zero-config path: one command, sensible defaults, no .conf to edit.
// `sovranx run qwen2.5-1.5b` downloads the model (once), picks threads,
// context and a cache location, and serves — like `ollama run`.

// A known model: a short alias mapped to a Hugging Face GGUF download.
struct ModelSpec {
    std::string name;      // alias the user types
    std::string url;       // direct GGUF download URL
    std::string filename;  // local filename under the models dir
    int default_ctx = 4096;
    std::string note;      // one-line description (RAM, size)
};

// Built-in catalog, small and curated (CPU-friendly picks).
const std::vector<ModelSpec>& model_catalog();

// Resolve a user token to a spec: an exact alias, a case-insensitive
// prefix match if unambiguous, else nullopt (caller may treat it as a path).
std::optional<ModelSpec> resolve_model(const std::string& token);

// A good thread count for this machine: physical-ish cores, leaving the
// last one free on larger boxes so the OS/UI stays responsive. `hw` is the
// reported hardware concurrency (0 => fall back to 1).
int auto_threads(unsigned hw);

// Fully-populated engine config from just a model path — home-dir cache,
// auto threads, quantized KV, prompt-lookup on. `home` is the user's home
// directory (cache goes under <home>/.sovranx/cache).
SovranXEngine::Config auto_config(const std::string& model_path,
                                  const std::string& home,
                                  unsigned hw,
                                  int ctx = 0);

}  // namespace sovranx::core
