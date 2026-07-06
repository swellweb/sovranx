#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sovrano/cache/disk_cache_store.hpp"
#include "sovrano/core/llama_backend.hpp"

namespace sovrano::cache {

struct CacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t stores = 0;
};

// Ties serializer + disk store together: named KV-cache snapshots that
// survive the process. Entry layout (inside the store's payload):
//   "SVK1" magic | u64 checksum(compressed) | u32 n_past | compressed blob
// Corrupt entries fail the checksum and are evicted on load.
class CacheManager {
public:
    struct Config {
        std::filesystem::path directory;
        std::uint64_t max_bytes = 0;  // 0 = unlimited
        bool compress = true;
    };

    explicit CacheManager(const Config& cfg);

    // Deterministic key for a token prefix under a given model.
    static std::string make_key(const std::string& model_tag,
                                const std::vector<TokenId>& tokens);

    // Snapshot the backend state under `key`. `n_past` = positions the
    // snapshot represents. Returns false if the entry cannot be stored
    // (e.g. larger than the whole budget).
    bool store_state(const std::string& key, LlamaBackend& backend,
                     std::uint32_t n_past);

    // Restores the snapshot into the backend. False on miss or corruption
    // (corrupt entries are removed).
    bool load_state(const std::string& key, LlamaBackend& backend);

    const CacheStats& stats() const { return stats_; }

private:
    DiskCacheStore store_;
    bool compress_;
    CacheStats stats_;
};

}  // namespace sovrano::cache
