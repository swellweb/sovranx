#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "sovrano/core/llama_backend.hpp"

namespace sovrano::cache {

class CacheError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// KV-cache (de)serialization primitives, DwarfStar4-style: the state is a
// byte blob that can be compressed, checksummed and parked on NVMe.
//
// The llama.cpp specifics stay behind LlamaBackend::state_data/set_state;
// everything here operates on bytes and is testable without a model.
class KVCacheSerializer {
public:
    // Snapshot of the backend's full state (RNG, logits, KV cache).
    static std::vector<char> serialize(LlamaBackend& backend);

    // Restores a snapshot. `n_past` = sequence positions the blob holds
    // (the blob itself does not carry the wrapper's counter).
    static void deserialize(LlamaBackend& backend,
                            const std::vector<char>& data,
                            std::uint32_t n_past);

    // Compression envelope: 1 flag byte ('Z' = zstd frame, 'R' = raw)
    // followed by the payload. With zstd unavailable at build time,
    // compress() always emits 'R' and decompress() still reads both
    // formats it can ('Z' without zstd -> CacheError).
    static std::vector<char> compress(const std::vector<char>& data);
    static std::vector<char> decompress(const std::vector<char>& data);

    // FNV-1a 64-bit checksum (integrity check for on-disk entries).
    static std::uint64_t checksum(const std::vector<char>& data);
};

}  // namespace sovrano::cache
