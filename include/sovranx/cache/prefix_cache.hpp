#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sovranx/cache/cache_manager.hpp"
#include "sovranx/core/llama_backend.hpp"

namespace sovranx::cache {

// Shared-prefix KV cache: prompts are split into fixed token blocks and a
// chain hash is computed per block boundary, so DIFFERENT prompts that
// share a prefix (same system prompt, same RAG context) reuse each
// other's snapshots — not just exact repeats.
//
//   prompt A: [ system prompt | question 1 ]  -> stores boundary snapshots
//   prompt B: [ system prompt | question 2 ]  -> restores the boundary
//                                               inside the shared part,
//                                               decodes only the tail
//
// Built on CacheManager (zstd, checksums, LRU budget all apply).
class PrefixCache {
public:
    // `block_tokens` trades storage for hit granularity: snapshots are
    // taken every `block_tokens` positions (plus one at the full prefix).
    PrefixCache(CacheManager& manager, std::string model_tag,
                int block_tokens = 256);

    // Snapshot positions for a prefix of `size` tokens: every multiple of
    // block_tokens, plus `size` itself. Empty when size == 0.
    std::vector<std::uint32_t> boundaries(std::size_t size) const;

    // Chain-hash key for the first `n` tokens of `prefix`.
    std::string key_for(const std::vector<TokenId>& prefix,
                        std::uint32_t n) const;

    // Ensures the backend's KV holds exactly `prefix`: restores the
    // longest cached boundary (falling back to reset+decode from scratch)
    // and decodes the remainder block-wise, snapshotting each boundary it
    // crosses. Returns the number of positions restored from cache
    // (0 = fully cold).
    std::uint32_t prefill(const std::vector<TokenId>& prefix,
                          LlamaBackend& backend);

private:
    CacheManager& manager_;
    std::string model_tag_;
    std::uint32_t block_tokens_;
};

}  // namespace sovranx::cache
