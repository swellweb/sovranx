#include "sovranx/cache/prefix_cache.hpp"

#include <algorithm>

#include "sovranx/cache/cache_serializer.hpp"

namespace sovranx::cache {

PrefixCache::PrefixCache(CacheManager& manager, std::string model_tag,
                         int block_tokens)
    : manager_(manager), model_tag_(std::move(model_tag)),
      block_tokens_(static_cast<std::uint32_t>(std::max(block_tokens, 1))) {}

std::vector<std::uint32_t> PrefixCache::boundaries(std::size_t size) const {
    std::vector<std::uint32_t> out;
    for (std::uint32_t b = block_tokens_; b < size; b += block_tokens_)
        out.push_back(b);
    if (size > 0) out.push_back(static_cast<std::uint32_t>(size));
    return out;
}

std::string PrefixCache::key_for(const std::vector<TokenId>& prefix,
                                 std::uint32_t n) const {
    std::vector<char> raw(model_tag_.begin(), model_tag_.end());
    raw.push_back('\0');
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto t = static_cast<std::uint32_t>(prefix[i]);
        for (int b = 0; b < 4; ++b)
            raw.push_back(static_cast<char>((t >> (8 * b)) & 0xFF));
    }
    const std::uint64_t h1 = KVCacheSerializer::checksum(raw);
    raw.push_back('x');
    const std::uint64_t h2 = KVCacheSerializer::checksum(raw);
    return "pfx-" + std::to_string(h1) + "-" + std::to_string(h2) + "-n" +
           std::to_string(n);
}

std::uint32_t PrefixCache::prefill(const std::vector<TokenId>& prefix,
                                   LlamaBackend& backend) {
    if (prefix.empty()) {
        backend.reset();
        return 0;
    }

    const auto marks = boundaries(prefix.size());

    // Longest cached boundary first.
    std::uint32_t restored = 0;
    for (auto it = marks.rbegin(); it != marks.rend(); ++it) {
        if (manager_.load_state(key_for(prefix, *it), backend)) {
            restored = *it;
            break;
        }
    }
    if (restored == 0) backend.reset();

    // Decode the remainder block-wise, snapshotting each boundary crossed.
    std::uint32_t pos = restored;
    for (const auto b : marks) {
        if (b <= restored) continue;
        backend.decode_append({prefix.begin() + pos, prefix.begin() + b});
        manager_.store_state(key_for(prefix, b), backend, b);
        pos = b;
    }
    return restored;
}

}  // namespace sovranx::cache
