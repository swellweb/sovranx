#include "sovranx/cache/cache_serializer.hpp"

#ifdef SOVRANX_HAS_ZSTD
#include <zstd.h>
#endif

namespace sovranx::cache {

namespace {

constexpr char kFlagZstd = 'Z';
constexpr char kFlagRaw = 'R';

}  // namespace

std::vector<char> KVCacheSerializer::serialize(LlamaBackend& backend) {
    return backend.state_data();
}

void KVCacheSerializer::deserialize(LlamaBackend& backend,
                                    const std::vector<char>& data,
                                    std::uint32_t n_past) {
    backend.set_state(data, n_past);
}

std::vector<char> KVCacheSerializer::compress(const std::vector<char>& data) {
#ifdef SOVRANX_HAS_ZSTD
    const std::size_t bound = ZSTD_compressBound(data.size());
    std::vector<char> out(1 + bound);
    out[0] = kFlagZstd;
    const std::size_t written =
        ZSTD_compress(out.data() + 1, bound, data.data(), data.size(),
                      /*level=*/3);
    if (ZSTD_isError(written))
        throw CacheError(std::string("zstd compression failed: ") +
                         ZSTD_getErrorName(written));
    out.resize(1 + written);
    return out;
#else
    std::vector<char> out;
    out.reserve(1 + data.size());
    out.push_back(kFlagRaw);
    out.insert(out.end(), data.begin(), data.end());
    return out;
#endif
}

std::vector<char> KVCacheSerializer::decompress(const std::vector<char>& data) {
    if (data.empty())
        throw CacheError("empty compression envelope");

    if (data[0] == kFlagRaw)
        return std::vector<char>(data.begin() + 1, data.end());

    if (data[0] == kFlagZstd) {
#ifdef SOVRANX_HAS_ZSTD
        const unsigned long long raw_size =
            ZSTD_getFrameContentSize(data.data() + 1, data.size() - 1);
        if (raw_size == ZSTD_CONTENTSIZE_ERROR ||
            raw_size == ZSTD_CONTENTSIZE_UNKNOWN)
            throw CacheError("corrupt zstd frame in cache entry");
        std::vector<char> out(static_cast<std::size_t>(raw_size));
        const std::size_t written =
            ZSTD_decompress(out.data(), out.size(), data.data() + 1,
                            data.size() - 1);
        if (ZSTD_isError(written) || written != out.size())
            throw CacheError("zstd decompression failed");
        return out;
#else
        throw CacheError(
            "cache entry is zstd-compressed but sovranx was built without "
            "zstd");
#endif
    }

    throw CacheError("unknown compression envelope flag '" +
                     std::string(1, data[0]) + "'");
}

std::uint64_t KVCacheSerializer::checksum(const std::vector<char>& data) {
    // FNV-1a 64-bit.
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const char c : data) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ull;
    }
    return hash;
}

}  // namespace sovranx::cache
