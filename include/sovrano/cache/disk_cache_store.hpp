#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sovrano::cache {

// LRU byte store on disk. One file per entry
// (<fnv64-of-key>.kvc: key header + payload); recency is tracked
// in-memory with a monotonic counter (file mtimes seed it on startup, so
// LRU order survives restarts at second granularity).
class DiskCacheStore {
public:
    struct Config {
        std::filesystem::path directory;
        std::uint64_t max_bytes = 0;  // 0 = unlimited
    };

    explicit DiskCacheStore(const Config& cfg);

    // Inserts/overwrites; evicts least-recently-used entries until the
    // budget fits (an entry larger than the whole budget is refused,
    // returning false).
    bool put(const std::string& key, const std::vector<char>& data);

    // Reads and bumps recency. nullopt when missing or unreadable
    // (unreadable/mismatched entries are dropped).
    std::optional<std::vector<char>> get(const std::string& key);

    bool contains(const std::string& key) const;
    bool remove(const std::string& key);
    void clear();

    std::uint64_t size_bytes() const;
    std::size_t count() const;

private:
    struct Entry {
        std::filesystem::path path;
        std::uint64_t size = 0;
        std::uint64_t recency = 0;
    };

    void evict_until_fits(std::uint64_t incoming);

    Config cfg_;
    std::map<std::string, Entry> entries_;
    std::uint64_t total_bytes_ = 0;
    std::uint64_t next_recency_ = 1;
};

}  // namespace sovrano::cache
