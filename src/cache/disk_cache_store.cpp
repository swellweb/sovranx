#include "sovrano/cache/disk_cache_store.hpp"

#include <algorithm>
#include <fstream>

#include "sovrano/cache/cache_serializer.hpp"

namespace sovrano::cache {

namespace fs = std::filesystem;

namespace {

// File layout: u32 key length | key bytes | payload. The stored key guards
// against hash collisions in the filename.
constexpr std::size_t kHeaderLen = sizeof(std::uint32_t);

std::string hex64(std::uint64_t v) {
    static const char* digits = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i, v >>= 4) out[static_cast<std::size_t>(i)] = digits[v & 0xF];
    return out;
}

std::uint64_t entry_size(const std::string& key, std::size_t payload) {
    return kHeaderLen + key.size() + payload;
}

}  // namespace

DiskCacheStore::DiskCacheStore(const Config& cfg) : cfg_(cfg) {
    fs::create_directories(cfg_.directory);

    // Rebuild the index from disk; file mtimes seed the recency order.
    std::vector<std::pair<fs::file_time_type, std::string>> found;
    for (const auto& item : fs::directory_iterator(cfg_.directory)) {
        if (!item.is_regular_file() || item.path().extension() != ".kvc")
            continue;

        std::ifstream in(item.path(), std::ios::binary);
        std::uint32_t key_len = 0;
        in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        if (!in || key_len > 4096) continue;  // ignore unreadable files
        std::string key(key_len, '\0');
        in.read(key.data(), key_len);
        if (!in) continue;

        Entry e;
        e.path = item.path();
        e.size = static_cast<std::uint64_t>(fs::file_size(item.path()));
        total_bytes_ += e.size;
        entries_.emplace(key, std::move(e));
        found.emplace_back(fs::last_write_time(item.path()), key);
    }
    std::sort(found.begin(), found.end());
    for (const auto& [mtime, key] : found)
        entries_.at(key).recency = next_recency_++;
}

bool DiskCacheStore::put(const std::string& key, const std::vector<char>& data) {
    const std::uint64_t incoming = entry_size(key, data.size());
    if (cfg_.max_bytes != 0 && incoming > cfg_.max_bytes) return false;

    remove(key);  // overwrite semantics
    evict_until_fits(incoming);

    const fs::path path =
        cfg_.directory /
        (hex64(KVCacheSerializer::checksum(
             std::vector<char>(key.begin(), key.end()))) +
         ".kvc");
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const auto key_len = static_cast<std::uint32_t>(key.size());
        out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        out.write(key.data(), static_cast<std::streamsize>(key.size()));
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out) {
            fs::remove(path);
            return false;
        }
    }

    Entry e;
    e.path = path;
    e.size = incoming;
    e.recency = next_recency_++;
    total_bytes_ += incoming;
    entries_[key] = std::move(e);
    return true;
}

std::optional<std::vector<char>> DiskCacheStore::get(const std::string& key) {
    const auto it = entries_.find(key);
    if (it == entries_.end()) return std::nullopt;

    std::ifstream in(it->second.path, std::ios::binary);
    std::uint32_t key_len = 0;
    in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
    std::string stored_key(key_len, '\0');
    if (in) in.read(stored_key.data(), key_len);
    if (!in || stored_key != key) {
        remove(key);  // unreadable or filename-hash collision
        return std::nullopt;
    }

    std::vector<char> payload(
        static_cast<std::size_t>(it->second.size - kHeaderLen - key.size()));
    in.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!in) {
        remove(key);
        return std::nullopt;
    }

    it->second.recency = next_recency_++;
    return payload;
}

bool DiskCacheStore::contains(const std::string& key) const {
    return entries_.find(key) != entries_.end();
}

bool DiskCacheStore::remove(const std::string& key) {
    const auto it = entries_.find(key);
    if (it == entries_.end()) return false;
    std::error_code ec;
    fs::remove(it->second.path, ec);
    total_bytes_ -= it->second.size;
    entries_.erase(it);
    return true;
}

void DiskCacheStore::clear() {
    while (!entries_.empty()) remove(entries_.begin()->first);
}

std::uint64_t DiskCacheStore::size_bytes() const { return total_bytes_; }

std::size_t DiskCacheStore::count() const { return entries_.size(); }

void DiskCacheStore::evict_until_fits(std::uint64_t incoming) {
    if (cfg_.max_bytes == 0) return;
    while (!entries_.empty() && total_bytes_ + incoming > cfg_.max_bytes) {
        const auto lru = std::min_element(
            entries_.begin(), entries_.end(), [](const auto& a, const auto& b) {
                return a.second.recency < b.second.recency;
            });
        remove(lru->first);
    }
}

}  // namespace sovrano::cache
