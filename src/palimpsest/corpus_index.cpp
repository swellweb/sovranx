#include "sovranx/palimpsest/corpus_index.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <unordered_map>

namespace sovranx::palimpsest {

namespace fs = std::filesystem;

namespace {

constexpr char kLogName[] = "corpus.log";

// FNV-1a over an n-gram of token ids.
std::uint64_t ngram_key(const TokenId* tokens, int n) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < n; ++i) {
        auto v = static_cast<std::uint32_t>(tokens[i]);
        for (int b = 0; b < 4; ++b) {
            h ^= (v >> (8 * b)) & 0xFF;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

}  // namespace

struct CorpusIndex::Impl {
    Config cfg;
    fs::path log_path;

    // Flat token log with stream boundaries (offsets one-past-the-end of
    // each observed stream, ascending).
    std::vector<TokenId> log;
    std::vector<std::size_t> stream_ends;
    std::size_t flushed_streams = 0;

    // n-gram key -> most recent continuation start offsets (newest last).
    std::unordered_map<std::uint64_t, std::deque<std::uint32_t>> index;

    explicit Impl(const Config& c)
        : cfg(c), log_path(c.directory / kLogName) {
        fs::create_directories(cfg.directory);
        load();
    }

    std::size_t max_tokens() const {
        return static_cast<std::size_t>(cfg.max_log_bytes / sizeof(TokenId));
    }

    std::size_t stream_end_for(std::size_t offset) const {
        // First stream end strictly greater than offset.
        const auto it = std::upper_bound(stream_ends.begin(),
                                         stream_ends.end(), offset);
        return it == stream_ends.end() ? log.size() : *it;
    }

    void index_stream(std::size_t begin, std::size_t end) {
        const auto n = static_cast<std::size_t>(cfg.ngram);
        if (end - begin < n + 1) return;  // need an n-gram plus 1 token
        for (std::size_t i = begin; i + n < end; ++i) {
            const auto key = ngram_key(log.data() + i, cfg.ngram);
            auto& sites = index[key];
            sites.push_back(static_cast<std::uint32_t>(i + n));
            while (sites.size() >
                   static_cast<std::size_t>(cfg.max_sites_per_ngram))
                sites.pop_front();
        }
    }

    void rebuild_index() {
        index.clear();
        std::size_t begin = 0;
        for (const auto end : stream_ends) {
            index_stream(begin, end);
            begin = end;
        }
    }

    void observe(const std::vector<TokenId>& tokens) {
        if (tokens.empty()) return;
        const std::size_t begin = log.size();
        log.insert(log.end(), tokens.begin(), tokens.end());
        stream_ends.push_back(log.size());
        index_stream(begin, log.size());

        if (log.size() > max_tokens()) evict_oldest();
    }

    // Drop the oldest streams until the log fits its budget, then rebuild
    // the index and rewrite the file: history is a cache, not a ledger.
    void evict_oldest() {
        std::size_t drop_streams = 0;
        std::size_t drop_tokens = 0;
        while (drop_streams < stream_ends.size() &&
               log.size() - drop_tokens > max_tokens()) {
            drop_tokens = stream_ends[drop_streams];
            ++drop_streams;
        }
        log.erase(log.begin(), log.begin() + static_cast<long>(drop_tokens));
        stream_ends.erase(stream_ends.begin(),
                          stream_ends.begin() + static_cast<long>(drop_streams));
        for (auto& e : stream_ends) e -= drop_tokens;
        rebuild_index();
        rewrite_file();
    }

    std::vector<TokenId> draft(const std::vector<TokenId>& tail,
                               int k) const {
        const auto n = static_cast<std::size_t>(cfg.ngram);
        if (k <= 0 || tail.size() < n) return {};
        const auto key = ngram_key(tail.data() + (tail.size() - n), cfg.ngram);
        const auto it = index.find(key);
        if (it == index.end()) return {};

        // Newest occurrence first; skip sites with no room to continue
        // (n-gram ended exactly at its stream boundary).
        const auto& sites = it->second;
        for (auto s = sites.rbegin(); s != sites.rend(); ++s) {
            const std::size_t start = *s;
            const std::size_t end = stream_end_for(start > 0 ? start - 1 : 0);
            if (start >= end) continue;
            // Guard against hash collisions: verify the n-gram really
            // matches before trusting the continuation.
            if (!std::equal(tail.end() - static_cast<long>(n), tail.end(),
                            log.begin() + static_cast<long>(start - n)))
                continue;
            const std::size_t take =
                std::min(end - start, static_cast<std::size_t>(k));
            return {log.begin() + static_cast<long>(start),
                    log.begin() + static_cast<long>(start + take)};
        }
        return {};
    }

    // ---- Persistence: framed streams (u32 length + raw tokens) ----------

    void load() {
        std::ifstream in(log_path, std::ios::binary);
        if (!in) return;
        while (true) {
            std::uint32_t len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            if (!in || len == 0 || len > (1u << 26)) break;
            std::vector<TokenId> tokens(len);
            in.read(reinterpret_cast<char*>(tokens.data()),
                    static_cast<std::streamsize>(len * sizeof(TokenId)));
            if (!in) break;
            const std::size_t begin = log.size();
            log.insert(log.end(), tokens.begin(), tokens.end());
            stream_ends.push_back(log.size());
            index_stream(begin, log.size());
        }
        flushed_streams = stream_ends.size();
        if (log.size() > max_tokens()) evict_oldest();
    }

    void flush() {
        if (flushed_streams >= stream_ends.size()) return;
        std::ofstream out(log_path, std::ios::binary | std::ios::app);
        for (std::size_t s = flushed_streams; s < stream_ends.size(); ++s) {
            const std::size_t begin = s == 0 ? 0 : stream_ends[s - 1];
            const std::size_t end = stream_ends[s];
            const auto len = static_cast<std::uint32_t>(end - begin);
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(reinterpret_cast<const char*>(log.data() + begin),
                      static_cast<std::streamsize>(len * sizeof(TokenId)));
        }
        flushed_streams = stream_ends.size();
    }

    void rewrite_file() {
        std::ofstream out(log_path, std::ios::binary | std::ios::trunc);
        std::size_t begin = 0;
        for (const auto end : stream_ends) {
            const auto len = static_cast<std::uint32_t>(end - begin);
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(reinterpret_cast<const char*>(log.data() + begin),
                      static_cast<std::streamsize>(len * sizeof(TokenId)));
            begin = end;
        }
        flushed_streams = stream_ends.size();
    }
};

CorpusIndex::CorpusIndex(const Config& cfg)
    : pimpl_(std::make_unique<Impl>(cfg)) {}

CorpusIndex::~CorpusIndex() {
    try {
        pimpl_->flush();
    } catch (...) {
    }
}

void CorpusIndex::observe(const std::vector<TokenId>& tokens) {
    pimpl_->observe(tokens);
}

std::vector<TokenId> CorpusIndex::draft(const std::vector<TokenId>& tail,
                                        int k) const {
    return pimpl_->draft(tail, k);
}

std::size_t CorpusIndex::key_count() const { return pimpl_->index.size(); }
std::size_t CorpusIndex::log_size() const { return pimpl_->log.size(); }
void CorpusIndex::flush() { pimpl_->flush(); }

}  // namespace sovranx::palimpsest
