#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "sovranx/core/llama_backend.hpp"

namespace sovranx::palimpsest {

// The generation memory of an inference server: an append-only log of
// every token stream the server has produced, indexed so that "what came
// after this n-gram, historically?" is answerable in O(1).
//
// This turns the server's own past output into a free speculative draft
// source (retrieval speculation): domain workloads repeat themselves —
// audits, product copy, structured reports — so the continuation that
// followed an n-gram yesterday is an excellent guess for today. Unlike a
// draft model it costs no RAM and no forward passes; unlike prompt-lookup
// it sees across requests, users and restarts.
//
// Persistence model: the token log is a flat file (appended in flushes);
// the n-gram index lives in memory and is rebuilt from the log on open.
// A byte budget caps the log; when exceeded, the oldest half is dropped
// and the index rebuilt (generation history is a cache, not a ledger).
class CorpusIndex {
public:
    struct Config {
        std::filesystem::path directory;
        int ngram = 3;                          // key length for the index
        std::uint64_t max_log_bytes = 64ull << 20;  // 64 MiB of tokens
        int max_sites_per_ngram = 8;            // most recent occurrences kept
    };

    explicit CorpusIndex(const Config& cfg);
    ~CorpusIndex();
    CorpusIndex(const CorpusIndex&) = delete;
    CorpusIndex& operator=(const CorpusIndex&) = delete;

    // Feed one generated stream (prompt + completion, or completion only —
    // whatever the caller wants remembered) into the memory.
    void observe(const std::vector<TokenId>& tokens);

    // Best historical continuation after `tail` (matching its last `ngram`
    // tokens): up to `k` tokens, most recent occurrence first. Empty when
    // the n-gram was never seen. Excludes trivial self-matches within the
    // same observed stream tail.
    std::vector<TokenId> draft(const std::vector<TokenId>& tail, int k) const;

    // Number of indexed n-gram keys (diagnostics).
    std::size_t key_count() const;
    // Total tokens currently in the log (diagnostics).
    std::size_t log_size() const;

    // Flush the in-memory tail of the log to disk (also called on
    // destruction). A crash loses at most the unflushed tail.
    void flush();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace sovranx::palimpsest
