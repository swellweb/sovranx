#pragma once

#include <optional>
#include <string>

#include "reame/arca/embedder.hpp"
#include "reame/arca/semantic_index.hpp"

namespace reame::arca {

// The L2 semantic cache: embed a prompt, look for a stored answer whose
// prompt embeds nearby. get() returns a cached value only when the closest
// stored prompt clears `threshold` cosine similarity — the knob that trades
// hit rate against serving the answer to a *different* question. A high
// threshold is the safe default; the correctness benchmark sets it.
//
// This is the retrieve stage. If the benchmark shows cosine alone is too
// loose, a cross-encoder rerank (the model's RANK pooling) can gate the
// top candidate — but only if the numbers demand it.
class SemanticCache {
public:
    SemanticCache(Embedder& embedder, float threshold)
        : embedder_(embedder), threshold_(threshold) {}

    std::optional<std::string> get(const std::string& prompt) {
        return index_.nearest(embedder_.embed(prompt), threshold_);
    }

    void put(const std::string& prompt, const std::string& value) {
        index_.add(embedder_.embed(prompt), value);
    }

    std::size_t size() const { return index_.size(); }

private:
    Embedder& embedder_;
    float threshold_;
    SemanticIndex index_;
};

}  // namespace reame::arca
