// Isolated tests for the L2 semantic cache glue: a mock embedder maps
// known prompts to hand-chosen vectors, so hit/miss behaviour at a given
// threshold is exact and the risk (serving a different question's answer)
// is pinned.

#include <catch2/catch_test_macros.hpp>

#include <map>

#include "reame/arca/semantic_cache.hpp"

using reame::arca::Embedder;
using reame::arca::SemanticCache;

namespace {

// Returns a canned vector per known text; unknown text embeds far away.
class MockEmbedder : public Embedder {
public:
    std::map<std::string, std::vector<float>> table;
    std::vector<float> embed(const std::string& text) override {
        auto it = table.find(text);
        if (it != table.end()) return it->second;
        return {0.0f, 0.0f, 1.0f};  // "unknown" direction, far from the rest
    }
};

}  // namespace

TEST_CASE("semantic cache: a paraphrase within threshold hits") {
    MockEmbedder emb;
    // Two ways of asking the same thing embed nearly identically.
    emb.table["What is the capital of France?"] = {1.0f, 0.1f, 0.0f};
    emb.table["France's capital city?"] = {1.0f, 0.0f, 0.0f};

    SemanticCache cache(emb, /*threshold=*/0.9f);
    cache.put("What is the capital of France?", "Paris");

    const auto hit = cache.get("France's capital city?");
    REQUIRE(hit.has_value());
    CHECK(*hit == "Paris");
}

TEST_CASE("semantic cache: an unrelated prompt misses") {
    MockEmbedder emb;
    emb.table["What is the capital of France?"] = {1.0f, 0.0f, 0.0f};
    // "recipe" embeds along the far axis (mock default {0,0,1}).

    SemanticCache cache(emb, 0.9f);
    cache.put("What is the capital of France?", "Paris");

    CHECK_FALSE(cache.get("Give me a recipe for pizza").has_value());
}

TEST_CASE("semantic cache: a nearby-but-different question is gated by threshold") {
    MockEmbedder emb;
    // "capital of France" and "capital of Germany" embed close (both are
    // capital questions) but must NOT share an answer.
    emb.table["capital of France"] = {1.0f, 0.0f, 0.0f};
    emb.table["capital of Germany"] = {0.95f, 0.31f, 0.0f};  // cos ~0.95

    SemanticCache cache(emb, /*threshold=*/0.98f);  // strict
    cache.put("capital of France", "Paris");

    // At a strict threshold the 0.95 neighbour is correctly rejected —
    // Germany does not get France's answer.
    CHECK_FALSE(cache.get("capital of Germany").has_value());

    // Loosen the threshold and the same query now (wrongly) hits — this is
    // exactly the risk the correctness benchmark measures.
    SemanticCache loose(emb, 0.9f);
    loose.put("capital of France", "Paris");
    CHECK(loose.get("capital of Germany").has_value());
}
