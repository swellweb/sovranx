// The L2 correctness benchmark — the crux of a semantic cache. With a real
// embedding model it measures, across thresholds, how often paraphrases
// hit (good) versus how often a different-but-nearby question wrongly hits
// (the danger). SKIPs without the model. Prints the table and asserts a
// viable operating point exists.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "reame/arca/embedder.hpp"
#include "reame/arca/semantic_cache.hpp"

using reame::arca::make_llama_embedder;
using reame::arca::SemanticCache;

namespace {

std::string embed_model_path() {
    if (const char* e = std::getenv("REAME_EMBED_MODEL")) return e;
    return "models/bge-small-en-v1.5-f16.gguf";
}

bool exists(const std::string& p) { return std::ifstream(p).good(); }

// Stored prompt -> its answer, and the paraphrase that should retrieve it.
struct Para {
    std::string stored, answer, paraphrase;
};
// A stored prompt and a DIFFERENT question that must NOT get its answer.
struct Diff {
    std::string stored, answer, different;
};

}  // namespace

TEST_CASE("[integration] L2 correctness: recall vs false-hit across thresholds",
          "[integration]") {
#ifndef REAME_HAS_LLAMA
    SKIP("built without llama.cpp");
#else
    const auto path = embed_model_path();
    if (!exists(path)) SKIP("embedding model not found: " + path);

    auto emb = make_llama_embedder(path, 4);

    const std::vector<Para> paras = {
        {"What is the capital of France?", "Paris", "France's capital city?"},
        {"How do I reset my password?", "Use the reset link",
         "steps to reset my password"},
        {"What time does the store open?", "9am", "store opening hours"},
        {"Is the product refundable?", "Yes within 30 days",
         "can I get a refund"},
        {"How much does shipping cost?", "5 euros", "what is the shipping fee"},
        {"Where is my order?", "Check tracking", "how do I track my order"},
    };
    const std::vector<Diff> diffs = {
        {"What is the capital of France?", "Paris", "capital of Germany"},
        {"How do I reset my password?", "Use the reset link",
         "how do I change my username"},
        {"How much does shipping cost?", "5 euros",
         "how long does delivery take"},
        {"Is the product refundable?", "Yes within 30 days",
         "what is the warranty period"},
        {"What time does the store open?", "9am", "what time does it close"},
        {"Where is my order?", "Check tracking", "how do I cancel my order"},
    };

    struct Row { float thr, recall, false_hit; };
    std::vector<Row> table;
    for (float thr = 0.60f; thr <= 0.96f; thr += 0.05f) {
        int hit = 0, wrong = 0;
        for (const auto& p : paras) {
            SemanticCache c(*emb, thr);
            c.put(p.stored, p.answer);
            const auto r = c.get(p.paraphrase);
            if (r && *r == p.answer) ++hit;
        }
        for (const auto& d : diffs) {
            SemanticCache c(*emb, thr);
            c.put(d.stored, d.answer);
            if (c.get(d.different).has_value()) ++wrong;  // any hit is wrong
        }
        table.push_back({thr, static_cast<float>(hit) / paras.size(),
                         static_cast<float>(wrong) / diffs.size()});
    }

    for (const auto& r : table)
        WARN("thr " << r.thr << "  recall " << r.recall << "  false-hit "
                    << r.false_hit);

    // A semantic cache is viable only if SOME threshold gives good recall
    // with near-zero wrong answers. If no such point exists, L2 is unsafe
    // for this model and this test must fail loudly.
    bool viable = false;
    for (const auto& r : table)
        if (r.recall >= 0.6f && r.false_hit <= 0.17f) viable = true;
    CHECK(viable);
#endif
}
