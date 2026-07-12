// Isolated tests for the zero-config layer: pure alias resolution, thread
// heuristic and config generation. No downloads, no filesystem.

#include <catch2/catch_test_macros.hpp>

#include "reame/core/autoconfig.hpp"

using reame::core::auto_config;
using reame::core::auto_threads;
using reame::core::model_catalog;
using reame::core::resolve_model;

TEST_CASE("catalog is non-empty and every entry is well-formed") {
    const auto& cat = model_catalog();
    REQUIRE(!cat.empty());
    for (const auto& m : cat) {
        CHECK(!m.name.empty());
        CHECK(m.url.rfind("https://", 0) == 0);
        CHECK(!m.filename.empty());
        CHECK(m.default_ctx > 0);
    }
}

TEST_CASE("resolve_model matches an exact alias") {
    const auto m = resolve_model(model_catalog().front().name);
    REQUIRE(m.has_value());
    CHECK(m->name == model_catalog().front().name);
}

TEST_CASE("resolve_model accepts an unambiguous prefix, case-insensitively") {
    // The catalog contains qwen2.5-* entries; a bare prefix that hits
    // exactly one resolves, an ambiguous one does not.
    const auto tiny = resolve_model("TINY");  // TinyLlama, unique prefix
    CHECK(tiny.has_value());

    // A token matching nothing is a miss (caller treats it as a path).
    CHECK_FALSE(resolve_model("definitely-not-a-model").has_value());
}

TEST_CASE("catalog carries the MoE pick with 1B-active economics") {
    // OLMoE 7B-A1B: 7B-class extraction quality at 1B decode cost —
    // measured 8/8 needles and 17.8 tok/s on a 2-core ARM free tier
    // (dense 7B: 3.3 tok/s). The catalog must offer it.
    const auto m = resolve_model("olmoe");
    REQUIRE(m.has_value());
    CHECK(m->filename == "olmoe-1b-7b-instruct-q4_k_m.gguf");
    CHECK(m->url.find("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf") !=
          std::string::npos);
    CHECK(m->note.find("MoE") != std::string::npos);
}

TEST_CASE("resolve_model returns nothing for an ambiguous prefix") {
    // "qwen" matches several catalog entries -> ambiguous -> nullopt.
    CHECK_FALSE(resolve_model("qwen").has_value());
}

TEST_CASE("catalog carries the judgment-capable pick (Qwen3.5-9B)") {
    // Measured on a real webpage audit (M3 Pro): correct alt-text
    // assessment, spotted an empty H2, no invented defects — where 1.5B
    // and OLMoE fabricated findings and a 30B-A3B miscounted. 9B is the
    // smallest model we measured doing judgment tasks right.
    const auto m = resolve_model("qwen3.5-9b");
    REQUIRE(m.has_value());
    CHECK(m->filename == "qwen3.5-9b-q4_k_m.gguf");
    CHECK(m->url.find("Qwen3.5-9B-GGUF") != std::string::npos);
    CHECK(m->default_ctx >= 8192);
    CHECK(m->note.find("9B") != std::string::npos);
}

TEST_CASE("auto_threads leaves headroom on big boxes, uses all on small ones") {
    CHECK(auto_threads(0) == 1);   // unknown -> safe minimum
    CHECK(auto_threads(1) == 1);
    CHECK(auto_threads(2) == 2);
    CHECK(auto_threads(4) == 4);   // small: use them all
    CHECK(auto_threads(8) == 7);   // leave one for the OS/UI
    CHECK(auto_threads(16) == 15);
}

TEST_CASE("auto_config produces a valid, ready-to-run config") {
    const auto cfg = auto_config("/models/qwen.gguf", "/home/user", 8);

    CHECK(cfg.model_path == "/models/qwen.gguf");
    CHECK(cfg.n_threads == 7);
    CHECK(cfg.n_ctx > 0);
    CHECK(cfg.kv_cache_type == "q8_0");      // save RAM by default
    CHECK(cfg.use_prompt_lookup == true);    // free speedup, on by default
    // Cache lands under the user's home so nothing needs creating by hand.
    CHECK(cfg.cache_dir == "/home/user/.reame/cache");
}

TEST_CASE("auto_config honours an explicit context override") {
    const auto cfg = auto_config("/m.gguf", "/home/u", 4, /*ctx=*/8192);
    CHECK(cfg.n_ctx == 8192);
}

TEST_CASE("auto_config falls back to a safe cache dir when home is empty") {
    const auto cfg = auto_config("/m.gguf", "", 4);
    // No home known -> cache next to the model dir, never the filesystem root.
    CHECK(cfg.cache_dir != "/.reame/cache");
    CHECK(!cfg.cache_dir.empty());
}
