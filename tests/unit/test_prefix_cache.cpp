// Isolated tests for PrefixCache: MockBackend + per-test temp directory.
// Block size 4 keeps every expectation hand-computable.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <numeric>
#include <vector>

#include "../mock/llama_mock.hpp"
#include "sovranx/cache/prefix_cache.hpp"

namespace fs = std::filesystem;
using sovranx::TokenId;
using sovranx::test::MockBackend;
using sovranx::cache::CacheManager;
using sovranx::cache::PrefixCache;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("sovranx-prefix-test-" + std::to_string(counter++));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
    static int counter;
};
int TempDir::counter = 0;

std::vector<TokenId> iota_tokens(int n, TokenId start = 0) {
    std::vector<TokenId> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), start);
    return v;
}

}  // namespace

TEST_CASE("boundaries: block multiples plus the full prefix") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache cache(mgr, "m", /*block_tokens=*/4);

    CHECK(cache.boundaries(0).empty());
    CHECK(cache.boundaries(3) == std::vector<std::uint32_t>{3});
    CHECK(cache.boundaries(4) == std::vector<std::uint32_t>{4});
    CHECK(cache.boundaries(10) == std::vector<std::uint32_t>{4, 8, 10});
}

TEST_CASE("keys: deterministic chain, sensitive to content, length and model") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache a(mgr, "model-a", 4);
    PrefixCache b(mgr, "model-b", 4);

    const auto p = iota_tokens(8);
    CHECK(a.key_for(p, 8) == a.key_for(p, 8));
    CHECK(a.key_for(p, 4) != a.key_for(p, 8));
    CHECK(a.key_for(p, 8) != b.key_for(p, 8));

    auto q = p;
    q[0] = 99;
    CHECK(a.key_for(p, 8) != a.key_for(q, 8));
    // Same first block -> same key at that boundary even if tails differ.
    auto r = p;
    r[7] = 77;
    CHECK(a.key_for(p, 4) == a.key_for(r, 4));
}

TEST_CASE("cold prefill decodes block-wise and snapshots every boundary") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache cache(mgr, "m", 4);
    MockBackend backend;

    const auto prefix = iota_tokens(10);
    const auto restored = cache.prefill(prefix, backend);

    CHECK(restored == 0);
    CHECK(backend.reset_calls == 1);
    REQUIRE(backend.decode_append_calls.size() == 3);
    CHECK(backend.decode_append_calls[0] == iota_tokens(4));
    CHECK(backend.decode_append_calls[1] == iota_tokens(4, 4));
    CHECK(backend.decode_append_calls[2] == std::vector<TokenId>{8, 9});
    CHECK(backend.n_past() == 10);
    CHECK(mgr.stats().stores == 3);  // boundaries 4, 8, 10
}

TEST_CASE("warm prefill: exact repeat restores everything, decodes nothing") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache cache(mgr, "m", 4);

    const auto prefix = iota_tokens(10);
    MockBackend cold;
    cold.state_data_result = {'S'};
    cache.prefill(prefix, cold);

    MockBackend warm;
    const auto restored = cache.prefill(prefix, warm);

    CHECK(restored == 10);
    REQUIRE(warm.set_state_calls.size() == 1);
    CHECK(warm.set_state_calls[0].second == 10);
    CHECK(warm.decode_append_calls.empty());
    CHECK(warm.n_past() == 10);
}

TEST_CASE("shared prefix: a different prompt reuses the common boundary") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache cache(mgr, "m", 4);

    // First prompt: {0..9}. Second: same first 8 tokens, different tail.
    MockBackend first;
    cache.prefill(iota_tokens(10), first);

    auto second_prefix = iota_tokens(8);
    second_prefix.push_back(20);
    second_prefix.push_back(21);

    MockBackend second;
    const auto restored = cache.prefill(second_prefix, second);

    // Boundary 8 is shared; only the new tail is decoded.
    CHECK(restored == 8);
    REQUIRE(second.set_state_calls.size() == 1);
    CHECK(second.set_state_calls[0].second == 8);
    REQUIRE(second.decode_append_calls.size() == 1);
    CHECK(second.decode_append_calls[0] == std::vector<TokenId>{20, 21});
    CHECK(second.n_past() == 10);
    // The new full-prefix boundary was snapshotted for next time (4 and 8
    // were already present and deduplicated).
    CHECK(mgr.stats().stores == 4);
}

TEST_CASE("identical blocks are stored once (dedup across prompts)") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache cache(mgr, "m", 4);

    MockBackend a, b;
    cache.prefill(iota_tokens(8), a);   // stores boundaries 4, 8
    cache.prefill(iota_tokens(8), b);   // exact hit: nothing new stored

    CHECK(mgr.stats().stores == 2);
    CHECK(mgr.stats().hits == 1);
}

TEST_CASE("a different model tag never matches") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache one(mgr, "model-one", 4);
    PrefixCache two(mgr, "model-two", 4);

    MockBackend a, b;
    one.prefill(iota_tokens(8), a);
    const auto restored = two.prefill(iota_tokens(8), b);

    CHECK(restored == 0);
    CHECK_FALSE(b.decode_append_calls.empty());
}

TEST_CASE("empty prefix resets and does nothing else") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, false});
    PrefixCache cache(mgr, "m", 4);
    MockBackend backend;

    CHECK(cache.prefill({}, backend) == 0);
    CHECK(backend.reset_calls == 1);
    CHECK(backend.decode_append_calls.empty());
    CHECK(mgr.stats().stores == 0);
}
