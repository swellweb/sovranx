// Isolated tests for the cache module. The backend is MockBackend; the
// disk store runs against a per-test temporary directory (the filesystem
// IS the unit under test there). Checksum expectations are the published
// FNV-1a 64 test vectors, computed independently of the implementation.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../mock/llama_mock.hpp"
#include "sovranx/cache/cache_manager.hpp"
#include "sovranx/cache/cache_serializer.hpp"
#include "sovranx/cache/disk_cache_store.hpp"

namespace fs = std::filesystem;
using sovranx::TokenId;
using sovranx::test::MockBackend;
using sovranx::cache::CacheError;
using sovranx::cache::CacheManager;
using sovranx::cache::DiskCacheStore;
using sovranx::cache::KVCacheSerializer;

namespace {

std::vector<char> bytes(const std::string& s) {
    return std::vector<char>(s.begin(), s.end());
}

// Fresh empty directory per test.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("sovranx-cache-test-" + std::to_string(counter++));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
    static int counter;
};
int TempDir::counter = 0;

}  // namespace

// ---------------------------------------------------------------------------
// KVCacheSerializer
// ---------------------------------------------------------------------------

TEST_CASE("checksum matches the published FNV-1a 64 test vectors") {
    CHECK(KVCacheSerializer::checksum({}) == 0xcbf29ce484222325ull);
    CHECK(KVCacheSerializer::checksum(bytes("a")) == 0xaf63dc4c8601ec8cull);
    CHECK(KVCacheSerializer::checksum(bytes("foobar")) ==
          0x85944171f73967e8ull);
}

TEST_CASE("compress/decompress round-trips arbitrary bytes") {
    const auto data = bytes("sovranx kv cache payload 123");

    const auto packed = KVCacheSerializer::compress(data);
    REQUIRE(!packed.empty());
    CHECK((packed[0] == 'Z' || packed[0] == 'R'));  // envelope flag

    CHECK(KVCacheSerializer::decompress(packed) == data);
}

TEST_CASE("compress round-trips empty input") {
    const std::vector<char> empty;
    CHECK(KVCacheSerializer::decompress(KVCacheSerializer::compress(empty)) ==
          empty);
}

#ifdef SOVRANX_HAS_ZSTD
TEST_CASE("zstd shrinks repetitive payloads") {
    const std::vector<char> data(64 * 1024, 'x');
    const auto packed = KVCacheSerializer::compress(data);
    CHECK(packed[0] == 'Z');
    CHECK(packed.size() < data.size() / 10);
    CHECK(KVCacheSerializer::decompress(packed) == data);
}
#endif

TEST_CASE("decompress rejects an unknown envelope flag") {
    CHECK_THROWS_AS(KVCacheSerializer::decompress(bytes("Qgarbage")),
                    CacheError);
    CHECK_THROWS_AS(KVCacheSerializer::decompress({}), CacheError);
}

TEST_CASE("serialize/deserialize go through the backend state API") {
    MockBackend backend;
    backend.state_data_result = bytes("model-state-blob");

    const auto blob = KVCacheSerializer::serialize(backend);
    CHECK(blob == bytes("model-state-blob"));
    CHECK(backend.state_data_calls == 1);

    MockBackend other;
    KVCacheSerializer::deserialize(other, blob, /*n_past=*/7);
    REQUIRE(other.set_state_calls.size() == 1);
    CHECK(other.set_state_calls[0].first == blob);
    CHECK(other.set_state_calls[0].second == 7);
    CHECK(other.n_past() == 7);
}

// ---------------------------------------------------------------------------
// DiskCacheStore
// ---------------------------------------------------------------------------

TEST_CASE("store: put/get round-trip and basic queries") {
    TempDir dir;
    DiskCacheStore store({dir.path, 0});

    CHECK(store.count() == 0);
    CHECK_FALSE(store.get("missing").has_value());

    REQUIRE(store.put("alpha", bytes("payload-A")));
    CHECK(store.contains("alpha"));
    CHECK(store.count() == 1);
    CHECK(store.size_bytes() > 0);

    const auto back = store.get("alpha");
    REQUIRE(back.has_value());
    CHECK(*back == bytes("payload-A"));

    CHECK(store.remove("alpha"));
    CHECK_FALSE(store.contains("alpha"));
    CHECK_FALSE(store.remove("alpha"));
}

TEST_CASE("store: overwrite replaces the payload") {
    TempDir dir;
    DiskCacheStore store({dir.path, 0});

    store.put("k", bytes("v1"));
    store.put("k", bytes("v2-longer"));

    CHECK(store.count() == 1);
    CHECK(*store.get("k") == bytes("v2-longer"));
}

TEST_CASE("store: entries survive a new instance on the same directory") {
    TempDir dir;
    {
        DiskCacheStore store({dir.path, 0});
        store.put("persisted", bytes("still-here"));
    }
    DiskCacheStore reopened({dir.path, 0});
    CHECK(reopened.count() == 1);
    const auto v = reopened.get("persisted");
    REQUIRE(v.has_value());
    CHECK(*v == bytes("still-here"));
}

TEST_CASE("store: LRU eviction respects recency bumps from get") {
    TempDir dir;
    // Each entry: key header (4 + keylen) + 8-byte payload. Budget fits
    // exactly two entries of this shape.
    const auto payload = bytes("12345678");
    const std::uint64_t one_entry = 4 + 1 + payload.size();
    DiskCacheStore store({dir.path, 2 * one_entry});

    REQUIRE(store.put("a", payload));
    REQUIRE(store.put("b", payload));
    REQUIRE(store.get("a").has_value());  // bump "a": now "b" is the LRU

    REQUIRE(store.put("c", payload));  // must evict "b"

    CHECK(store.contains("a"));
    CHECK_FALSE(store.contains("b"));
    CHECK(store.contains("c"));
    CHECK(store.size_bytes() <= 2 * one_entry);
}

TEST_CASE("store: refuses an entry larger than the whole budget") {
    TempDir dir;
    DiskCacheStore store({dir.path, 8});
    CHECK_FALSE(store.put("huge", bytes("way-more-than-eight-bytes")));
    CHECK(store.count() == 0);
}

TEST_CASE("store: clear removes everything") {
    TempDir dir;
    DiskCacheStore store({dir.path, 0});
    store.put("x", bytes("1"));
    store.put("y", bytes("2"));

    store.clear();

    CHECK(store.count() == 0);
    CHECK(store.size_bytes() == 0);
    CHECK_FALSE(store.get("x").has_value());
}

// ---------------------------------------------------------------------------
// CacheManager
// ---------------------------------------------------------------------------

TEST_CASE("manager: make_key is deterministic and discriminates") {
    const auto k1 = CacheManager::make_key("model-a", {1, 2, 3});
    CHECK(k1 == CacheManager::make_key("model-a", {1, 2, 3}));
    CHECK(k1 != CacheManager::make_key("model-b", {1, 2, 3}));
    CHECK(k1 != CacheManager::make_key("model-a", {1, 2, 4}));
    CHECK(k1 != CacheManager::make_key("model-a", {1, 2}));
}

TEST_CASE("manager: store/load round-trips backend state across instances") {
    TempDir dir;
    MockBackend source;
    source.state_data_result = bytes("kv-state-of-prompt");

    {
        CacheManager mgr({dir.path, 0, /*compress=*/true});
        REQUIRE(mgr.store_state("key1", source, /*n_past=*/42));
        CHECK(mgr.stats().stores == 1);
    }

    CacheManager mgr({dir.path, 0, true});
    MockBackend target;
    REQUIRE(mgr.load_state("key1", target));

    REQUIRE(target.set_state_calls.size() == 1);
    CHECK(target.set_state_calls[0].first == bytes("kv-state-of-prompt"));
    CHECK(target.set_state_calls[0].second == 42);
    CHECK(mgr.stats().hits == 1);
    CHECK(mgr.stats().misses == 0);
}

TEST_CASE("manager: miss on unknown key") {
    TempDir dir;
    CacheManager mgr({dir.path, 0, true});
    MockBackend backend;

    CHECK_FALSE(mgr.load_state("nope", backend));
    CHECK(backend.set_state_calls.empty());
    CHECK(mgr.stats().misses == 1);
}

TEST_CASE("manager: corrupt entries fail the checksum and are evicted") {
    TempDir dir;
    MockBackend source;
    source.state_data_result = bytes("state-to-corrupt");

    CacheManager mgr({dir.path, 0, true});
    REQUIRE(mgr.store_state("victim", source, 5));

    // Flip one byte at the tail of the single .kvc file (payload area).
    fs::path file;
    for (const auto& e : fs::directory_iterator(dir.path)) file = e.path();
    REQUIRE(!file.empty());
    {
        std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-1, std::ios::end);
        const char x = 'X';
        f.write(&x, 1);
    }

    MockBackend target;
    CHECK_FALSE(mgr.load_state("victim", target));
    CHECK(target.set_state_calls.empty());
    // The corrupt entry is gone: directory is empty again.
    CHECK(fs::directory_iterator(dir.path) == fs::directory_iterator{});
}

TEST_CASE("manager: uncompressed mode still round-trips") {
    TempDir dir;
    MockBackend source;
    source.state_data_result = bytes("raw-mode-state");

    CacheManager mgr({dir.path, 0, /*compress=*/false});
    REQUIRE(mgr.store_state("k", source, 3));

    MockBackend target;
    REQUIRE(mgr.load_state("k", target));
    CHECK(target.set_state_calls[0].first == bytes("raw-mode-state"));
}
