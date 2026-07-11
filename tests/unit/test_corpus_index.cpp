// Isolated tests for palimpsest::CorpusIndex — pure token sequences over a
// per-test temp directory; every expectation derived by hand from the
// "most recent occurrence wins" rule.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <vector>

#include "sovranx/palimpsest/corpus_index.hpp"

namespace fs = std::filesystem;
using sovranx::TokenId;
using sovranx::palimpsest::CorpusIndex;

namespace {

using Tokens = std::vector<TokenId>;

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("sovranx-corpus-test-" + std::to_string(counter++));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
    static int counter;
};
int TempDir::counter = 0;

CorpusIndex::Config cfg(const fs::path& dir, int ngram = 2) {
    CorpusIndex::Config c;
    c.directory = dir;
    c.ngram = ngram;
    return c;
}

}  // namespace

TEST_CASE("draft returns the continuation seen after the n-gram") {
    TempDir dir;
    CorpusIndex corpus(cfg(dir.path));

    // Stream: after {2,3} came {4,5,6}.
    corpus.observe({1, 2, 3, 4, 5, 6});

    CHECK(corpus.draft({9, 2, 3}, 2) == Tokens{4, 5});
    CHECK(corpus.draft({2, 3}, 8) == Tokens{4, 5, 6});
    // Unknown n-gram: nothing.
    CHECK(corpus.draft({7, 8}, 4).empty());
    // Tail shorter than the n-gram: nothing.
    CHECK(corpus.draft({3}, 4).empty());
}

TEST_CASE("the most recent occurrence wins across observations") {
    TempDir dir;
    CorpusIndex corpus(cfg(dir.path));

    corpus.observe({2, 3, 10, 11});   // older: {2,3} -> {10,11}
    corpus.observe({2, 3, 20, 21});   // newer: {2,3} -> {20,21}

    CHECK(corpus.draft({2, 3}, 2) == Tokens{20, 21});
}

TEST_CASE("streams do not bleed into each other") {
    TempDir dir;
    CorpusIndex corpus(cfg(dir.path));

    // {8,9} ends the first stream: its "continuation" must NOT be the
    // beginning of the second stream.
    corpus.observe({7, 8, 9});
    corpus.observe({30, 31, 32});

    CHECK(corpus.draft({8, 9}, 4).empty());
}

TEST_CASE("memory survives a reopen (rebuilt from the on-disk log)") {
    TempDir dir;
    {
        CorpusIndex corpus(cfg(dir.path));
        corpus.observe({1, 2, 3, 4, 5});
    }  // destructor flushes

    CorpusIndex reopened(cfg(dir.path));
    CHECK(reopened.draft({2, 3}, 2) == Tokens{4, 5});
    CHECK(reopened.log_size() == 5);
}

TEST_CASE("the byte budget drops the oldest history, keeps the newest") {
    TempDir dir;
    auto c = cfg(dir.path);
    // Budget of ~24 tokens (4 bytes each): two 10-token streams fit, the
    // third pushes the first out.
    c.max_log_bytes = 24 * 4;
    CorpusIndex corpus(c);

    corpus.observe({1, 2, 101, 102, 103, 104, 105, 106, 107, 108});
    corpus.observe({3, 4, 201, 202, 203, 204, 205, 206, 207, 208});
    corpus.observe({5, 6, 301, 302, 303, 304, 305, 306, 307, 308});

    // Newest still answers; oldest was evicted.
    CHECK(corpus.draft({5, 6}, 2) == Tokens{301, 302});
    CHECK(corpus.draft({1, 2}, 2).empty());
    CHECK(corpus.log_size() <= 24);
}

TEST_CASE("draft never returns more than k tokens and k<=0 yields nothing") {
    TempDir dir;
    CorpusIndex corpus(cfg(dir.path));
    corpus.observe({1, 2, 3, 4, 5, 6, 7});

    CHECK(corpus.draft({1, 2}, 1) == Tokens{3});
    CHECK(corpus.draft({1, 2}, 0).empty());
    CHECK(corpus.draft({1, 2}, -2).empty());
}

TEST_CASE("key_count reflects distinct n-grams that have a continuation") {
    TempDir dir;
    CorpusIndex corpus(cfg(dir.path));

    // Only n-grams followed by at least one token become keys: {1,2}->3.
    // ({2,3} ends the stream: nothing to retrieve, not indexed.)
    corpus.observe({1, 2, 3});
    CHECK(corpus.key_count() == 1);

    corpus.observe({4, 5, 6, 7});  // adds {4,5}->6 and {5,6}->7
    CHECK(corpus.key_count() == 3);

    corpus.observe({1, 2, 9});  // {1,2} already a key: no new entry
    CHECK(corpus.key_count() == 3);
}
