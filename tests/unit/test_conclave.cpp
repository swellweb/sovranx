// Isolated tests for the Conclave election — pure string logic, every
// expectation derived by hand from the Jaccard/medoid rules.

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_test_macros.hpp>

#include "sovranx/core/conclave.hpp"

using Catch::Matchers::WithinAbs;
using sovranx::core::answer_similarity;
using sovranx::core::elect;
using sovranx::core::elect_numeric;
using sovranx::core::final_number;

TEST_CASE("similarity: identical answers score 1, disjoint score 0") {
    CHECK_THAT(answer_similarity("the answer is rome", "the answer is rome"),
               WithinAbs(1.0, 1e-9));
    CHECK_THAT(answer_similarity("alpha beta", "gamma delta"),
               WithinAbs(0.0, 1e-9));
}

TEST_CASE("similarity: Jaccard on word sets, case-insensitive") {
    // {rome, is, big} vs {rome, is, old}: intersection 2, union 4 -> 0.5.
    CHECK_THAT(answer_similarity("Rome is big", "rome is old"),
               WithinAbs(0.5, 1e-9));
    // Repeated words count once: {a, b} vs {a, b} -> 1.
    CHECK_THAT(answer_similarity("a a b", "b a"), WithinAbs(1.0, 1e-9));
}

TEST_CASE("similarity: empty answers") {
    CHECK_THAT(answer_similarity("", ""), WithinAbs(0.0, 1e-9));
    CHECK_THAT(answer_similarity("x", ""), WithinAbs(0.0, 1e-9));
}

TEST_CASE("elect: the recurring answer beats the outliers") {
    // Two candidates say Rome, one says Paris: a Rome answer wins.
    const auto i = elect({"the capital is rome",
                          "capital of italy is rome",
                          "the capital is paris"});
    CHECK((i == 0 || i == 1));
}

TEST_CASE("elect: exact majority elects a member of the majority") {
    const auto i = elect({"42", "41", "42", "42", "17"});
    // All "42" answers are identical; medoid must be one of them.
    CHECK((i == 0 || i == 2 || i == 3));
}

TEST_CASE("elect: degenerate inputs") {
    CHECK(elect({}) == 0);
    CHECK(elect({"only one"}) == 0);
    // All-different answers: first wins the tie.
    CHECK(elect({"alpha", "beta", "gamma"}) == 0);
}

TEST_CASE("final_number: last number in the text, sign and decimals kept") {
    CHECK(final_number("12 x 13 = 156, minus 50 gives 106.") == "106");
    CHECK(final_number("the temperature dropped to -3.5 degrees") == "-3.5");
    // Thousands separators are stripped: 1,024 is one number.
    CHECK(final_number("total: 1,024 items") == "1024");
    CHECK(final_number("no digits here") == "");
    CHECK(final_number("") == "");
    // A hyphen between words is not a minus sign.
    CHECK(final_number("twenty-two is 22") == "22");
}

TEST_CASE("elect_numeric: exact majority on final numbers wins") {
    // Three candidates conclude 106 (verbose or terse alike), two err.
    // Majority number 106 -> first carrier, index 0.
    CHECK(elect_numeric({"step by step... the answer is 106",
                         "I think it's 96",
                         "12*13=156; 156-50=106",
                         "77",
                         "so the result equals 106"}) == 0);
}

TEST_CASE("elect_numeric: majority beats the medoid's verbosity bias") {
    // The two wrong answers share many words (high mutual Jaccard) and
    // would drag a text medoid; the numeric vote ignores the prose.
    CHECK(elect_numeric({"the final answer to the question is 42",
                         "the final answer to the question is 41",
                         "42",
                         "result: 42"}) == 0);
}

TEST_CASE("elect_numeric: falls back to medoid without a numeric majority") {
    // Only 1 of 3 ends with a number: text election applies, and the two
    // Rome answers outvote Paris.
    const auto i = elect_numeric({"the capital is rome",
                                  "capital of italy is rome",
                                  "paris, founded in 250"});
    CHECK((i == 0 || i == 1));
    // Full numeric tie (1 vote each): medoid fallback, first wins.
    CHECK(elect_numeric({"1", "2", "3"}) == 0);
}

TEST_CASE("elect_numeric: degenerate inputs") {
    CHECK(elect_numeric({}) == 0);
    CHECK(elect_numeric({"only 7"}) == 0);
}

TEST_CASE("conclave_attempt: attempt 0 is the untouched anchor") {
    sovranx::core::GenerationConfig g;
    g.temperature = 0.0f;  // greedy caller
    g.seed = 42;
    const auto a0 = sovranx::core::conclave_attempt(g, 0);
    CHECK(a0.temperature == 0.0f);
    CHECK(a0.seed == 42);
}

TEST_CASE("conclave_attempt: explorers shift seed and heat up") {
    sovranx::core::GenerationConfig g;
    g.temperature = 0.0f;
    g.seed = 42;
    const auto a3 = sovranx::core::conclave_attempt(g, 3);
    CHECK(a3.seed == 45);
    CHECK_THAT(a3.temperature, WithinAbs(0.7, 1e-6));
    // A caller already hotter than 0.7 keeps its own temperature.
    g.temperature = 0.9f;
    CHECK_THAT(sovranx::core::conclave_attempt(g, 1).temperature,
               WithinAbs(0.9, 1e-6));
}
