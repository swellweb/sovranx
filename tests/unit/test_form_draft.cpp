// Isolated tests for FormDraft — pure string logic, every expectation
// written by hand from the stated rules.

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "sovranx/speculative/form_draft.hpp"

using sovranx::speculative::FormDraft;

TEST_CASE("numbered list: proposes the next number after a completed item") {
    CHECK(FormDraft::propose("Benefits:\n1. Better focus\n") == "2. ");
    CHECK(FormDraft::propose("1. alpha\n2. beta\n") == "3. ");
    // Two-digit increments carry over.
    CHECK(FormDraft::propose("9. nine\n10. ten\n") == "11. ");
    // The ')' list style is proposed in kind.
    CHECK(FormDraft::propose("1) uno\n") == "2) ");
}

TEST_CASE("numbered list: only fires right after the newline") {
    // Item still being written: no proposal.
    CHECK(FormDraft::propose("1. Better focus").empty());
    // Last line is not a list item: no proposal.
    CHECK(FormDraft::propose("1. one\nSome paragraph.\n").empty());
}

TEST_CASE("bullet list: continues the same marker") {
    CHECK(FormDraft::propose("Points:\n- first\n") == "- ");
    CHECK(FormDraft::propose("* first\n* second\n") == "* ");
    // Mixed content after the bullets: no proposal.
    CHECK(FormDraft::propose("- first\nplain text\n").empty());
}

TEST_CASE("no structure, no proposal") {
    CHECK(FormDraft::propose("").empty());
    CHECK(FormDraft::propose("Just prose without any list.").empty());
    CHECK(FormDraft::propose("Prose ending in newline.\n").empty());
}

TEST_CASE("numbering restarts are respected, not summed") {
    // The most recent item wins: after a fresh "1." the bet is "2.".
    CHECK(FormDraft::propose("1. a\n2. b\n\nOther list:\n1. x\n") == "2. ");
}
