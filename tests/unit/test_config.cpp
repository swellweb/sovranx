// Isolated tests for sovranx::Config. No filesystem, no other components:
// parsing is exercised via parse_string / std::istringstream only.

#include <catch2/catch_test_macros.hpp>

#include <sstream>

#include "sovranx/utils/config.hpp"

using sovranx::Config;
using sovranx::ConfigError;

TEST_CASE("parses key=value pairs namespaced by section") {
    const auto cfg = Config::parse_string(
        "[server]\n"
        "port = 8080\n"
        "host = 0.0.0.0\n"
        "[model]\n"
        "path = /models/qwen-30b.gguf\n");

    CHECK(cfg.size() == 3);
    CHECK(cfg.get_string("server.host") == "0.0.0.0");
    CHECK(cfg.get_int("server.port") == 8080);
    CHECK(cfg.get_string("model.path") == "/models/qwen-30b.gguf");
}

TEST_CASE("keys outside any section have no prefix") {
    const auto cfg = Config::parse_string("threads = 16\n");
    CHECK(cfg.get_int("threads") == 16);
}

TEST_CASE("trims whitespace, skips blank lines and comments") {
    const auto cfg = Config::parse_string(
        "\n"
        "# full-line hash comment\n"
        "; full-line semicolon comment\n"
        "  name   =   sovranx  \n"
        "\n");

    CHECK(cfg.size() == 1);
    CHECK(cfg.get_string("name") == "sovranx");
}

TEST_CASE("duplicate key: last value wins") {
    const auto cfg = Config::parse_string("a = 1\na = 2\n");
    CHECK(cfg.get_int("a") == 2);
}

TEST_CASE("typed getters convert values") {
    const auto cfg = Config::parse_string(
        "i = -42\n"
        "d = 0.75\n"
        "t1 = true\nt2 = YES\nt3 = on\nt4 = 1\n"
        "f1 = false\nf2 = No\nf3 = OFF\nf4 = 0\n");

    CHECK(cfg.get_int("i") == -42);
    CHECK(cfg.get_double("d") == 0.75);
    for (const char* k : {"t1", "t2", "t3", "t4"}) CHECK(cfg.get_bool(k));
    for (const char* k : {"f1", "f2", "f3", "f4"}) CHECK_FALSE(cfg.get_bool(k));
}

TEST_CASE("defaulted getters: default only when key missing") {
    const auto cfg = Config::parse_string("present = 7\n");

    CHECK(cfg.get_int("present", 99) == 7);
    CHECK(cfg.get_int("absent", 99) == 99);
    CHECK(cfg.get_string("absent", "fallback") == "fallback");
    CHECK(cfg.get_bool("absent", true) == true);
    CHECK(cfg.get_double("absent", 1.5) == 1.5);
}

TEST_CASE("missing key throws on non-defaulted getter") {
    const auto cfg = Config::parse_string("");
    CHECK_FALSE(cfg.has("nope"));
    CHECK_THROWS_AS(cfg.get_string("nope"), ConfigError);
}

TEST_CASE("invalid conversions throw even with a default") {
    const auto cfg = Config::parse_string(
        "not_int = 12abc\n"
        "not_bool = maybe\n"
        "not_double = 1.2.3\n");

    CHECK_THROWS_AS(cfg.get_int("not_int"), ConfigError);
    CHECK_THROWS_AS(cfg.get_int("not_int", 5), ConfigError);
    CHECK_THROWS_AS(cfg.get_bool("not_bool"), ConfigError);
    CHECK_THROWS_AS(cfg.get_double("not_double"), ConfigError);
}

TEST_CASE("malformed lines report 1-based line number") {
    try {
        Config::parse_string("ok = 1\nthis line has no equals\n");
        FAIL("expected ConfigError");
    } catch (const ConfigError& e) {
        CHECK(e.line() == 2);
    }

    try {
        Config::parse_string("[unclosed\n");
        FAIL("expected ConfigError");
    } catch (const ConfigError& e) {
        CHECK(e.line() == 1);
    }

    // Empty key is malformed too.
    CHECK_THROWS_AS(Config::parse_string("= value\n"), ConfigError);
}

TEST_CASE("value may contain '=' characters") {
    const auto cfg = Config::parse_string("expr = a=b=c\n");
    CHECK(cfg.get_string("expr") == "a=b=c");
}

TEST_CASE("parse from stream matches parse_string") {
    std::istringstream in("[s]\nk = v\n");
    const auto cfg = Config::parse(in);
    CHECK(cfg.get_string("s.k") == "v");
}
