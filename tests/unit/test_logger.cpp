// Isolated tests for sovranx::Logger. The output stream is injected
// (ostringstream) and timestamps are disabled for deterministic output.

#include <catch2/catch_test_macros.hpp>

#include <sstream>

#include "sovranx/utils/logger.hpp"

using sovranx::Logger;
using sovranx::LogLevel;

TEST_CASE("formats level and message on one line") {
    std::ostringstream out;
    Logger log(out, LogLevel::Debug, /*timestamps=*/false);

    log.info("engine started");

    CHECK(out.str() == "[INFO] engine started\n");
}

TEST_CASE("each level uses its own tag") {
    std::ostringstream out;
    Logger log(out, LogLevel::Debug, false);

    log.debug("d");
    log.info("i");
    log.warn("w");
    log.error("e");

    CHECK(out.str() ==
          "[DEBUG] d\n"
          "[INFO] i\n"
          "[WARN] w\n"
          "[ERROR] e\n");
}

TEST_CASE("messages below min level are suppressed") {
    std::ostringstream out;
    Logger log(out, LogLevel::Warn, false);

    log.debug("hidden");
    log.info("hidden");
    log.warn("shown");
    log.error("shown");

    CHECK(out.str() == "[WARN] shown\n[ERROR] shown\n");
}

TEST_CASE("set_level changes filtering at runtime") {
    std::ostringstream out;
    Logger log(out, LogLevel::Error, false);

    log.info("hidden");
    log.set_level(LogLevel::Info);
    log.info("shown");

    CHECK(out.str() == "[INFO] shown\n");
}

TEST_CASE("timestamped lines are prefixed with [YYYY-MM-DD HH:MM:SS]") {
    std::ostringstream out;
    Logger log(out, LogLevel::Info, /*timestamps=*/true);

    log.info("x");

    const std::string line = out.str();
    // "[" + 19 timestamp chars + "] " + rest — check shape, not wall time.
    REQUIRE(line.size() > 22);
    CHECK(line[0] == '[');
    CHECK(line[5] == '-');
    CHECK(line[8] == '-');
    CHECK(line[11] == ' ');
    CHECK(line[14] == ':');
    CHECK(line[17] == ':');
    CHECK(line[20] == ']');
    CHECK(line.substr(21) == " [INFO] x\n");
}

TEST_CASE("level_name maps enum to tag") {
    CHECK(std::string(Logger::level_name(LogLevel::Debug)) == "DEBUG");
    CHECK(std::string(Logger::level_name(LogLevel::Info)) == "INFO");
    CHECK(std::string(Logger::level_name(LogLevel::Warn)) == "WARN");
    CHECK(std::string(Logger::level_name(LogLevel::Error)) == "ERROR");
}

TEST_CASE("level_from_string is case-insensitive and validates") {
    CHECK(Logger::level_from_string("debug") == LogLevel::Debug);
    CHECK(Logger::level_from_string("INFO") == LogLevel::Info);
    CHECK(Logger::level_from_string("Warn") == LogLevel::Warn);
    CHECK(Logger::level_from_string("error") == LogLevel::Error);
    CHECK_THROWS_AS(Logger::level_from_string("verbose"), std::invalid_argument);
}
