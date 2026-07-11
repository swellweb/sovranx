#include "sovranx/utils/logger.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace sovranx {

namespace {

std::string timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

}  // namespace

Logger::Logger(std::ostream& out, LogLevel min_level, bool timestamps)
    : out_(out), min_level_(min_level), timestamps_(timestamps) {}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < min_level_) return;

    std::ostringstream line;
    if (timestamps_) line << '[' << timestamp_now() << "] ";
    line << '[' << level_name(level) << "] " << message << '\n';

    // Single stream insertion keeps lines whole under concurrent use.
    out_ << line.str();
    out_.flush();
}

const char* Logger::level_name(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "UNKNOWN";
}

LogLevel Logger::level_from_string(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "debug") return LogLevel::Debug;
    if (lower == "info")  return LogLevel::Info;
    if (lower == "warn")  return LogLevel::Warn;
    if (lower == "error") return LogLevel::Error;
    throw std::invalid_argument("unknown log level: " + name);
}

}  // namespace sovranx
