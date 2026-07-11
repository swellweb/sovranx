#pragma once

#include <ostream>
#include <string>

namespace sovranx {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

// Minimal synchronous logger writing to an injected stream, so it is fully
// testable without touching stdout or the filesystem.
//
// Line format:  "[YYYY-MM-DD HH:MM:SS] [LEVEL] message"
// With timestamps disabled (tests): "[LEVEL] message"
class Logger {
public:
    explicit Logger(std::ostream& out,
                    LogLevel min_level = LogLevel::Info,
                    bool timestamps = true);

    void log(LogLevel level, const std::string& message);

    void debug(const std::string& message) { log(LogLevel::Debug, message); }
    void info(const std::string& message)  { log(LogLevel::Info, message); }
    void warn(const std::string& message)  { log(LogLevel::Warn, message); }
    void error(const std::string& message) { log(LogLevel::Error, message); }

    void set_level(LogLevel min_level) noexcept { min_level_ = min_level; }
    LogLevel level() const noexcept { return min_level_; }

    static const char* level_name(LogLevel level) noexcept;
    // Parses "debug"/"info"/"warn"/"error" (case-insensitive); throws
    // std::invalid_argument otherwise.
    static LogLevel level_from_string(const std::string& name);

private:
    std::ostream& out_;
    LogLevel min_level_;
    bool timestamps_;
};

}  // namespace sovranx
