#pragma once

#include <cstdint>
#include <istream>
#include <map>
#include <stdexcept>
#include <string>

namespace sovranx {

// Parse/lookup error. Carries the 1-based line number for parse errors
// (0 when the error is not tied to a specific line, e.g. missing key).
class ConfigError : public std::runtime_error {
public:
    ConfigError(const std::string& message, std::size_t line = 0)
        : std::runtime_error(message), line_(line) {}

    std::size_t line() const noexcept { return line_; }

private:
    std::size_t line_;
};

// INI-style configuration:
//   - "key = value" pairs, keys namespaced by section as "section.key"
//   - "[section]" headers
//   - "#" and ";" start a comment (full line only)
//   - whitespace around keys/values is trimmed
//   - a duplicate key overwrites the previous value
class Config {
public:
    static Config parse(std::istream& in);
    static Config parse_string(const std::string& text);
    static Config load(const std::string& path);

    bool has(const std::string& key) const;
    std::size_t size() const noexcept { return values_.size(); }

    // Throwing getters: ConfigError if the key is missing or the value
    // cannot be converted.
    std::string get_string(const std::string& key) const;
    std::int64_t get_int(const std::string& key) const;
    double get_double(const std::string& key) const;
    bool get_bool(const std::string& key) const;

    // Defaulted getters: return `def` if the key is missing; still throw
    // on conversion failure (a present-but-invalid value is a config bug).
    std::string get_string(const std::string& key, const std::string& def) const;
    std::int64_t get_int(const std::string& key, std::int64_t def) const;
    double get_double(const std::string& key, double def) const;
    bool get_bool(const std::string& key, bool def) const;

private:
    std::map<std::string, std::string> values_;
};

}  // namespace sovranx
