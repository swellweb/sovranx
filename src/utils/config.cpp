#include "sovranx/utils/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace sovranx {

namespace {

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

Config Config::parse(std::istream& in) {
    Config cfg;
    std::string section;
    std::string line;
    std::size_t line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;

        if (t[0] == '[') {
            if (t.back() != ']')
                throw ConfigError("unterminated section header: " + t, line_no);
            section = trim(t.substr(1, t.size() - 2));
            if (section.empty())
                throw ConfigError("empty section name", line_no);
            continue;
        }

        const auto eq = t.find('=');
        if (eq == std::string::npos)
            throw ConfigError("expected 'key = value': " + t, line_no);

        const std::string key = trim(t.substr(0, eq));
        if (key.empty())
            throw ConfigError("empty key", line_no);

        const std::string value = trim(t.substr(eq + 1));
        const std::string full_key = section.empty() ? key : section + "." + key;
        cfg.values_[full_key] = value;
    }
    return cfg;
}

Config Config::parse_string(const std::string& text) {
    std::istringstream in(text);
    return parse(in);
}

Config Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw ConfigError("cannot open config file: " + path);
    return parse(in);
}

bool Config::has(const std::string& key) const {
    return values_.find(key) != values_.end();
}

std::string Config::get_string(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end())
        throw ConfigError("missing config key: " + key);
    return it->second;
}

std::int64_t Config::get_int(const std::string& key) const {
    const std::string raw = get_string(key);
    try {
        std::size_t pos = 0;
        const long long v = std::stoll(raw, &pos);
        if (pos != raw.size())
            throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw ConfigError("key '" + key + "' is not an integer: '" + raw + "'");
    }
}

double Config::get_double(const std::string& key) const {
    const std::string raw = get_string(key);
    try {
        std::size_t pos = 0;
        const double v = std::stod(raw, &pos);
        if (pos != raw.size())
            throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw ConfigError("key '" + key + "' is not a number: '" + raw + "'");
    }
}

bool Config::get_bool(const std::string& key) const {
    const std::string raw = to_lower(get_string(key));
    if (raw == "true" || raw == "yes" || raw == "on" || raw == "1") return true;
    if (raw == "false" || raw == "no" || raw == "off" || raw == "0") return false;
    throw ConfigError("key '" + key + "' is not a boolean: '" + get_string(key) + "'");
}

std::string Config::get_string(const std::string& key, const std::string& def) const {
    return has(key) ? get_string(key) : def;
}

std::int64_t Config::get_int(const std::string& key, std::int64_t def) const {
    return has(key) ? get_int(key) : def;
}

double Config::get_double(const std::string& key, double def) const {
    return has(key) ? get_double(key) : def;
}

bool Config::get_bool(const std::string& key, bool def) const {
    return has(key) ? get_bool(key) : def;
}

}  // namespace sovranx
