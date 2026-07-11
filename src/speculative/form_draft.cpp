#include "sovranx/speculative/form_draft.hpp"

#include <cctype>

namespace sovranx::speculative {

namespace {

// The line just before the final newline of `tail` (empty when the tail
// does not end with a newline).
std::string last_completed_line(const std::string& tail) {
    if (tail.empty() || tail.back() != '\n') return {};
    const auto end = tail.size() - 1;
    const auto begin = tail.rfind('\n', end - (end > 0 ? 1 : 0));
    return tail.substr(begin == std::string::npos ? 0 : begin + 1,
                       end - (begin == std::string::npos ? 0 : begin + 1));
}

// Parses "<number><'.'|')'> ..." — returns the number and its separator,
// or 0 when the line is not a list item.
long parse_numbered_item(const std::string& line, char& sep) {
    std::size_t i = 0;
    while (i < line.size() &&
           std::isdigit(static_cast<unsigned char>(line[i])))
        ++i;
    if (i == 0 || i > 9) return 0;
    if (i >= line.size() || (line[i] != '.' && line[i] != ')')) return 0;
    if (i + 1 >= line.size() || line[i + 1] != ' ') return 0;
    sep = line[i];
    return std::stol(line.substr(0, i));
}

bool is_bullet_item(const std::string& line, char& marker) {
    if (line.size() < 2) return false;
    if ((line[0] == '-' || line[0] == '*') && line[1] == ' ') {
        marker = line[0];
        return true;
    }
    return false;
}

}  // namespace

std::string FormDraft::propose(const std::string& tail) {
    const std::string line = last_completed_line(tail);
    if (line.empty()) return {};

    char sep = '.';
    if (const long n = parse_numbered_item(line, sep); n > 0)
        return std::to_string(n + 1) + sep + " ";

    char marker = '-';
    if (is_bullet_item(line, marker)) return std::string(1, marker) + " ";

    return {};
}

}  // namespace sovranx::speculative
