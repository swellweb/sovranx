#include "sovranx/core/conclave.hpp"

#include <algorithm>
#include <cctype>
#include <set>

namespace sovranx::core {

namespace {

std::set<std::string> word_set(const std::string& text) {
    std::set<std::string> words;
    std::string cur;
    for (const char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            cur.push_back(
                static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        } else if (!cur.empty()) {
            words.insert(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) words.insert(cur);
    return words;
}

}  // namespace

double answer_similarity(const std::string& a, const std::string& b) {
    const auto wa = word_set(a);
    const auto wb = word_set(b);
    if (wa.empty() || wb.empty()) return 0.0;

    std::size_t common = 0;
    for (const auto& w : wa)
        if (wb.count(w) != 0) ++common;
    const std::size_t uni = wa.size() + wb.size() - common;
    return uni == 0 ? 0.0 : static_cast<double>(common) / static_cast<double>(uni);
}

std::string final_number(const std::string& text) {
    std::string last;
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        const bool digit = std::isdigit(static_cast<unsigned char>(text[i])) != 0;
        // A '-' is a sign only before a digit and not glued to a word
        // ("twenty-two" has no minus; "-3.5" does).
        const bool sign =
            text[i] == '-' && i + 1 < n &&
            std::isdigit(static_cast<unsigned char>(text[i + 1])) != 0 &&
            (i == 0 ||
             !std::isalnum(static_cast<unsigned char>(text[i - 1])));
        if (!digit && !sign) {
            ++i;
            continue;
        }
        std::string cur;
        if (sign) {
            cur.push_back('-');
            ++i;
        }
        bool seen_dot = false;
        while (i < n) {
            const char ch = text[i];
            if (std::isdigit(static_cast<unsigned char>(ch))) {
                cur.push_back(ch);
                ++i;
            } else if (ch == ',' && i + 1 < n &&
                       std::isdigit(static_cast<unsigned char>(text[i + 1]))) {
                ++i;  // thousands separator: skip
            } else if (ch == '.' && !seen_dot && i + 1 < n &&
                       std::isdigit(static_cast<unsigned char>(text[i + 1]))) {
                seen_dot = true;
                cur.push_back('.');
                ++i;
            } else {
                break;
            }
        }
        last = std::move(cur);
    }
    return last;
}

std::size_t elect_numeric(const std::vector<std::string>& candidates) {
    if (candidates.size() < 2) return 0;

    std::vector<std::string> numbers(candidates.size());
    std::size_t numeric = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        numbers[i] = final_number(candidates[i]);
        if (!numbers[i].empty()) ++numeric;
    }
    // The numeric vote is meaningful only when at least half the conclave
    // concluded with a number.
    if (numeric * 2 >= candidates.size()) {
        std::size_t best = candidates.size();
        std::size_t best_votes = 1;  // a majority needs at least 2 votes
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (numbers[i].empty()) continue;
            std::size_t votes = 0;
            for (std::size_t j = 0; j < candidates.size(); ++j)
                if (numbers[j] == numbers[i]) ++votes;
            if (votes > best_votes) {
                best_votes = votes;
                best = i;  // first carrier: i only decreases never revisits
            }
        }
        if (best < candidates.size()) return best;
    }
    return elect(candidates);
}

GenerationConfig conclave_attempt(const GenerationConfig& gen, int i) {
    if (i == 0) return gen;
    auto g = gen;
    g.seed = gen.seed + i;
    if (g.temperature < 0.7f) g.temperature = 0.7f;
    return g;
}

std::size_t elect(const std::vector<std::string>& candidates) {
    if (candidates.size() < 2) return 0;

    std::size_t best = 0;
    double best_score = -1.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        double score = 0.0;
        for (std::size_t j = 0; j < candidates.size(); ++j)
            if (i != j) score += answer_similarity(candidates[i], candidates[j]);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

}  // namespace sovranx::core
