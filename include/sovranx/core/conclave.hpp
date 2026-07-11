#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sovranx/core/engine.hpp"

namespace sovranx::core {

// The Conclave: N candidate answers deliberate, one emerges.
//
// On memory-bound CPUs the interleaved scheduler makes N parallel
// generations cost barely more than one (the weights are read once per
// step for all sequences). The Conclave spends that free parallelism on
// N attempts at the SAME prompt (different seeds) and elects the answer
// the candidates agree on most — consensus is a strong truth signal:
// wrong answers scatter, the right one recurs. This is how a small model
// out-answers a bigger one at comparable wall-time.

// Similarity between two answers in [0,1]: Jaccard overlap of their
// lowercased word sets. Pure and hand-checkable.
double answer_similarity(const std::string& a, const std::string& b);

// Index of the elected answer: the medoid — the candidate with the
// highest summed similarity to all the others (ties: first wins).
// Empty input returns 0; single candidate returns 0.
std::size_t elect(const std::vector<std::string>& candidates);

// The LAST number appearing in the text (sign and decimals included),
// or "" when there is none. Verbose chains of thought end with their
// conclusion: the last number IS the answer in numeric tasks.
std::string final_number(const std::string& text);

// Sharper election for numeric tasks: exact-majority vote over each
// candidate's final number; the winner is the first candidate carrying
// the majority number. Falls back to the text medoid when fewer than
// half the candidates end with a number, or on a full tie.
std::size_t elect_numeric(const std::vector<std::string>& candidates);

// Sampling config of conclave attempt `i` out of n. Attempt 0 is the
// anchor: the caller's config untouched (greedy stays greedy — the
// model's single best shot always sits in the conclave). Attempts i>0
// explore: seed shifted by i, temperature raised to at least 0.7 so the
// candidates genuinely diverge.
GenerationConfig conclave_attempt(const GenerationConfig& gen, int i);

}  // namespace sovranx::core
