#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "sovranx/core/engine.hpp"
#include "sovranx/core/llama_backend.hpp"

namespace sovranx::palimpsest {
class CorpusIndex;
}

namespace sovranx::speculative {

struct SpeculativeMetrics {
    std::uint64_t total_draft_tokens = 0;
    std::uint64_t total_accepted_tokens = 0;
    std::uint64_t total_rejected_tokens = 0;
    std::uint64_t generated_tokens = 0;  // tokens actually emitted
    double draft_time_s = 0.0;           // time spent in the draft model
    double target_time_s = 0.0;          // time spent in the target model
    double total_time_s = 0.0;           // wall time of generation

    double acceptance_rate() const {
        return total_draft_tokens == 0
                   ? 0.0
                   : static_cast<double>(total_accepted_tokens) /
                         static_cast<double>(total_draft_tokens);
    }
    double draft_speed() const {
        return draft_time_s <= 0.0 ? 0.0
                                   : static_cast<double>(total_draft_tokens) /
                                         draft_time_s;
    }
    double target_speed() const {
        return target_time_s <= 0.0
                   ? 0.0
                   : static_cast<double>(generated_tokens) / target_time_s;
    }
    double overall_speed() const {
        return total_time_s <= 0.0
                   ? 0.0
                   : static_cast<double>(generated_tokens) / total_time_s;
    }
};

// Orchestrates draft -> batched verification -> acceptance, with plain
// sequential decoding as fallback (no draft model, use_speculative=false,
// draft failure at runtime, or sustained low acceptance).
class SpeculativeDecoder {
public:
    struct Config {
        // ModelDraft: a second, smaller model proposes tokens (requires a
        // draft backend). PromptLookup: proposals come from n-gram matches
        // in the tokens already seen — no second model, draft cost zero;
        // shines when the output quotes or rephrases the input.
        enum class Mode { ModelDraft, PromptLookup };
        Mode mode = Mode::ModelDraft;

        int draft_tokens = 16;       // starting draft length
        int min_draft_tokens = 2;    // adaptive lower bound
        int max_draft_tokens = 32;   // adaptive upper bound
        int lookup_max_ngram = 3;    // PromptLookup pattern lengths
        int lookup_min_ngram = 1;
        // Optional server memory (PromptLookup mode): when the current
        // prompt has no matching n-gram, drafts are retrieved from past
        // generations, and every finished generation is observed back
        // into the archive. Non-owning; may be shared, calls are made
        // from the generation thread only.
        palimpsest::CorpusIndex* corpus = nullptr;
        bool use_speculative = true;
        // Auto-disable: if after at least `disable_after_drafted` drafted
        // tokens the acceptance rate is below `disable_below_acceptance`,
        // speculation is switched off for the rest of this decoder's life
        // (the draft model is wasting CPU the target needs).
        std::uint64_t disable_after_drafted = 64;
        double disable_below_acceptance = 0.15;
    };

    // `draft_backend` may be null: the decoder then always runs the plain
    // path. Throws EngineError if the two backends' vocabularies differ.
    SpeculativeDecoder(LlamaBackend& target_backend,
                       LlamaBackend* draft_backend,
                       const Config& cfg);

    std::vector<TokenId> generate(const std::vector<TokenId>& prompt_tokens,
                                  const core::GenerationConfig& gen_config);

    // `callback` gets each generated token; returning false stops. EOS is
    // never delivered.
    void generate_stream(const std::vector<TokenId>& prompt_tokens,
                         const std::function<bool(TokenId)>& callback,
                         const core::GenerationConfig& gen_config);

    const SpeculativeMetrics& metrics() const { return metrics_; }
    // Current adaptive draft length.
    int current_draft_tokens() const { return current_draft_tokens_; }
    // False when running plain decoding (no draft model, disabled by
    // config, auto-disabled, or draft failed at runtime).
    bool speculative_active() const { return speculative_active_; }

private:
    void adapt_draft_length(std::size_t accepted, std::size_t drafted);
    void maybe_auto_disable();

    LlamaBackend& target_;
    LlamaBackend* draft_;
    Config cfg_;
    SpeculativeMetrics metrics_;
    int current_draft_tokens_;
    bool speculative_active_;
};

}  // namespace sovranx::speculative
