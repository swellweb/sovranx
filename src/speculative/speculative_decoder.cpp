#include "reame/speculative/speculative_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <utility>

#include "reame/core/sampler.hpp"
#include "reame/palimpsest/corpus_index.hpp"
#include "reame/speculative/batch_verifier.hpp"
#include "reame/speculative/draft_generator.hpp"
#include "reame/speculative/form_draft.hpp"
#include "reame/speculative/prompt_lookup.hpp"

namespace reame::speculative {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

SpeculativeDecoder::SpeculativeDecoder(LlamaBackend& target,
                                       LlamaBackend* draft, const Config& cfg)
    : target_(target),
      draft_(draft),
      cfg_(cfg),
      current_draft_tokens_(std::clamp(cfg.draft_tokens, cfg.min_draft_tokens,
                                       cfg.max_draft_tokens)),
      speculative_active_(
          cfg.use_speculative &&
          (cfg.mode == Config::Mode::PromptLookup || draft != nullptr)) {
    if (draft != nullptr) {
        // Same tokenizer required, but model families pad the embedding
        // table differently (Qwen2.5-7B: 152064, 0.5B: 151936). Tolerate
        // up to 128 entries of padding, like llama.cpp's speculative
        // example; the guard in the generation loop handles the rare case
        // of an emitted token outside the draft's table.
        const int diff = std::abs(draft->vocab_size() - target.vocab_size());
        if (diff > 128)
            throw core::EngineError(
                "draft and target vocabularies differ (" +
                std::to_string(draft->vocab_size()) + " vs " +
                std::to_string(target.vocab_size()) +
                "): speculative decoding requires a shared tokenizer");
    }
}

std::vector<TokenId> SpeculativeDecoder::generate(
    const std::vector<TokenId>& prompt_tokens,
    const core::GenerationConfig& gen_config) {
    std::vector<TokenId> out;
    generate_stream(
        prompt_tokens,
        [&out](TokenId t) {
            out.push_back(t);
            return true;
        },
        gen_config);
    return out;
}

void SpeculativeDecoder::generate_stream(
    const std::vector<TokenId>& prompt_tokens,
    const std::function<bool(TokenId)>& callback,
    const core::GenerationConfig& gen_config) {
    if (prompt_tokens.empty())
        throw core::EngineError("prompt is empty");
    const auto n_ctx = static_cast<std::size_t>(target_.context_length());
    if (prompt_tokens.size() > n_ctx)
        throw core::EngineError("prompt of " +
                                std::to_string(prompt_tokens.size()) +
                                " tokens exceeds context length " +
                                std::to_string(n_ctx));

    const auto wall_start = Clock::now();

    core::Sampler sampler(gen_config);
    std::mt19937 accept_rng(
        static_cast<std::mt19937::result_type>(gen_config.seed) ^ 0x9E3779B9u);

    // KV invariant: each model holds every token of `history` except the
    // newest one (`cur`), which is fed on the next forward.
    std::vector<TokenId> history = prompt_tokens;

    // Server memory: whatever this generation produced becomes retrieval
    // material for future ones, on every exit path.
    struct ObserveGuard {
        palimpsest::CorpusIndex* corpus;
        const std::vector<TokenId>& tokens;
        ~ObserveGuard() {
            if (corpus == nullptr) return;
            try {
                corpus->observe(tokens);
            } catch (...) {
            }
        }
    } observe_guard{
        cfg_.mode == Config::Mode::PromptLookup ? cfg_.corpus : nullptr,
        history};

    TokenId cur = history.back();
    std::size_t base = history.size() - 1;  // positions currently in KV

    {
        const auto t0 = Clock::now();
        target_.reset();
        if (base > 0)
            target_.decode_append(
                {history.begin(), history.begin() + static_cast<long>(base)});
        metrics_.target_time_s += seconds_since(t0);
    }
    const bool model_mode = cfg_.mode == Config::Mode::ModelDraft;
    if (speculative_active_ && model_mode) {
        try {
            const auto t0 = Clock::now();
            draft_->reset();
            if (base > 0)
                draft_->decode_append(
                    {history.begin(),
                     history.begin() + static_cast<long>(base)});
            metrics_.draft_time_s += seconds_since(t0);
        } catch (const std::exception&) {
            speculative_active_ = false;  // draft broken: plain decoding
        }
    }

    DraftGenerator draft_gen(
        (model_mode && draft_ != nullptr) ? *draft_ : target_,
        {cfg_.draft_tokens});
    const PromptLookup lookup(
        {cfg_.lookup_max_ngram, cfg_.lookup_min_ngram});
    BatchVerifier verifier(target_);

    std::size_t emitted = 0;
    const auto max_tokens = static_cast<std::size_t>(
        std::max(gen_config.max_tokens, 0));

    const TokenId draft_vocab =
        (model_mode && draft_ != nullptr) ? draft_->vocab_size()
                                          : target_.vocab_size();
    bool lookup_skip = false;  // this iteration found no n-gram match

    while (emitted < max_tokens && history.size() < n_ctx) {
        // A token beyond the draft's (smaller, padding-trimmed) vocabulary
        // cannot be fed back to the draft model: finish the generation on
        // the plain path. (Lookup proposals are history tokens: always in
        // the target vocabulary.)
        if (speculative_active_ && model_mode && cur >= draft_vocab)
            speculative_active_ = false;

        if (speculative_active_ && !lookup_skip) {
            // ---- Draft rollout ----
            const int room = static_cast<int>(n_ctx - history.size());
            const int n_draft = std::min(current_draft_tokens_, room);
            DraftResult draft;
            const auto d0 = Clock::now();
            if (model_mode) {
                try {
                    draft =
                        draft_gen.generate_draft(cur, history, sampler, n_draft);
                } catch (const std::exception&) {
                    metrics_.draft_time_s += seconds_since(d0);
                    speculative_active_ = false;  // fall back, state untouched
                    continue;
                }
            } else {
                // Prompt lookup: zero-cost proposals from the history;
                // when the prompt has no repeats, fall back to the server
                // archive (past generations). A miss is fine — one plain
                // step and try again.
                draft.tokens = lookup.find_draft(history, n_draft);
                if (draft.tokens.empty() && cfg_.corpus != nullptr)
                    draft.tokens = cfg_.corpus->draft(history, n_draft);
                if (draft.tokens.empty()) {
                    // The prompter: structural continuations (list
                    // numbering, bullets) predictable on content nobody
                    // has ever generated before.
                    const std::size_t tail_tokens =
                        std::min<std::size_t>(history.size(), 48);
                    const auto tail_text = target_.detokenize(
                        {history.end() - static_cast<long>(tail_tokens),
                         history.end()});
                    const auto proposal = FormDraft::propose(tail_text);
                    if (!proposal.empty())
                        draft.tokens = target_.tokenize(
                            proposal, /*add_special=*/false);
                    if (static_cast<int>(draft.tokens.size()) > n_draft)
                        draft.tokens.resize(
                            static_cast<std::size_t>(n_draft));
                }
                const auto n_vocab =
                    static_cast<std::size_t>(target_.vocab_size());
                for (const TokenId t : draft.tokens) {
                    std::vector<float> one_hot(n_vocab, 0.0f);
                    one_hot[static_cast<std::size_t>(t)] = 1.0f;
                    draft.probs.push_back(std::move(one_hot));
                }
            }
            metrics_.draft_time_s += seconds_since(d0);
            if (draft.tokens.empty()) {
                if (model_mode) {
                    speculative_active_ = false;
                } else {
                    lookup_skip = true;  // plain step below, stay armed
                }
                continue;
            }

            // ---- Batched verification ----
            const auto t0 = Clock::now();
            const auto vr = verifier.verify_batch(cur, draft, sampler,
                                                  accept_rng, history);
            metrics_.target_time_s += seconds_since(t0);

            metrics_.total_draft_tokens += draft.tokens.size();
            metrics_.total_accepted_tokens += vr.accepted_tokens.size();
            metrics_.total_rejected_tokens +=
                static_cast<std::uint64_t>(vr.rejected_count);

            // ---- KV sync ----
            // The batch advanced both models by draft.tokens.size()
            // positions. On rejection, roll back to base + cur + accepted.
            if (vr.has_correction) {
                const auto keep = static_cast<std::uint32_t>(
                    base + 1 + vr.accepted_tokens.size());
                target_.truncate_to(keep);
                if (model_mode && draft_ != nullptr) draft_->truncate_to(keep);
                base = keep;
            } else {
                base += draft.tokens.size();
            }

            // ---- Emission ----
            std::vector<TokenId> new_tokens = vr.accepted_tokens;
            if (vr.has_correction) new_tokens.push_back(vr.corrected_token);

            for (const TokenId t : new_tokens) {
                if (target_.is_eog(t)) {
                    metrics_.total_time_s += seconds_since(wall_start);
                    return;
                }
                if (!callback(t)) {
                    metrics_.total_time_s += seconds_since(wall_start);
                    return;
                }
                history.push_back(t);
                ++emitted;
                ++metrics_.generated_tokens;
                cur = t;
                if (emitted >= max_tokens || history.size() >= n_ctx) break;
            }

            adapt_draft_length(vr.accepted_tokens.size(), draft.tokens.size());
            maybe_auto_disable();
        } else {
            // ---- Plain sequential decoding ----
            lookup_skip = false;  // retry the lookup on the next iteration
            const auto t0 = Clock::now();
            std::vector<float> logits;
            logits = target_.decode_append({cur});
            metrics_.target_time_s += seconds_since(t0);
            base += 1;

            const TokenId next = sampler.sample(std::move(logits), history);
            if (target_.is_eog(next)) break;
            if (!callback(next)) break;
            history.push_back(next);
            ++emitted;
            ++metrics_.generated_tokens;
            cur = next;
        }
    }

    metrics_.total_time_s += seconds_since(wall_start);
}

void SpeculativeDecoder::adapt_draft_length(std::size_t accepted,
                                            std::size_t drafted) {
    if (drafted == 0) return;
    const double rate =
        static_cast<double>(accepted) / static_cast<double>(drafted);
    if (rate >= 0.9)
        current_draft_tokens_ =
            std::min(current_draft_tokens_ + 2, cfg_.max_draft_tokens);
    else if (rate <= 0.4)
        current_draft_tokens_ =
            std::max(current_draft_tokens_ - 2, cfg_.min_draft_tokens);
}

void SpeculativeDecoder::maybe_auto_disable() {
    if (!speculative_active_) return;
    if (metrics_.total_draft_tokens < cfg_.disable_after_drafted) return;
    if (metrics_.acceptance_rate() < cfg_.disable_below_acceptance)
        speculative_active_ = false;
}

}  // namespace reame::speculative
