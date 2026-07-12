#include "reame/core/scheduler.hpp"

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <utility>

#include "reame/core/sampler.hpp"

namespace reame::core {

namespace {

struct Request {
    std::uint64_t id = 0;
    std::vector<TokenId> history;  // prompt, then prompt+generated
    std::size_t prompt_len = 0;
    GenerationConfig gen;
    Scheduler::TokenCallback on_token;
    std::unique_ptr<Sampler> sampler;
    int emitted = 0;
    // Slot state (active only).
    std::int32_t seq_id = -1;
    std::uint32_t n_past = 0;      // positions already in the KV
    bool needs_prefill = true;
};

}  // namespace

struct Scheduler::Impl {
    LlamaBackend& backend;
    Config cfg;
    std::uint32_t n_ctx;

    mutable std::mutex mutex;
    // Completion state lives under its OWN mutex: finished()/error() must
    // never contend with the main mutex, which step() holds for the whole
    // batched decode — on loaded machines that contention starves waiters
    // (unfair mutexes let the stepping loop re-acquire forever) and
    // cross-thread signals like the conclave's early stop never land.
    mutable std::mutex done_mutex;
    std::deque<Request> pending;
    std::vector<Request> active;        // at most cfg.n_parallel
    std::vector<bool> slot_used;        // seq_id occupancy
    std::map<std::uint64_t, std::exception_ptr> done;  // id -> error (or null)
    std::uint64_t next_id = 1;

    Impl(LlamaBackend& b, const Config& c)
        : backend(b), cfg(c), n_ctx(b.context_length()),
          slot_used(static_cast<std::size_t>(std::max(c.n_parallel, 1)),
                    false) {}

    std::uint32_t cells_used() const {
        std::uint32_t used = 0;
        for (const auto& r : active)
            used += static_cast<std::uint32_t>(r.history.size());
        return used;
    }

    void finish(Request& r, std::exception_ptr err) {
        if (r.seq_id >= 0) {
            backend.clear_seq(r.seq_id);
            slot_used[static_cast<std::size_t>(r.seq_id)] = false;
        }
        std::lock_guard<std::mutex> done_lock(done_mutex);
        done[r.id] = std::move(err);
    }

    // An active request whose prompt equals `prompt` (donor for a KV
    // clone), or nullptr.
    const Request* find_donor(const std::vector<TokenId>& prompt) const {
        for (const auto& a : active) {
            if (a.prompt_len != prompt.size()) continue;
            if (std::equal(prompt.begin(), prompt.end(), a.history.begin()))
                return &a;
        }
        return nullptr;
    }

    // Move pending requests into free slots while they fit the KV budget.
    void admit() {
        while (!pending.empty() &&
               active.size() < static_cast<std::size_t>(cfg.n_parallel)) {
            const auto need =
                static_cast<std::uint32_t>(pending.front().history.size());
            if (cells_used() + need > n_ctx) break;  // wait for space

            // Shared prefill: an identical prompt already in flight is
            // cloned (copy the donor's prompt KV, then decode only the
            // last prompt token — it yields the same logits a prefill
            // would). A donor still prefilling is worth one step's wait.
            const Request* donor = pending.front().history.size() > 1
                                       ? find_donor(pending.front().history)
                                       : nullptr;
            if (donor != nullptr && donor->n_past < donor->prompt_len)
                break;  // donor's prefill lands next step

            Request r = std::move(pending.front());
            pending.pop_front();
            for (std::size_t s = 0; s < slot_used.size(); ++s) {
                if (!slot_used[s]) {
                    slot_used[s] = true;
                    r.seq_id = static_cast<std::int32_t>(s);
                    break;
                }
            }
            if (donor != nullptr) {
                const auto shared =
                    static_cast<std::uint32_t>(r.prompt_len - 1);
                backend.copy_seq(donor->seq_id, r.seq_id, shared);
                r.n_past = shared;
                r.needs_prefill = false;
            }
            active.push_back(std::move(r));
        }
    }

    bool step() {
        std::lock_guard<std::mutex> lock(mutex);
        admit();
        if (active.empty()) return !pending.empty();

        // One slice per active request: the full prompt when prefilling,
        // otherwise the single token sampled last round.
        std::vector<SeqSlice> slices;
        slices.reserve(active.size());
        for (auto& r : active) {
            SeqSlice s;
            s.seq_id = r.seq_id;
            if (r.needs_prefill) {
                s.tokens = r.history;
                s.pos_start = 0;
            } else {
                s.tokens = {r.history.back()};
                s.pos_start = r.n_past;
            }
            slices.push_back(std::move(s));
        }

        std::vector<std::vector<float>> logits;
        try {
            logits = backend.decode_seqs(slices);
        } catch (...) {
            // The whole batch failed: fail everything that was in flight.
            const auto err = std::current_exception();
            for (auto& r : active) finish(r, err);
            active.clear();
            return !pending.empty();
        }

        // Sample one token per request; retire the finished ones. A fault
        // in one request's sampling or callback fails ONLY that request —
        // it must never take down the shared worker thread (and with it
        // the whole server).
        std::vector<Request> still_active;
        still_active.reserve(active.size());
        for (std::size_t i = 0; i < active.size(); ++i) {
            auto& r = active[i];
            r.n_past += static_cast<std::uint32_t>(slices[i].tokens.size());
            r.needs_prefill = false;

            try {
                const TokenId next =
                    r.sampler->sample(std::move(logits[i]), r.history);
                if (backend.is_eog(next)) {
                    finish(r, nullptr);
                    continue;
                }
                const bool keep_going = r.on_token(next);
                r.history.push_back(next);
                ++r.emitted;

                if (!keep_going || r.emitted >= r.gen.max_tokens ||
                    r.history.size() >= n_ctx) {
                    finish(r, nullptr);
                    continue;
                }
                still_active.push_back(std::move(r));
            } catch (...) {
                finish(r, std::current_exception());
            }
        }
        active = std::move(still_active);
        return !active.empty() || !pending.empty();
    }
};

Scheduler::Scheduler(LlamaBackend& backend, const Config& cfg)
    : pimpl_(std::make_unique<Impl>(backend, cfg)) {
    if (cfg.n_parallel < 1)
        throw EngineError("n_parallel must be >= 1");
}

Scheduler::~Scheduler() = default;

std::uint64_t Scheduler::submit(std::vector<TokenId> prompt_tokens,
                                const GenerationConfig& gen,
                                TokenCallback on_token) {
    if (prompt_tokens.empty())
        throw EngineError("prompt is empty");
    if (prompt_tokens.size() > pimpl_->n_ctx)
        throw EngineError("prompt of " +
                          std::to_string(prompt_tokens.size()) +
                          " tokens exceeds context length " +
                          std::to_string(pimpl_->n_ctx));
    if (!on_token) throw EngineError("callback is null");

    std::lock_guard<std::mutex> lock(pimpl_->mutex);
    Request r;
    r.id = pimpl_->next_id++;
    r.history = std::move(prompt_tokens);
    r.prompt_len = r.history.size();
    r.gen = gen;
    r.on_token = std::move(on_token);
    r.sampler = std::make_unique<Sampler>(gen);
    const auto id = r.id;
    pimpl_->pending.push_back(std::move(r));
    return id;
}

bool Scheduler::step() { return pimpl_->step(); }

void Scheduler::run_until_idle() {
    while (step()) {
    }
}

bool Scheduler::finished(std::uint64_t id) const {
    std::lock_guard<std::mutex> lock(pimpl_->done_mutex);
    return pimpl_->done.find(id) != pimpl_->done.end();
}

std::exception_ptr Scheduler::error(std::uint64_t id) const {
    std::lock_guard<std::mutex> lock(pimpl_->done_mutex);
    const auto it = pimpl_->done.find(id);
    return it == pimpl_->done.end() ? nullptr : it->second;
}

int Scheduler::active_count() const {
    std::lock_guard<std::mutex> lock(pimpl_->mutex);
    return static_cast<int>(pimpl_->active.size());
}

bool Scheduler::idle() const {
    std::lock_guard<std::mutex> lock(pimpl_->mutex);
    return pimpl_->active.empty() && pimpl_->pending.empty();
}

}  // namespace reame::core
