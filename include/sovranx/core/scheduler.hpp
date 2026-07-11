#pragma once

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include "sovranx/core/engine.hpp"
#include "sovranx/core/llama_backend.hpp"

namespace sovranx::core {

// Interleaved multi-request scheduler: every step() packs one slice per
// active request into a SINGLE multi-sequence forward pass, so N users
// share each read of the model weights — on memory-bound CPUs that is
// nearly the cost of one user.
//
// Deliberately synchronous and single-threaded: submit() enqueues,
// step() advances everything by one batch. Tests drive step() directly;
// production wraps it in one worker thread (see SovranXEngine).
class Scheduler {
public:
    struct Config {
        int n_parallel = 2;  // max concurrent sequences (context n_seq_max)
    };

    // `on_token` is invoked once per generated token, from whichever
    // thread calls step(); returning false ends that request early.
    using TokenCallback = std::function<bool(TokenId)>;

    Scheduler(LlamaBackend& backend, const Config& cfg);
    ~Scheduler();

    // Enqueues a request. Throws EngineError if the prompt alone can never
    // fit the context. Thread-safe.
    std::uint64_t submit(std::vector<TokenId> prompt_tokens,
                         const GenerationConfig& gen, TokenCallback on_token);

    // Admits pending requests into free slots and advances every active
    // request by one batched decode. Returns true while any request is
    // pending or active. A backend failure fails the requests that were in
    // flight (their error() is set) and the scheduler keeps serving.
    bool step();

    // Test/embedding helper: step() until everything drains.
    void run_until_idle();

    bool finished(std::uint64_t id) const;
    // Non-null if the request failed; rethrowable.
    std::exception_ptr error(std::uint64_t id) const;

    int active_count() const;
    // True when nothing is pending or active.
    bool idle() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace sovranx::core
