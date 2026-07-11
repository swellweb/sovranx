#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "sovranx/core/engine.hpp"
#include "sovranx/server/http_types.hpp"

namespace sovranx::server {

// Protocol layer: maps HTTP requests onto the engine. Pure with respect to
// the network — input is an HttpRequest, output goes to a ResponseWriter —
// so the whole API surface is unit-testable without sockets.
//
// Endpoints (OpenAI-compatible where applicable):
//   GET    /health                      liveness (auth-exempt)
//   GET    /v1/models                   configured model
//   POST   /v1/completions              text completion (stream via SSE)
//   POST   /v1/chat/completions         chat completion (stream via SSE)
//   POST   /v1/sessions                 create session -> {"id"}
//   POST   /v1/sessions/{id}/save       snapshot current context
//   POST   /v1/sessions/{id}/load       restore snapshot
//   DELETE /v1/sessions/{id}            drop session
//   GET    /metrics                     server + speculative metrics
//
// Errors use the OpenAI envelope: {"error":{"message","type","code"}}.
class ApiHandler {
public:
    struct Config {
        std::string api_key;  // empty = authentication disabled
        bool enable_cors = true;
        bool enable_metrics = true;
        int max_concurrent_requests = 10;  // generation endpoints -> 503
        std::string model_id = "sovranx";
    };

    ApiHandler(const Config& cfg, core::SovranXEngine& engine);
    ~ApiHandler();
    ApiHandler(const ApiHandler&) = delete;
    ApiHandler& operator=(const ApiHandler&) = delete;

    // Streaming-capable entry point (SSE goes through the writer).
    void handle(const HttpRequest& request, ResponseWriter& writer);

    // Buffering convenience for non-streaming callers and tests.
    HttpResponse handle(const HttpRequest& request);

    struct Stats {
        std::uint64_t total_requests = 0;
        std::uint64_t errors = 0;
    };
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace sovranx::server
