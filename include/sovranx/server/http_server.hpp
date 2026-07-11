#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "sovranx/core/engine.hpp"

namespace sovranx::server {

// Thin Boost.Asio shell around ApiHandler: accepts connections, reads one
// request (bounded by max_request_size_mb), routes it, writes the response
// and closes. Generation is serialized inside the handler; excess
// concurrent generation requests get 503.
class HttpServer {
public:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8080;  // 0 = ephemeral (tests); see port()
        int threads = 2;
        int max_concurrent_requests = 10;
        int timeout_seconds = 300;
        std::size_t max_request_size_mb = 10;
        bool enable_cors = true;
        bool enable_metrics = true;
        bool enable_request_logging = true;
        std::string api_key;  // empty = authentication disabled
        std::string model_id = "sovranx";
    };

    HttpServer(Config config, std::shared_ptr<core::SovranXEngine> engine);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void start();  // non-blocking: spawns the I/O threads
    void stop();
    bool is_running() const;
    int port() const;  // actual bound port (after start)

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace sovranx::server
