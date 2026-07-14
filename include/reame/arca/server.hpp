#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

namespace reame::arca {

// The ARCA daemon: a TCP server speaking RESP2, so every language's Redis
// client can reach the realm's archive with no SDK. One accept thread, a
// detached worker per connection, all sharing one thread-safe Archive.
// The socket pattern mirrors the HTTP server: a non-blocking acceptor
// polled at 20 ms, because close() does not wake a blocked accept() on
// Linux and stop() would otherwise hang on the join.
class ArcaServer {
public:
    struct Config {
        int port = 6420;                  // 0 = ephemeral (the OS picks)
        std::filesystem::path directory;  // where L1 entries live
        std::uint64_t max_bytes = 0;      // 0 = unlimited LRU budget
        // Refuse a single request line/bulk larger than this — a bad
        // client must not be able to buffer the daemon to death.
        std::size_t max_request_bytes = 64 * 1024 * 1024;
    };

    explicit ArcaServer(const Config& cfg);
    ~ArcaServer();

    ArcaServer(const ArcaServer&) = delete;
    ArcaServer& operator=(const ArcaServer&) = delete;

    // Binds, then spawns the accept thread. Throws on bind failure.
    void start();
    // Stops accepting and joins the accept thread. Idempotent.
    void stop();

    // The actually-bound port (meaningful after start(), needed when the
    // config asked for an ephemeral port 0).
    int port() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace reame::arca
