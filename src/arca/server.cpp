#include "reame/arca/server.hpp"

#include <boost/asio.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>

#include "reame/arca/archive.hpp"
#include "reame/arca/resp.hpp"

namespace reame::arca {

namespace asio = boost::asio;
using asio::ip::tcp;

struct ArcaServer::Impl {
    Config cfg;
    Archive archive;

    asio::io_context io;
    std::unique_ptr<tcp::acceptor> acceptor;
    std::thread accept_thread;
    std::atomic<bool> running{false};
    int bound_port = 0;

    // A blocking connection worker owns a socket that references `io`, so
    // every worker MUST finish before Impl (and its io_context) destructs.
    // stop() forces that: it shuts down every live socket, which makes the
    // blocked read_some return, so the workers exit before teardown.
    std::mutex conns_mutex;
    std::set<tcp::socket*> conns;
    std::atomic<int> active_connections{0};

    explicit Impl(const Config& c)
        : cfg(c), archive({c.directory, c.max_bytes}) {}

    // One accept thread. Non-blocking accept polled at 20 ms so stop()
    // never hangs on the join (close() does not wake a blocked accept on
    // Linux). Each connection is served by a detached, counted worker.
    void accept_loop() {
        while (running.load()) {
            tcp::socket socket(io);
            boost::system::error_code ec;
            acceptor->accept(socket, ec);
            if (ec == asio::error::would_block ||
                ec == asio::error::try_again) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (ec) continue;
            socket.non_blocking(false, ec);
            ++active_connections;
            std::thread([this](tcp::socket s) {
                serve(std::move(s));
                --active_connections;
            }, std::move(socket)).detach();
        }
    }

    // Reads bytes, assembles commands with the incremental parser,
    // dispatches each to the shared archive, writes replies back. A Redis
    // client keeps the connection open and pipelines many commands.
    void serve(tcp::socket socket) {
        {
            std::lock_guard<std::mutex> lock(conns_mutex);
            conns.insert(&socket);
        }
        RespParser parser;
        char buf[8192];
        for (;;) {
            boost::system::error_code ec;
            const std::size_t n = socket.read_some(asio::buffer(buf), ec);
            if (ec) break;  // client hung up, or stop() shut us down
            try {
                parser.feed(std::string_view(buf, n));
            } catch (const RespError& e) {
                asio::write(socket,
                            asio::buffer(resp_error(std::string("ERR ") +
                                                    e.what())),
                            ec);
                break;
            }
            std::string out;
            for (const auto& cmd : parser.take()) out += archive.dispatch(cmd);
            if (!out.empty()) {
                asio::write(socket, asio::buffer(out), ec);
                if (ec) break;
            }
        }
        std::lock_guard<std::mutex> lock(conns_mutex);
        conns.erase(&socket);
    }
};

ArcaServer::ArcaServer(const Config& cfg)
    : pimpl_(std::make_unique<Impl>(cfg)) {}

ArcaServer::~ArcaServer() { stop(); }

void ArcaServer::start() {
    auto& p = *pimpl_;
    if (p.running.load()) return;
    tcp::endpoint endpoint(tcp::v4(),
                           static_cast<unsigned short>(p.cfg.port));
    p.acceptor = std::make_unique<tcp::acceptor>(p.io);
    p.acceptor->open(endpoint.protocol());
    p.acceptor->set_option(asio::socket_base::reuse_address(true));
    p.acceptor->bind(endpoint);
    p.acceptor->listen();
    p.acceptor->non_blocking(true);
    p.bound_port = p.acceptor->local_endpoint().port();
    p.running.store(true);
    p.accept_thread = std::thread([&p] { p.accept_loop(); });
}

void ArcaServer::stop() {
    auto& p = *pimpl_;
    if (!p.running.exchange(false)) return;
    if (p.accept_thread.joinable()) p.accept_thread.join();
    boost::system::error_code ec;
    if (p.acceptor) p.acceptor->close(ec);
    // Force every blocked worker to return by shutting down its socket,
    // then wait for them to unwind — no worker may outlive the io_context.
    {
        std::lock_guard<std::mutex> lock(p.conns_mutex);
        for (tcp::socket* s : p.conns)
            s->shutdown(tcp::socket::shutdown_both, ec);
    }
    while (p.active_connections.load() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

int ArcaServer::port() const { return pimpl_->bound_port; }

}  // namespace reame::arca
