#include "sovranx/server/http_server.hpp"

#include <boost/asio.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "sovranx/server/api_handler.hpp"
#include "sovranx/server/http_types.hpp"
#include "sovranx/utils/logger.hpp"

namespace sovranx::server {

namespace asio = boost::asio;
using asio::ip::tcp;

namespace {

// Writes the handler's output straight to the socket. Non-streaming
// responses are buffered and sent with Content-Length; streaming (SSE)
// sends headers immediately and each chunk as it is produced, closing the
// connection to mark the end (HTTP/1.0-style, curl-compatible).
class SocketWriter final : public ResponseWriter {
public:
    explicit SocketWriter(tcp::socket& socket) : socket_(socket) {}

    void begin(int status, const HeaderMap& headers) override {
        response_.status = status;
        response_.headers = headers;
        streaming_ = response_.header("Content-Type") == "text/event-stream";
        if (streaming_) {
            HttpResponse head = response_;
            head.headers["Connection"] = "close";
            std::string wire = serialize_http_response(head);
            // Drop the auto Content-Length: the stream length is unknown.
            const auto pos = wire.find("Content-Length: 0\r\n");
            if (pos != std::string::npos) wire.erase(pos, 19);
            write(wire);
        }
    }

    void chunk(const std::string& data) override {
        if (streaming_)
            write(data);
        else
            response_.body += data;
    }

    void end() override {
        if (!streaming_) write(serialize_http_response(response_));
    }

private:
    void write(const std::string& data) {
        boost::system::error_code ec;
        asio::write(socket_, asio::buffer(data), ec);
    }

    tcp::socket& socket_;
    HttpResponse response_;
    bool streaming_ = false;
};

}  // namespace

struct HttpServer::Impl {
    Config cfg;
    std::shared_ptr<core::SovranXEngine> engine;
    ApiHandler handler;
    Logger log{std::cout, LogLevel::Info};

    asio::io_context io;
    std::unique_ptr<tcp::acceptor> acceptor;
    std::thread accept_thread;
    std::atomic<bool> running{false};
    std::atomic<int> active_connections{0};
    int bound_port = 0;

    Impl(Config c, std::shared_ptr<core::SovranXEngine> e)
        : cfg(std::move(c)),
          engine(std::move(e)),
          handler(
              [this] {
                  ApiHandler::Config hc;
                  hc.api_key = cfg.api_key;
                  hc.enable_cors = cfg.enable_cors;
                  hc.enable_metrics = cfg.enable_metrics;
                  hc.max_concurrent_requests = cfg.max_concurrent_requests;
                  hc.model_id = cfg.model_id;
                  return hc;
              }(),
              *engine) {}

    // Exactly ONE thread runs this. The accept is NON-blocking with a
    // short poll: on Linux, close()ing the acceptor does not wake a thread
    // blocked inside accept(), so stop() would hang forever on the join.
    // Each accepted connection is handled in a detached worker so a long
    // generation never blocks the accept loop; workers are counted so
    // stop() can drain them.
    void accept_loop() {
        while (running.load()) {
            tcp::socket socket(io);
            boost::system::error_code ec;
            acceptor->accept(socket, ec);
            if (ec == asio::error::would_block || ec == asio::error::try_again) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (ec) {
                if (running.load())
                    log.warn(std::string("accept failed: ") + ec.message());
                continue;
            }
            socket.non_blocking(false, ec);  // workers use blocking reads
            ++active_connections;
            std::thread([this](tcp::socket s) {
                handle_connection(std::move(s));
                --active_connections;
            }, std::move(socket)).detach();
        }
    }

    void handle_connection(tcp::socket socket) {
        const std::size_t max_bytes = cfg.max_request_size_mb * 1024 * 1024;
        boost::system::error_code ec;

        // Read headers, then the declared body.
        std::string data;
        char buf[8192];
        std::size_t headers_end = std::string::npos;
        while (headers_end == std::string::npos) {
            const std::size_t n = socket.read_some(asio::buffer(buf), ec);
            if (ec) return;  // client hung up mid-headers
            data.append(buf, n);
            headers_end = data.find("\r\n\r\n");
            if (data.size() > 64 * 1024 && headers_end == std::string::npos)
                return;  // header flood
        }

        // Enforce the size limit BEFORE buffering the body. Content-Length
        // is read straight from the header block: the full parser would
        // reject the (intentionally) body-less prefix.
        std::size_t content_length = 0;
        {
            const std::string head = data.substr(0, headers_end);
            std::size_t pos = 0;
            while (pos < head.size()) {
                auto eol = head.find("\r\n", pos);
                if (eol == std::string::npos) eol = head.size();
                const std::string line = head.substr(pos, eol - pos);
                pos = eol + 2;
                const auto colon = line.find(':');
                if (colon == std::string::npos) continue;
                std::string name = line.substr(0, colon);
                for (char& c : name)
                    c = static_cast<char>(std::tolower(
                        static_cast<unsigned char>(c)));
                if (name == "content-length") {
                    content_length = static_cast<std::size_t>(std::strtoull(
                        line.c_str() + colon + 1, nullptr, 10));
                    break;
                }
            }
            if (content_length > max_bytes) {
                HttpResponse resp;
                resp.status = 413;
                resp.body = R"({"error":{"message":"request too large",)"
                            R"("type":"invalid_request_error",)"
                            R"("code":"payload_too_large"}})";
                resp.headers["Content-Type"] = "application/json";
                asio::write(socket, asio::buffer(serialize_http_response(resp)),
                            ec);
                return;
            }
        }
        while (data.size() - headers_end - 4 < content_length) {
            const std::size_t n = socket.read_some(asio::buffer(buf), ec);
            if (ec) return;
            data.append(buf, n);
        }

        const auto request = parse_http_request(data);
        if (!request.has_value()) {
            HttpResponse resp;
            resp.status = 400;
            resp.body = R"({"error":{"message":"malformed HTTP request",)"
                        R"("type":"invalid_request_error",)"
                        R"("code":"bad_request"}})";
            resp.headers["Content-Type"] = "application/json";
            asio::write(socket, asio::buffer(serialize_http_response(resp)),
                        ec);
            return;
        }

        if (cfg.enable_request_logging)
            log.info(request->method + " " + request->target);

        SocketWriter writer(socket);
        handler.handle(*request, writer);

        boost::system::error_code ignore;
        socket.shutdown(tcp::socket::shutdown_both, ignore);
    }
};

HttpServer::HttpServer(Config config,
                       std::shared_ptr<core::SovranXEngine> engine)
    : pimpl_(std::make_unique<Impl>(std::move(config), std::move(engine))) {
    if (pimpl_->engine == nullptr)
        throw core::EngineError("HttpServer requires an engine");
}

HttpServer::~HttpServer() {
    if (pimpl_ != nullptr && pimpl_->running.load()) stop();
}

void HttpServer::start() {
    auto& impl = *pimpl_;
    if (impl.running.load()) return;

    const auto address = asio::ip::make_address(impl.cfg.host);
    impl.acceptor = std::make_unique<tcp::acceptor>(
        impl.io, tcp::endpoint(address,
                               static_cast<unsigned short>(impl.cfg.port)));
    impl.bound_port = impl.acceptor->local_endpoint().port();
    impl.acceptor->non_blocking(true);
    impl.running.store(true);

    impl.accept_thread = std::thread([this] { pimpl_->accept_loop(); });

    impl.log.info("sovranx server listening on " + impl.cfg.host + ":" +
                  std::to_string(impl.bound_port));
}

void HttpServer::stop() {
    auto& impl = *pimpl_;
    if (!impl.running.exchange(false)) return;
    boost::system::error_code ec;
    if (impl.acceptor != nullptr) impl.acceptor->close(ec);
    if (impl.accept_thread.joinable()) impl.accept_thread.join();
    // Drain detached connection workers (they reference this Impl) with a
    // bounded wait: generations already in flight get up to 30s.
    for (int i = 0; i < 600 && impl.active_connections.load() > 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

bool HttpServer::is_running() const { return pimpl_->running.load(); }

int HttpServer::port() const { return pimpl_->bound_port; }

}  // namespace sovranx::server
