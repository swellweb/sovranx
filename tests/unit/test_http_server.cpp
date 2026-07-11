// End-to-end loop test for the Asio server shell: real TCP socket, engine
// over MockBackend (no model, no llama). Verifies accept -> parse -> route
// -> respond, streaming included.

#include <catch2/catch_test_macros.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "../mock/llama_mock.hpp"
#include "sovranx/server/http_server.hpp"

using nlohmann::json;
using sovranx::test::MockBackend;
using sovranx::core::SovranXEngine;
using sovranx::server::HttpServer;

namespace {

std::vector<float> peak(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 10.0f;
    return v;
}

std::shared_ptr<SovranXEngine> make_engine() {
    SovranXEngine::Config ec;
    ec.model_path = "/models/test.gguf";
    ec.n_ctx = 2048;
    ec.n_threads = 4;
    auto backend = std::make_unique<MockBackend>();
    backend->vocab_size_value = 5;
    backend->eos_token_value = 4;
    backend->tokenize_result = {1, 2};
    backend->piece_map = {{3, "foo"}, {0, "bar"}};
    backend->decode_queue = {peak(5, 3), peak(5, 0), peak(5, 4)};
    return std::make_shared<SovranXEngine>(ec, std::move(backend));
}

// Blocking one-shot HTTP client: connect, send, read to EOF.
std::string http_roundtrip(int port, const std::string& raw) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    REQUIRE(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) ==
            0);

    REQUIRE(::send(fd, raw.data(), raw.size(), 0) ==
            static_cast<ssize_t>(raw.size()));

    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0)
        out.append(buf, static_cast<std::size_t>(n));
    ::close(fd);
    return out;
}

std::string body_of(const std::string& raw) {
    const auto pos = raw.find("\r\n\r\n");
    return pos == std::string::npos ? "" : raw.substr(pos + 4);
}

}  // namespace

TEST_CASE("server: health and completions over a real socket") {
    HttpServer::Config cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 0;  // ephemeral
    cfg.enable_request_logging = false;

    HttpServer server(cfg, make_engine());
    server.start();
    REQUIRE(server.is_running());
    const int port = server.port();
    REQUIRE(port > 0);

    SECTION("GET /health") {
        const auto raw = http_roundtrip(
            port, "GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
        CHECK(raw.rfind("HTTP/1.1 200 OK", 0) == 0);
        CHECK(json::parse(body_of(raw))["status"] == "ok");
    }

    SECTION("POST /v1/completions") {
        const std::string body =
            json{{"prompt", "hi"}, {"temperature", 0.0}}.dump();
        const std::string raw_req =
            "POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" +
            body;
        const auto raw = http_roundtrip(port, raw_req);
        CHECK(raw.rfind("HTTP/1.1 200 OK", 0) == 0);
        CHECK(json::parse(body_of(raw))["choices"][0]["text"] == "foobar");
    }

    SECTION("streaming completions arrive as SSE and end with [DONE]") {
        const std::string body =
            json{{"prompt", "hi"}, {"temperature", 0.0}, {"stream", true}}
                .dump();
        const std::string raw_req =
            "POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" +
            body;
        const auto raw = http_roundtrip(port, raw_req);
        CHECK(raw.find("Content-Type: text/event-stream") != std::string::npos);
        CHECK(raw.find("data: ") != std::string::npos);
        CHECK(raw.find("data: [DONE]") != std::string::npos);
    }

    SECTION("malformed request gets 400") {
        const auto raw = http_roundtrip(port, "NONSENSE\r\n\r\n");
        CHECK(raw.rfind("HTTP/1.1 400", 0) == 0);
    }

    server.stop();
    CHECK_FALSE(server.is_running());
}

TEST_CASE("server: oversized requests are rejected with 413") {
    HttpServer::Config cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 0;
    cfg.max_request_size_mb = 1;
    cfg.enable_request_logging = false;

    HttpServer server(cfg, make_engine());
    server.start();

    // Declare a body far beyond the limit; the server must refuse without
    // buffering it all.
    const std::string raw_req =
        "POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
        "Content-Length: 99999999\r\n\r\n";
    const auto raw = http_roundtrip(server.port(), raw_req);
    CHECK(raw.rfind("HTTP/1.1 413", 0) == 0);

    server.stop();
}
