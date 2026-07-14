// Integration tests for the ARCA daemon: a real loopback socket speaks
// RESP to the running server, exactly as redis-cli would. No model, no
// llama.cpp — just the protocol, the archive and the socket.

#include <catch2/catch_test_macros.hpp>

#include <boost/asio.hpp>

#include <filesystem>
#include <string>
#include <unistd.h>

#include "reame/arca/server.hpp"

using boost::asio::ip::tcp;
using reame::arca::ArcaServer;

namespace {

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() /
               ("arca-srv-" + std::to_string(::getpid()) + "-" +
                std::to_string(counter++));
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::filesystem::remove_all(path); }
    static inline int counter = 0;
};

// A minimal blocking RESP client over a real socket.
class Client {
public:
    explicit Client(int port) : socket_(io_) {
        socket_.connect(
            tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"),
                          static_cast<unsigned short>(port)));
    }

    // Sends raw bytes, reads until `expected_len` bytes come back.
    std::string roundtrip(const std::string& wire, std::size_t expected_len) {
        boost::asio::write(socket_, boost::asio::buffer(wire));
        std::string out;
        char buf[512];
        while (out.size() < expected_len) {
            boost::system::error_code ec;
            const std::size_t n =
                socket_.read_some(boost::asio::buffer(buf), ec);
            if (ec) break;
            out.append(buf, n);
        }
        return out;
    }

private:
    boost::asio::io_context io_;
    tcp::socket socket_;
};

ArcaServer::Config cfg(const TempDir& dir) {
    ArcaServer::Config c;
    c.port = 0;  // ephemeral: the OS picks a free port
    c.directory = dir.path;
    return c;
}

}  // namespace

TEST_CASE("[integration] arca daemon: PING over a real socket") {
    TempDir dir;
    ArcaServer server(cfg(dir));
    server.start();

    Client c(server.port());
    CHECK(c.roundtrip("*1\r\n$4\r\nPING\r\n", 7) == "+PONG\r\n");

    server.stop();
}

TEST_CASE("[integration] arca daemon: SET then GET round-trips") {
    TempDir dir;
    ArcaServer server(cfg(dir));
    server.start();

    Client c(server.port());
    CHECK(c.roundtrip("*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$5\r\nhello\r\n", 5) ==
          "+OK\r\n");
    CHECK(c.roundtrip("*2\r\n$3\r\nGET\r\n$1\r\nk\r\n", 11) ==
          "$5\r\nhello\r\n");

    server.stop();
}

TEST_CASE("[integration] arca daemon: pipelined commands on one connection") {
    TempDir dir;
    ArcaServer server(cfg(dir));
    server.start();

    Client c(server.port());
    // Two commands in one write; two replies must come back in order.
    const auto reply =
        c.roundtrip("*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nPING\r\n", 14);
    CHECK(reply == "+PONG\r\n+PONG\r\n");

    server.stop();
}

TEST_CASE("[integration] arca daemon: persistence across restart") {
    TempDir dir;
    {
        ArcaServer server(cfg(dir));
        server.start();
        Client c(server.port());
        c.roundtrip("*3\r\n$3\r\nSET\r\n$3\r\nans\r\n$2\r\n42\r\n", 5);
        server.stop();
    }
    // A second daemon on the same directory sees the first one's write.
    ArcaServer server(cfg(dir));
    server.start();
    Client c(server.port());
    CHECK(c.roundtrip("*2\r\n$3\r\nGET\r\n$3\r\nans\r\n", 8) == "$2\r\n42\r\n");
    server.stop();
}

TEST_CASE("[integration] arca daemon: start/stop is clean and repeatable") {
    TempDir dir;
    for (int i = 0; i < 3; ++i) {
        ArcaServer server(cfg(dir));
        server.start();
        CHECK(server.port() > 0);
        Client c(server.port());
        CHECK(c.roundtrip("*1\r\n$4\r\nPING\r\n", 7) == "+PONG\r\n");
        server.stop();
    }
}
