// Isolated tests for the HTTP request parser / response serializer. Pure
// string-in/string-out, no sockets.

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "sovranx/server/http_types.hpp"

using sovranx::server::HttpResponse;
using sovranx::server::parse_http_request;
using sovranx::server::serialize_http_response;

TEST_CASE("parses a GET request with headers") {
    const auto req = parse_http_request(
        "GET /health HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "Accept: application/json\r\n"
        "\r\n");

    REQUIRE(req.has_value());
    CHECK(req->method == "GET");
    CHECK(req->target == "/health");
    CHECK(req->header("Host") == "localhost:8080");
    CHECK(req->header("Accept") == "application/json");
    CHECK(req->body.empty());
}

TEST_CASE("parses a POST body via Content-Length") {
    const auto req = parse_http_request(
        "POST /v1/completions HTTP/1.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 16\r\n"
        "\r\n"
        "{\"prompt\":\"hi\"}\n");

    REQUIRE(req.has_value());
    CHECK(req->method == "POST");
    CHECK(req->body == "{\"prompt\":\"hi\"}\n");
}

TEST_CASE("header lookup is case-insensitive") {
    const auto req = parse_http_request(
        "GET / HTTP/1.1\r\n"
        "AUTHORIZATION: Bearer sk-x\r\n"
        "\r\n");

    REQUIRE(req.has_value());
    CHECK(req->header("authorization") == "Bearer sk-x");
    CHECK(req->header("Authorization") == "Bearer sk-x");
}

TEST_CASE("header values are trimmed of surrounding whitespace") {
    const auto req = parse_http_request(
        "GET / HTTP/1.1\r\n"
        "X-Padded:    value with spaces   \r\n"
        "\r\n");

    REQUIRE(req.has_value());
    CHECK(req->header("X-Padded") == "value with spaces");
}

TEST_CASE("the query string is stripped from the target") {
    const auto req = parse_http_request(
        "GET /health?verbose=1&x=2 HTTP/1.1\r\n\r\n");

    REQUIRE(req.has_value());
    CHECK(req->target == "/health");
}

TEST_CASE("malformed requests are rejected") {
    CHECK_FALSE(parse_http_request("").has_value());
    CHECK_FALSE(parse_http_request("garbage").has_value());
    // Missing the blank line terminating the headers.
    CHECK_FALSE(parse_http_request("GET / HTTP/1.1\r\nHost: x\r\n").has_value());
    // Request line without a target.
    CHECK_FALSE(parse_http_request("GET\r\n\r\n").has_value());
    // Body shorter than the declared Content-Length (incomplete request).
    CHECK_FALSE(parse_http_request("POST / HTTP/1.1\r\n"
                                   "Content-Length: 10\r\n"
                                   "\r\n"
                                   "abc")
                    .has_value());
}

TEST_CASE("serializes a response with automatic Content-Length") {
    HttpResponse resp;
    resp.status = 200;
    resp.headers["Content-Type"] = "application/json";
    resp.body = "{\"ok\":true}";

    const auto wire = serialize_http_response(resp);

    CHECK(wire.find("HTTP/1.1 200 OK\r\n") == 0);
    CHECK(wire.find("Content-Type: application/json\r\n") != std::string::npos);
    CHECK(wire.find("Content-Length: 11\r\n") != std::string::npos);
    // Headers terminated by a blank line, then the body.
    CHECK(wire.find("\r\n\r\n{\"ok\":true}") != std::string::npos);
}

TEST_CASE("serializes common error status reasons") {
    HttpResponse resp;
    resp.status = 404;
    CHECK(serialize_http_response(resp).find("HTTP/1.1 404 Not Found\r\n") == 0);
    resp.status = 401;
    CHECK(serialize_http_response(resp).find("HTTP/1.1 401 Unauthorized\r\n") ==
          0);
    resp.status = 503;
    CHECK(serialize_http_response(resp).find(
              "HTTP/1.1 503 Service Unavailable\r\n") == 0);
}
