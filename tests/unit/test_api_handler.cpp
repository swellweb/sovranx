// Isolated tests for the API handler: HttpRequest in, HttpResponse/SSE out.
// No sockets; the engine runs over MockBackend with scripted logits.

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../mock/llama_mock.hpp"
#include "reame/server/api_handler.hpp"

using nlohmann::json;
using reame::TokenId;
using reame::test::MockBackend;
using reame::core::ReameEngine;
using reame::server::ApiHandler;
using reame::server::HeaderMap;
using reame::server::HttpRequest;
using reame::server::HttpResponse;
using reame::server::ResponseWriter;

namespace {

std::vector<float> peak(std::size_t n, std::size_t idx) {
    std::vector<float> v(n, 0.0f);
    v[idx] = 10.0f;
    return v;
}

// Prompt -> {1, 2}; generation: "foo" (3), "bar" (0), EOS (4).
void script_foobar(MockBackend* m) {
    m->vocab_size_value = 5;
    m->eos_token_value = 4;
    m->tokenize_result = {1, 2};
    m->piece_map = {{3, "foo"}, {0, "bar"}};
    m->decode_queue = {peak(5, 3), peak(5, 0), peak(5, 4)};
}

struct Fixture {
    std::unique_ptr<ReameEngine> engine;
    MockBackend* mock = nullptr;
    std::unique_ptr<ApiHandler> handler;

    explicit Fixture(ApiHandler::Config cfg = {}) {
        ReameEngine::Config ec;
        ec.model_path = "/models/test.gguf";
        ec.n_ctx = 2048;
        ec.n_threads = 4;
        auto backend = std::make_unique<MockBackend>();
        mock = backend.get();
        script_foobar(mock);
        engine = std::make_unique<ReameEngine>(ec, std::move(backend));
        handler = std::make_unique<ApiHandler>(cfg, *engine);
    }
};

HttpRequest request(const std::string& method, const std::string& target,
                    const std::string& body = "",
                    const HeaderMap& headers = {}) {
    HttpRequest r;
    r.method = method;
    r.target = target;
    r.body = body;
    r.headers = headers;
    return r;
}

// Greedy completion request body (deterministic against the script).
std::string completion_body(bool stream = false) {
    json j{{"prompt", "hi"}, {"temperature", 0.0}, {"max_tokens", 16}};
    if (stream) j["stream"] = true;
    return j.dump();
}

// Collects streaming output.
struct FakeWriter : ResponseWriter {
    int status = 0;
    HeaderMap headers;
    std::vector<std::string> chunks;
    bool ended = false;

    void begin(int s, const HeaderMap& h) override {
        status = s;
        headers = h;
    }
    void chunk(const std::string& d) override { chunks.push_back(d); }
    void end() override { ended = true; }
};

}  // namespace

// ---------------------------------------------------------------------------
// Basic endpoints
// ---------------------------------------------------------------------------

TEST_CASE("health returns ok") {
    Fixture f;
    const auto resp = f.handler->handle(request("GET", "/health"));

    CHECK(resp.status == 200);
    CHECK(json::parse(resp.body)["status"] == "ok");
}

TEST_CASE("models lists the configured model id") {
    ApiHandler::Config cfg;
    cfg.model_id = "reame-30b";
    Fixture f(cfg);

    const auto resp = f.handler->handle(request("GET", "/v1/models"));

    REQUIRE(resp.status == 200);
    const auto j = json::parse(resp.body);
    CHECK(j["object"] == "list");
    CHECK(j["data"][0]["id"] == "reame-30b");
}

TEST_CASE("unknown path is 404, wrong method 405, invalid JSON 400") {
    Fixture f;

    CHECK(f.handler->handle(request("GET", "/nope")).status == 404);
    CHECK(f.handler->handle(request("GET", "/v1/completions")).status == 405);

    const auto bad =
        f.handler->handle(request("POST", "/v1/completions", "{not json"));
    CHECK(bad.status == 400);
    CHECK(json::parse(bad.body)["error"]["type"] == "invalid_request_error");
}

TEST_CASE("completions requires a prompt") {
    Fixture f;
    const auto resp =
        f.handler->handle(request("POST", "/v1/completions", "{}"));
    CHECK(resp.status == 400);
}

// ---------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------

TEST_CASE("completions returns the generated text with usage") {
    Fixture f;
    const auto resp = f.handler->handle(
        request("POST", "/v1/completions", completion_body()));

    REQUIRE(resp.status == 200);
    const auto j = json::parse(resp.body);
    CHECK(j["object"] == "text_completion");
    CHECK(j["choices"][0]["text"] == "foobar");
    CHECK(j["choices"][0]["finish_reason"] == "stop");
    CHECK(j["usage"]["prompt_tokens"] == 2);      // tokenize -> {1, 2}
    CHECK(j["usage"]["completion_tokens"] == 2);  // "foo", "bar"
    CHECK(resp.header("Content-Type") == "application/json");
}

TEST_CASE("completions honours max_tokens from the request") {
    Fixture f;
    f.mock->decode_queue.clear();
    f.mock->decode_result = peak(5, 3);  // "foo" forever

    const json body{{"prompt", "hi"}, {"temperature", 0.0}, {"max_tokens", 3}};
    const auto resp =
        f.handler->handle(request("POST", "/v1/completions", body.dump()));

    REQUIRE(resp.status == 200);
    CHECK(json::parse(resp.body)["choices"][0]["text"] == "foofoofoo");
    CHECK(json::parse(resp.body)["choices"][0]["finish_reason"] == "length");
}

TEST_CASE("completions streams SSE chunks ending with [DONE]") {
    Fixture f;
    FakeWriter writer;
    f.handler->handle(
        request("POST", "/v1/completions", completion_body(/*stream=*/true)),
        writer);

    CHECK(writer.status == 200);
    CHECK(writer.headers.at("Content-Type") == "text/event-stream");
    REQUIRE(writer.ended);
    REQUIRE(writer.chunks.size() == 3);  // foo, bar, [DONE]

    const auto first = writer.chunks[0];
    REQUIRE(first.rfind("data: ", 0) == 0);
    const auto payload = json::parse(first.substr(6));
    CHECK(payload["choices"][0]["text"] == "foo");
    CHECK(writer.chunks.back() == "data: [DONE]\n\n");
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

TEST_CASE("chat completions apply the model's own chat template") {
    Fixture f;
    const json body{
        {"messages",
         json::array({{{"role", "system"}, {"content", "be brief"}},
                      {{"role", "user"}, {"content", "hi there"}}})},
        {"temperature", 0.0}};

    const auto resp =
        f.handler->handle(request("POST", "/v1/chat/completions", body.dump()));

    REQUIRE(resp.status == 200);
    const auto j = json::parse(resp.body);
    CHECK(j["object"] == "chat.completion");
    CHECK(j["choices"][0]["message"]["role"] == "assistant");
    CHECK(j["choices"][0]["message"]["content"] == "foobar");

    // The messages went through the backend's chat template — NOT a
    // hand-rolled "role: content" concatenation the model was never
    // trained on (that's what kept EOS from ever being emitted).
    REQUIRE(f.mock->format_chat_messages_calls.size() == 1);
    const auto& msgs = f.mock->format_chat_messages_calls[0];
    REQUIRE(msgs.size() == 2);
    CHECK(msgs[0].role == "system");
    CHECK(msgs[0].content == "be brief");
    CHECK(msgs[1].role == "user");
    CHECK(msgs[1].content == "hi there");

    // And the templated string is what got tokenized.
    REQUIRE_FALSE(f.mock->tokenize_calls.empty());
    CHECK(f.mock->tokenize_calls[0].first ==
          "<M:system>be brief<M:user>hi there<A>");
}

TEST_CASE("template-less models fall back to plain role-prefixed turns") {
    Fixture f;
    f.mock->chat_template_empty = true;
    const json body{
        {"messages",
         json::array({{{"role", "system"}, {"content", "be brief"}},
                      {{"role", "user"}, {"content", "hi there"}}})},
        {"temperature", 0.0}};

    const auto resp =
        f.handler->handle(request("POST", "/v1/chat/completions", body.dump()));

    REQUIRE(resp.status == 200);
    REQUIRE_FALSE(f.mock->tokenize_calls.empty());
    CHECK(f.mock->tokenize_calls[0].first ==
          "system: be brief\nuser: hi there\nassistant:");
}

TEST_CASE("chat completions require a non-empty messages array") {
    Fixture f;
    CHECK(f.handler
              ->handle(request("POST", "/v1/chat/completions",
                               json{{"messages", json::array()}}.dump()))
              .status == 400);
}

TEST_CASE("chat completions stream deltas") {
    Fixture f;
    const json body{{"messages", json::array({{{"role", "user"},
                                               {"content", "hi"}}})},
                    {"temperature", 0.0},
                    {"stream", true}};
    FakeWriter writer;
    f.handler->handle(request("POST", "/v1/chat/completions", body.dump()),
                      writer);

    REQUIRE(writer.chunks.size() == 3);
    const auto payload = json::parse(writer.chunks[0].substr(6));
    CHECK(payload["choices"][0]["delta"]["content"] == "foo");
    CHECK(writer.chunks.back() == "data: [DONE]\n\n");
}

// ---------------------------------------------------------------------------
// Authentication / CORS / limits
// ---------------------------------------------------------------------------

TEST_CASE("api key protects /v1 endpoints but not /health") {
    ApiHandler::Config cfg;
    cfg.api_key = "sk-secret";
    Fixture f(cfg);

    CHECK(f.handler->handle(request("GET", "/v1/models")).status == 401);
    CHECK(f.handler
              ->handle(request("GET", "/v1/models", "",
                               {{"Authorization", "Bearer wrong"}}))
              .status == 401);
    CHECK(f.handler
              ->handle(request("GET", "/v1/models", "",
                               {{"Authorization", "Bearer sk-secret"}}))
              .status == 200);
    CHECK(f.handler->handle(request("GET", "/health")).status == 200);
}

TEST_CASE("CORS headers and preflight when enabled, absent when disabled") {
    {
        Fixture f;  // enable_cors defaults to true
        const auto resp = f.handler->handle(request("GET", "/health"));
        CHECK(resp.header("Access-Control-Allow-Origin") == "*");

        const auto pre = f.handler->handle(request("OPTIONS", "/v1/models"));
        CHECK(pre.status == 204);
        CHECK(pre.header("Access-Control-Allow-Methods") != "");
    }
    {
        ApiHandler::Config cfg;
        cfg.enable_cors = false;
        Fixture f(cfg);
        CHECK(f.handler->handle(request("GET", "/health"))
                  .header("Access-Control-Allow-Origin") == "");
    }
}

TEST_CASE("generation endpoints return 503 when the server is saturated") {
    ApiHandler::Config cfg;
    cfg.max_concurrent_requests = 0;  // saturate immediately
    Fixture f(cfg);

    const auto resp = f.handler->handle(
        request("POST", "/v1/completions", completion_body()));
    CHECK(resp.status == 503);
    // Non-generation endpoints are unaffected.
    CHECK(f.handler->handle(request("GET", "/health")).status == 200);
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

TEST_CASE("malformed session paths get a clean 4xx, never a crash") {
    Fixture f;

    // No id at all (previously threw std::out_of_range -> 500).
    CHECK(f.handler->handle(request("DELETE", "/v1/sessions")).status == 405);
    CHECK(f.handler->handle(request("GET", "/v1/sessions")).status == 405);
    // Prefix-like but bogus.
    CHECK(f.handler->handle(request("POST", "/v1/sessionsXYZ")).status == 404);
    CHECK(f.handler->handle(request("POST", "/v1/sessions/")).status == 405);
}

TEST_CASE("session lifecycle over HTTP") {
    Fixture f;

    const auto created =
        f.handler->handle(request("POST", "/v1/sessions"));
    REQUIRE(created.status == 200);
    const std::string id = json::parse(created.body)["id"];
    CHECK(!id.empty());

    CHECK(f.handler->handle(request("POST", "/v1/sessions/" + id + "/save"))
              .status == 200);
    CHECK(f.handler->handle(request("POST", "/v1/sessions/" + id + "/load"))
              .status == 200);
    CHECK(f.handler->handle(request("DELETE", "/v1/sessions/" + id)).status ==
          200);
    // Deleted -> engine throws -> 404.
    CHECK(f.handler->handle(request("POST", "/v1/sessions/" + id + "/load"))
              .status == 404);
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

TEST_CASE("metrics report request counters when enabled, 404 when disabled") {
    {
        Fixture f;
        f.handler->handle(request("GET", "/health"));

        const auto resp = f.handler->handle(request("GET", "/metrics"));
        REQUIRE(resp.status == 200);
        const auto j = json::parse(resp.body);
        CHECK(j["server"]["total_requests"].get<int>() >= 1);
    }
    {
        ApiHandler::Config cfg;
        cfg.enable_metrics = false;
        Fixture f(cfg);
        CHECK(f.handler->handle(request("GET", "/metrics")).status == 404);
    }
}

TEST_CASE("engine errors surface as a 500 error envelope") {
    Fixture f;
    f.mock->fail_decodes = true;

    const auto resp = f.handler->handle(
        request("POST", "/v1/completions", completion_body()));

    CHECK(resp.status == 500);
    CHECK(json::parse(resp.body)["error"]["type"] == "server_error");
}
