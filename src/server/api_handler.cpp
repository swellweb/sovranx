#include "reame/server/api_handler.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <utility>

#include "reame/speculative/speculative_decoder.hpp"

namespace reame::server {

using nlohmann::json;

namespace {

json error_json(const std::string& message, const std::string& type,
                const std::string& code) {
    return json{{"error",
                 {{"message", message}, {"type", type}, {"code", code}}}};
}

// RAII slot in the generation-concurrency gate.
class BusyGuard {
public:
    BusyGuard(std::atomic<int>& active, int max)
        : active_(active), acquired_(false) {
        const int current = active_.fetch_add(1);
        if (current >= max) {
            active_.fetch_sub(1);
            return;
        }
        acquired_ = true;
    }
    ~BusyGuard() {
        if (acquired_) active_.fetch_sub(1);
    }
    bool acquired() const { return acquired_; }

private:
    std::atomic<int>& active_;
    bool acquired_;
};

// Buffers the writer protocol into a plain HttpResponse.
class BufferingWriter final : public ResponseWriter {
public:
    void begin(int status, const HeaderMap& headers) override {
        response_.status = status;
        response_.headers = headers;
    }
    void chunk(const std::string& data) override { response_.body += data; }
    void end() override {}
    HttpResponse take() { return std::move(response_); }

private:
    HttpResponse response_;
};

}  // namespace

struct ApiHandler::Impl {
    Config cfg;
    core::ReameEngine& engine;
    // The engine is not thread-safe; every engine call is serialized.
    std::mutex engine_mutex;
    std::atomic<int> active_generations{0};
    std::atomic<std::uint64_t> total_requests{0};
    std::atomic<std::uint64_t> errors{0};

    Impl(const Config& c, core::ReameEngine& e) : cfg(c), engine(e) {}

    // ---- Helpers -----------------------------------------------------

    HeaderMap base_headers(const std::string& content_type) const {
        HeaderMap h;
        h["Content-Type"] = content_type;
        if (cfg.enable_cors) {
            h["Access-Control-Allow-Origin"] = "*";
            h["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS";
            h["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        }
        return h;
    }

    void respond(ResponseWriter& w, int status, const json& body) {
        if (status >= 400) ++errors;
        w.begin(status, base_headers("application/json"));
        w.chunk(body.dump());
        w.end();
    }

    bool authorized(const HttpRequest& r) const {
        if (cfg.api_key.empty()) return true;
        const std::string expected = "Bearer " + cfg.api_key;
        const std::string got = r.header("Authorization");
        // Constant-time comparison: a byte-wise early exit would leak how
        // many leading characters of the key are correct.
        unsigned char diff =
            static_cast<unsigned char>(got.size() != expected.size());
        for (std::size_t i = 0; i < got.size(); ++i)
            diff |= static_cast<unsigned char>(
                got[i] ^ expected[i < expected.size() ? i : 0]);
        return diff == 0;
    }

    static core::GenerationConfig gen_config_from(const json& j) {
        core::GenerationConfig gen;
        gen.max_tokens = j.value("max_tokens", 512);
        gen.temperature = j.value("temperature", 0.7f);
        gen.top_p = j.value("top_p", 0.95f);
        gen.repeat_penalty = j.value("repeat_penalty", 1.1f);
        gen.seed = j.value("seed", 42);
        gen.echo_prompt = j.value("echo", false);
        return gen;
    }

    // ---- Generation core (shared by completions and chat) -------------

    struct GenerationOutcome {
        std::string text;
        int completion_tokens = 0;
        int prompt_tokens = 0;
        bool hit_max_tokens = false;
    };

    // `on_piece`: non-null for streaming (already-formatted SSE emission is
    // the caller's job). Throws EngineError upward.
    GenerationOutcome run_generation(
        const std::string& prompt, const core::GenerationConfig& gen,
        const std::function<void(const std::string&)>& on_piece) {
        // A parallel-capable engine interleaves concurrent generations
        // internally; serializing here would defeat it.
        std::unique_lock<std::mutex> lock(engine_mutex, std::defer_lock);
        if (!engine.parallel_capable()) lock.lock();
        GenerationOutcome out;
        out.prompt_tokens = engine.count_tokens(prompt);
        engine.generate_stream(
            prompt,
            [&](const std::string& piece) {
                out.text += piece;
                ++out.completion_tokens;
                if (on_piece) on_piece(piece);
                return true;
            },
            gen);
        out.hit_max_tokens = out.completion_tokens >= gen.max_tokens;
        return out;
    }

    void handle_generation(const HttpRequest& r, ResponseWriter& w,
                           bool chat) {
        json body;
        try {
            body = json::parse(r.body);
        } catch (const json::exception&) {
            respond(w, 400,
                    error_json("request body is not valid JSON",
                               "invalid_request_error", "invalid_json"));
            return;
        }

        std::string prompt;
        if (chat) {
            if (!body.contains("messages") || !body["messages"].is_array() ||
                body["messages"].empty()) {
                respond(w, 400,
                        error_json("'messages' must be a non-empty array",
                                   "invalid_request_error", "invalid_messages"));
                return;
            }
            std::vector<ChatMessage> messages;
            messages.reserve(body["messages"].size());
            for (const auto& m : body["messages"])
                messages.push_back(
                    {m.value("role", "user"), m.value("content", "")});
            prompt = engine.format_chat(messages);
        } else {
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                respond(w, 400,
                        error_json("'prompt' (string) is required",
                                   "invalid_request_error", "missing_prompt"));
                return;
            }
            prompt = body["prompt"];
        }

        BusyGuard guard(active_generations, cfg.max_concurrent_requests);
        if (!guard.acquired()) {
            respond(w, 503,
                    error_json("server is at capacity, retry later",
                               "server_error", "overloaded"));
            return;
        }

        const auto gen = gen_config_from(body);
        const bool stream = body.value("stream", false);
        const char* object = chat ? "chat.completion" : "text_completion";

        try {
            if (stream) {
                auto headers = base_headers("text/event-stream");
                headers["Cache-Control"] = "no-cache";
                w.begin(200, headers);
                run_generation(prompt, gen, [&](const std::string& piece) {
                    json chunk;
                    chunk["object"] = chat ? "chat.completion.chunk"
                                           : "text_completion.chunk";
                    chunk["model"] = cfg.model_id;
                    if (chat)
                        chunk["choices"] =
                            json::array({{{"index", 0},
                                          {"delta", {{"content", piece}}}}});
                    else
                        chunk["choices"] = json::array(
                            {{{"index", 0}, {"text", piece}}});
                    w.chunk("data: " + chunk.dump() + "\n\n");
                });
                w.chunk("data: [DONE]\n\n");
                w.end();
                return;
            }

            const auto out = run_generation(prompt, gen, nullptr);
            json resp;
            resp["object"] = object;
            resp["model"] = cfg.model_id;
            const std::string finish =
                out.hit_max_tokens ? "length" : "stop";
            if (chat)
                resp["choices"] = json::array(
                    {{{"index", 0},
                      {"message",
                       {{"role", "assistant"}, {"content", out.text}}},
                      {"finish_reason", finish}}});
            else
                resp["choices"] = json::array({{{"index", 0},
                                                {"text", out.text},
                                                {"finish_reason", finish}}});
            resp["usage"] = {
                {"prompt_tokens", out.prompt_tokens},
                {"completion_tokens", out.completion_tokens},
                {"total_tokens", out.prompt_tokens + out.completion_tokens}};
            respond(w, 200, resp);
        } catch (const core::EngineError& e) {
            respond(w, 500,
                    error_json(e.what(), "server_error", "generation_failed"));
        } catch (const std::exception& e) {
            respond(w, 500,
                    error_json(e.what(), "server_error", "internal"));
        }
    }

    // ---- Sessions ------------------------------------------------------

    void handle_sessions(const HttpRequest& r, ResponseWriter& w) {
        std::lock_guard<std::mutex> lock(engine_mutex);
        try {
            if (r.target == "/v1/sessions" && r.method == "POST") {
                respond(w, 200, json{{"id", engine.create_session()}});
                return;
            }

            // /v1/sessions/{id}[/save|/load] — anything without an id
            // (bare "/v1/sessions" with the wrong method) is not a valid
            // session operation.
            if (r.target.size() <= 13) {
                respond(w, 405,
                        error_json("method not allowed",
                                   "invalid_request_error", "bad_method"));
                return;
            }
            const std::string rest = r.target.substr(13);  // after prefix
            const auto slash = rest.find('/');
            const std::string id =
                slash == std::string::npos ? rest : rest.substr(0, slash);
            const std::string action =
                slash == std::string::npos ? "" : rest.substr(slash + 1);

            if (r.method == "DELETE" && action.empty()) {
                engine.delete_session(id);
                respond(w, 200, json{{"ok", true}});
            } else if (r.method == "POST" && action == "save") {
                engine.save_session(id);
                respond(w, 200, json{{"ok", true}});
            } else if (r.method == "POST" && action == "load") {
                engine.load_session(id);
                respond(w, 200, json{{"ok", true}});
            } else {
                respond(w, 405,
                        error_json("method not allowed",
                                   "invalid_request_error", "bad_method"));
            }
        } catch (const core::EngineError& e) {
            respond(w, 404,
                    error_json(e.what(), "invalid_request_error",
                               "session_not_found"));
        }
    }

    // ---- Metrics ---------------------------------------------------------

    void handle_metrics(ResponseWriter& w) {
        json j;
        j["server"] = {{"total_requests", total_requests.load()},
                       {"errors", errors.load()},
                       {"active_generations", active_generations.load()}};
        j["model"] = {{"id", cfg.model_id},
                      {"context_size", engine.context_size()},
                      {"vocab_size", engine.vocab_size()}};
        if (const auto* m = engine.speculative_metrics()) {
            j["speculative"] = {
                {"acceptance_rate", m->acceptance_rate()},
                {"total_draft_tokens", m->total_draft_tokens},
                {"total_accepted_tokens", m->total_accepted_tokens},
                {"total_rejected_tokens", m->total_rejected_tokens},
                {"draft_speed_tps", m->draft_speed()},
                {"target_speed_tps", m->target_speed()},
                {"overall_speed_tps", m->overall_speed()}};
        }
        respond(w, 200, j);
    }

    // ---- Router ----------------------------------------------------------

    void route(const HttpRequest& r, ResponseWriter& w) {
        ++total_requests;

        if (cfg.enable_cors && r.method == "OPTIONS") {
            w.begin(204, base_headers("text/plain"));
            w.end();
            return;
        }

        if (r.target == "/health") {
            respond(w, 200, json{{"status", "ok"}});
            return;
        }

        if (!authorized(r)) {
            respond(w, 401,
                    error_json("missing or invalid API key",
                               "invalid_request_error", "invalid_api_key"));
            return;
        }

        if (r.target == "/metrics") {
            if (!cfg.enable_metrics) {
                respond(w, 404,
                        error_json("metrics are disabled",
                                   "invalid_request_error", "not_found"));
                return;
            }
            handle_metrics(w);
            return;
        }

        if (r.target == "/v1/models") {
            if (r.method != "GET") {
                respond(w, 405,
                        error_json("method not allowed",
                                   "invalid_request_error", "bad_method"));
                return;
            }
            respond(w, 200,
                    json{{"object", "list"},
                         {"data", json::array({{{"id", cfg.model_id},
                                                {"object", "model"},
                                                {"owned_by", "reame"}}})}});
            return;
        }

        if (r.target == "/v1/completions" ||
            r.target == "/v1/chat/completions") {
            if (r.method != "POST") {
                respond(w, 405,
                        error_json("method not allowed",
                                   "invalid_request_error", "bad_method"));
                return;
            }
            handle_generation(r, w, r.target == "/v1/chat/completions");
            return;
        }

        if (r.target == "/v1/sessions" ||
            r.target.rfind("/v1/sessions/", 0) == 0) {
            handle_sessions(r, w);
            return;
        }

        respond(w, 404,
                error_json("unknown endpoint: " + r.target,
                           "invalid_request_error", "not_found"));
    }
};

ApiHandler::ApiHandler(const Config& cfg, core::ReameEngine& engine)
    : pimpl_(std::make_unique<Impl>(cfg, engine)) {}

ApiHandler::~ApiHandler() = default;

void ApiHandler::handle(const HttpRequest& request, ResponseWriter& writer) {
    pimpl_->route(request, writer);
}

HttpResponse ApiHandler::handle(const HttpRequest& request) {
    BufferingWriter writer;
    pimpl_->route(request, writer);
    return writer.take();
}

ApiHandler::Stats ApiHandler::stats() const {
    return {pimpl_->total_requests.load(), pimpl_->errors.load()};
}

}  // namespace reame::server
