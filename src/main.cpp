// Reame entry point. For this first step it only loads the configuration
// and sets up logging; engine/server wiring lands in later steps.

#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "reame/cache/cache_manager.hpp"
#include "reame/core/autoconfig.hpp"
#include "reame/core/engine.hpp"
#include "reame/speculative/speculative_decoder.hpp"
#include "reame/utils/config.hpp"
#include "reame/utils/logger.hpp"

#include <cstdio>
#include <filesystem>
#include <thread>

#ifdef REAME_WITH_SERVER
#include <condition_variable>
#include <mutex>

#include "reame/server/http_server.hpp"
#endif

namespace {

constexpr const char* kVersion = "0.1.3";

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " run <model> [\"prompt\"] [--serve]\n"
        << "      zero-config: download if needed, auto-tune, then chat/serve\n"
        << "  " << argv0 << " list\n"
        << "      show the built-in model catalog\n"
        << "  " << argv0
        << " --config <path> [--prompt <text>] [--max-tokens <n>] [--serve]\n"
        << "      advanced: run from a config file\n";
}

#ifdef REAME_WITH_SERVER
std::condition_variable g_shutdown_cv;
std::mutex g_shutdown_mutex;
bool g_shutdown = false;

void request_shutdown(int) {
    {
        std::lock_guard<std::mutex> lock(g_shutdown_mutex);
        g_shutdown = true;
    }
    g_shutdown_cv.notify_all();
}
#endif

std::string home_dir() {
    if (const char* h = std::getenv("HOME")) return h;
    return "";
}

// Downloads `url` to `dest` with curl if the file is not already there.
// Returns true on success.
bool ensure_downloaded(const std::string& url, const std::string& dest,
                       reame::Logger& log) {
    if (std::filesystem::exists(dest)) return true;
    std::filesystem::create_directories(
        std::filesystem::path(dest).parent_path());
    log.info("downloading model (first run only)...");
    const std::string tmp = dest + ".part";
    const std::string cmd =
        "curl -L -C - --fail --progress-bar -o '" + tmp + "' '" + url + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::filesystem::remove(tmp);
        log.error("download failed");
        return false;
    }
    std::filesystem::rename(tmp, dest);
    return true;
}

// `reame run <model> ["prompt"] [--serve]`.
int run_zeroconfig(int argc, char** argv) {
    reame::Logger log(std::cout, reame::LogLevel::Info);
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " run <model> [\"prompt\"]\n";
        return EXIT_FAILURE;
    }
    const std::string token = argv[2];
    std::string prompt;
    bool serve = false;
    int max_tokens = 512;
    float temperature = 0.7f;
    int best_of = 1;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--serve") {
            serve = true;
        } else if ((a == "--max-tokens" || a == "--temp" ||
                    a == "--best-of") &&
                   i + 1 < argc) {
            const std::string v = argv[++i];
            if (a == "--max-tokens") max_tokens = std::stoi(v);
            else if (a == "--temp") temperature = std::stof(v);
            else best_of = std::stoi(v);
        } else if (prompt.empty()) {
            prompt = a;
        }
    }

    // Resolve: a catalog alias downloads to ~/.reame/models; anything
    // else is treated as a local GGUF path.
    std::string model_path;
    const auto spec = reame::core::resolve_model(token);
    if (spec.has_value()) {
        const std::string dir = home_dir() + "/.reame/models";
        model_path = dir + "/" + spec->filename;
        if (!ensure_downloaded(spec->url, model_path, log))
            return EXIT_FAILURE;
    } else if (std::filesystem::exists(token)) {
        model_path = token;
    } else {
        std::cerr << "unknown model '" << token
                  << "' and no such file. Try: " << argv[0] << " list\n";
        return EXIT_FAILURE;
    }

    const unsigned hw = std::thread::hardware_concurrency();
    auto cfg = reame::core::auto_config(
        model_path, home_dir(), hw, spec.has_value() ? spec->default_ctx : 0);
    if (best_of > 1) {
        // Conclave: interleaved candidates; scale the TOTAL KV budget so
        // each keeps the full per-request context (parallel mode excludes
        // speculation and the disk cache).
        cfg.n_parallel = best_of;
        cfg.n_ctx *= best_of;
        cfg.use_speculative = false;
        cfg.use_prompt_lookup = false;
        cfg.cache_dir.clear();
    }

    try {
        log.info("loading model (" + std::to_string(cfg.n_threads) +
                 " threads)...");
        auto engine =
            std::make_shared<reame::core::ReameEngine>(cfg);
        log.info("ready.");

        if (serve) {
#ifdef REAME_WITH_SERVER
            reame::server::HttpServer::Config sc;
            sc.port = 8080;
            reame::server::HttpServer server(sc, engine);
            server.start();
            log.info("serving on http://127.0.0.1:8080 (Ctrl-C to stop)");
            std::signal(SIGINT, request_shutdown);
            std::signal(SIGTERM, request_shutdown);
            std::unique_lock<std::mutex> lock(g_shutdown_mutex);
            g_shutdown_cv.wait(lock, [] { return g_shutdown; });
            server.stop();
            return EXIT_SUCCESS;
#else
            std::cerr << "this build has no server support\n";
            return EXIT_FAILURE;
#endif
        }

        reame::core::GenerationConfig gen;
        gen.max_tokens = max_tokens;
        gen.temperature = temperature;
        const auto stream = [](const std::string& piece) {
            std::cout << piece << std::flush;
            return true;
        };

        if (!prompt.empty()) {
            // The model's own chat template (from the GGUF); raw
            // completion for template-less models.
            const auto chat = engine->format_chat(prompt);
            if (best_of > 1) {
                int votes = 0;
                std::cout << engine->generate_best(chat, gen, best_of,
                                                   &votes)
                          << "\n";
                std::cerr << "CONCLAVE consensus=" << votes << "/" << best_of
                          << "\n";
            } else {
                engine->generate_stream(chat, stream, gen);
                std::cout << "\n";
            }
            return EXIT_SUCCESS;
        }

        // Interactive chat: read lines, answer each.
        std::cout << "Chatting with " << token
                  << ". Type a message, Ctrl-D to quit.\n";
        std::string line;
        while (true) {
            std::cout << "\n> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            engine->generate_stream(engine->format_chat(line), stream, gen);
            std::cout << "\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        log.error(std::string("fatal: ") + e.what());
        return EXIT_FAILURE;
    }
}

int list_models() {
    std::cout << "Built-in models (reame run <name>):\n\n";
    for (const auto& m : reame::core::model_catalog())
        std::cout << "  " << m.name << "\n      " << m.note << "\n";
    std::cout << "\nOr: reame run /path/to/your-model.gguf\n";
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    // Zero-config subcommands come first: `reame run <model>` / `list`.
    if (argc >= 2) {
        const std::string sub = argv[1];
        if (sub == "run") return run_zeroconfig(argc, argv);
        if (sub == "list") return list_models();
    }

    std::string config_path = "config/reame.conf";
    std::string prompt;
    int max_tokens = 128;
    float temperature = 0.7f;
    int best_of = 1;
    bool serve = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg == "--serve" || arg == "-s") {
            serve = true;
            continue;
        }
        if (arg == "--version" || arg == "-v") {
            std::cout << "reame " << kVersion << "\n";
            return EXIT_SUCCESS;
        }
        if (arg == "--config" || arg == "-c" || arg == "--prompt" ||
            arg == "-p" || arg == "--max-tokens" || arg == "--temp" ||
            arg == "--best-of") {
            if (i + 1 >= argc) {
                std::cerr << "error: " << arg << " requires an argument\n";
                return EXIT_FAILURE;
            }
            const std::string value = argv[++i];
            if (arg == "--config" || arg == "-c") config_path = value;
            else if (arg == "--max-tokens") max_tokens = std::stoi(value);
            else if (arg == "--temp") temperature = std::stof(value);
            else if (arg == "--best-of") best_of = std::stoi(value);
            else prompt = value;
            continue;
        }
        std::cerr << "error: unknown argument '" << arg << "'\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = reame::Config::load(config_path);

        const auto level =
            reame::Logger::level_from_string(cfg.get_string("logging.level", "info"));
        reame::Logger log(std::cout, level);

        log.info("reame " + std::string(kVersion) + " starting");
        log.info("config loaded from " + config_path +
                 " (" + std::to_string(cfg.size()) + " keys)");
        log.info("model path: " + cfg.get_string("model.path", "<unset>"));
#ifdef REAME_WITH_SERVER
        log.info("server: enabled (port " +
                 std::to_string(cfg.get_int("server.port", 8080)) + ")");
#else
        log.info("server: disabled in this build");
#endif

        if (prompt.empty() && !serve) {
            log.info("nothing to do: pass --prompt <text> or --serve");
            return EXIT_SUCCESS;
        }

        reame::core::ReameEngine::Config engine_cfg;
        engine_cfg.model_path = cfg.get_string("model.path");
        engine_cfg.n_ctx =
            static_cast<int>(cfg.get_int("model.context_length", 4096));
        engine_cfg.n_threads = static_cast<int>(cfg.get_int("model.threads", 4));
        engine_cfg.use_mmap = cfg.get_bool("memory.use_mmap", true);
        engine_cfg.use_mlock = cfg.get_bool("memory.use_mlock", false);
        engine_cfg.kv_cache_type =
            cfg.get_string("memory.kv_cache_type", "f16");
        engine_cfg.n_ubatch =
            static_cast<int>(cfg.get_int("model.ubatch", 0));
        engine_cfg.use_speculative = cfg.get_bool("speculative.enabled", true);
        // mode: model (default, needs draft_model_path) | lookup (n-gram
        // proposals from the prompt itself, no second model)
        engine_cfg.use_prompt_lookup =
            cfg.get_string("speculative.mode", "model") == "lookup";
        engine_cfg.draft_model_path =
            cfg.get_string("speculative.draft_model_path", "");
        engine_cfg.draft_tokens =
            static_cast<int>(cfg.get_int("speculative.draft_tokens", 16));
        engine_cfg.cache_dir = cfg.get_string("cache.directory", "");
        engine_cfg.cache_max_mb = static_cast<std::uint64_t>(
            cfg.get_int("cache.max_size_mb", 512));
        engine_cfg.cache_compress = cfg.get_bool("cache.compress", true);
        engine_cfg.cache_block_tokens =
            static_cast<int>(cfg.get_int("cache.block_tokens", 256));
        engine_cfg.n_parallel =
            static_cast<int>(cfg.get_int("server.parallel", 1));
        if (best_of > 1) {
            // Conclave from the CLI: attempts run interleaved; parallel
            // mode excludes the disk cache and speculation. n_ctx is the
            // TOTAL KV budget across sequences — scale it so every
            // candidate keeps the configured per-request context.
            engine_cfg.n_parallel = best_of;
            engine_cfg.n_ctx *= best_of;
            engine_cfg.use_speculative = false;
            engine_cfg.use_prompt_lookup = false;
            engine_cfg.cache_dir.clear();
        }

        // Fail fast on missing model files (main AND draft) with a clear
        // message, before llama.cpp can surface a bare strerror.
        if (const auto missing =
                reame::core::missing_model_file_error(engine_cfg);
            !missing.empty()) {
            log.error(missing);
            return EXIT_FAILURE;
        }

        log.info("loading model...");
        auto engine =
            std::make_shared<reame::core::ReameEngine>(engine_cfg);
        log.info("model loaded (vocab " +
                 std::to_string(engine->vocab_size()) + ", ctx " +
                 std::to_string(engine->context_size()) + ")");
        if (engine->speculative_metrics() != nullptr)
            log.info("speculative decoding: on (" +
                     (engine_cfg.use_prompt_lookup
                          ? std::string("prompt-lookup n-grams, no draft model")
                          : "draft " + engine_cfg.draft_model_path) +
                     ")");

        if (serve) {
#ifdef REAME_WITH_SERVER
            reame::server::HttpServer::Config sc;
            sc.host = cfg.get_string("server.host", "0.0.0.0");
            sc.port = static_cast<int>(cfg.get_int("server.port", 8080));
            sc.threads =
                static_cast<int>(cfg.get_int("server.threads", 2));
            sc.max_concurrent_requests = static_cast<int>(
                cfg.get_int("server.max_concurrent_requests", 10));
            sc.timeout_seconds = static_cast<int>(
                cfg.get_int("server.timeout_seconds", 300));
            sc.max_request_size_mb = static_cast<std::size_t>(
                cfg.get_int("server.max_request_size_mb", 10));
            sc.enable_cors = cfg.get_bool("server.enable_cors", true);
            sc.enable_metrics = cfg.get_bool("server.enable_metrics", true);
            sc.enable_request_logging =
                cfg.get_bool("server.enable_request_logging", true);
            sc.api_key = cfg.get_string("server.api_key", "");
            sc.model_id = cfg.get_string("server.model_id", "reame");

            reame::server::HttpServer server(sc, engine);
            server.start();

            std::signal(SIGINT, request_shutdown);
            std::signal(SIGTERM, request_shutdown);
            {
                std::unique_lock<std::mutex> lock(g_shutdown_mutex);
                g_shutdown_cv.wait(lock, [] { return g_shutdown; });
            }
            log.info("shutting down");
            server.stop();
            return EXIT_SUCCESS;
#else
            std::cerr << "error: reame was built without server support "
                         "(Boost/nlohmann-json missing at configure time)\n";
            return EXIT_FAILURE;
#endif
        }

        reame::core::GenerationConfig gen;
        gen.max_tokens = max_tokens;
        gen.temperature = temperature;
        if (best_of > 1) {
            int votes = 0;
            std::cout << engine->generate_best(prompt, gen, best_of, &votes)
                      << "\n";
            // Agreement strength on stderr: lets a caller escalate (e.g.
            // retry with a reasoning prompt) only when the conclave split.
            std::cerr << "CONCLAVE consensus=" << votes << "/" << best_of
                      << "\n";
        } else {
            engine->generate_stream(prompt, [](const std::string& piece) {
                std::cout << piece << std::flush;
                return true;
            }, gen);
            std::cout << "\n";
        }

        if (const auto* c = engine->cache_stats()) {
            log.info("cache: " + std::to_string(c->hits) + " hits, " +
                     std::to_string(c->misses) + " misses, " +
                     std::to_string(c->stores) + " stores");
        }
        if (const auto* m = engine->speculative_metrics()) {
            log.info("speculative metrics: acceptance " +
                     std::to_string(m->acceptance_rate()) + ", overall " +
                     std::to_string(m->overall_speed()) + " tok/s (draft " +
                     std::to_string(m->total_draft_tokens) + ", accepted " +
                     std::to_string(m->total_accepted_tokens) + ", rejected " +
                     std::to_string(m->total_rejected_tokens) + ")");
        }
        return EXIT_SUCCESS;
    } catch (const reame::core::EngineError& e) {
        std::cerr << "engine error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const reame::ConfigError& e) {
        std::cerr << "config error";
        if (e.line() != 0) std::cerr << " (line " << e.line() << ")";
        std::cerr << ": " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
