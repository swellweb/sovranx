// Sovrano entry point. For this first step it only loads the configuration
// and sets up logging; engine/server wiring lands in later steps.

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "sovrano/core/engine.hpp"
#include "sovrano/utils/config.hpp"
#include "sovrano/utils/logger.hpp"

namespace {

constexpr const char* kVersion = "0.1.0";

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--config <path>] [--prompt <text>] [--max-tokens <n>]"
                 " [--help] [--version]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "config/sovrano.conf";
    std::string prompt;
    int max_tokens = 128;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg == "--version" || arg == "-v") {
            std::cout << "sovrano " << kVersion << "\n";
            return EXIT_SUCCESS;
        }
        if (arg == "--config" || arg == "-c" || arg == "--prompt" ||
            arg == "-p" || arg == "--max-tokens") {
            if (i + 1 >= argc) {
                std::cerr << "error: " << arg << " requires an argument\n";
                return EXIT_FAILURE;
            }
            const std::string value = argv[++i];
            if (arg == "--config" || arg == "-c") config_path = value;
            else if (arg == "--max-tokens") max_tokens = std::stoi(value);
            else prompt = value;
            continue;
        }
        std::cerr << "error: unknown argument '" << arg << "'\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = sovrano::Config::load(config_path);

        const auto level =
            sovrano::Logger::level_from_string(cfg.get_string("logging.level", "info"));
        sovrano::Logger log(std::cout, level);

        log.info("sovrano " + std::string(kVersion) + " starting");
        log.info("config loaded from " + config_path +
                 " (" + std::to_string(cfg.size()) + " keys)");
        log.info("model path: " + cfg.get_string("model.path", "<unset>"));
#ifdef SOVRANO_WITH_SERVER
        log.info("server: enabled (port " +
                 std::to_string(cfg.get_int("server.port", 8080)) + ")");
#else
        log.info("server: disabled in this build");
#endif

        if (prompt.empty()) {
            log.info("no --prompt given; exiting (HTTP server arrives in a "
                     "later step)");
            return EXIT_SUCCESS;
        }

        sovrano::core::SovranoEngine::Config engine_cfg;
        engine_cfg.model_path = cfg.get_string("model.path");
        engine_cfg.n_ctx =
            static_cast<int>(cfg.get_int("model.context_length", 4096));
        engine_cfg.n_threads = static_cast<int>(cfg.get_int("model.threads", 4));
        engine_cfg.use_mmap = cfg.get_bool("memory.use_mmap", true);
        engine_cfg.use_mlock = cfg.get_bool("memory.use_mlock", false);

        log.info("loading model...");
        sovrano::core::SovranoEngine engine(engine_cfg);
        log.info("model loaded (vocab " + std::to_string(engine.vocab_size()) +
                 ", ctx " + std::to_string(engine.context_size()) + ")");

        sovrano::core::GenerationConfig gen;
        gen.max_tokens = max_tokens;
        engine.generate_stream(prompt, [](const std::string& piece) {
            std::cout << piece << std::flush;
            return true;
        }, gen);
        std::cout << "\n";
        return EXIT_SUCCESS;
    } catch (const sovrano::core::EngineError& e) {
        std::cerr << "engine error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const sovrano::ConfigError& e) {
        std::cerr << "config error";
        if (e.line() != 0) std::cerr << " (line " << e.line() << ")";
        std::cerr << ": " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
