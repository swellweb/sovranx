# Module libraries. core/speculative/memory/server get real sources in later
# steps; for now only utils has code.

add_library(reame_utils STATIC
    utils/config.cpp
    utils/logger.cpp
)
target_include_directories(reame_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(reame_utils PUBLIC reame_flags Threads::Threads)

# Core: LlamaModel wrapper. llama.cpp is confined to llama_backend_real.cpp;
# without the submodule a stub backend keeps the build green.
add_library(reame_core STATIC
    core/model.cpp
    core/sampler.cpp
    core/engine.cpp
    core/scheduler.cpp
    core/autoconfig.cpp
    core/conclave.cpp
    arca/resp.cpp
    arca/archive.cpp
    speculative/acceptance.cpp
    speculative/draft_generator.cpp
    speculative/batch_verifier.cpp
    speculative/speculative_decoder.cpp
    speculative/prompt_lookup.cpp
    speculative/form_draft.cpp
    cache/cache_serializer.cpp
    cache/disk_cache_store.cpp
    cache/cache_manager.cpp
    cache/prefix_cache.cpp
    palimpsest/corpus_index.cpp
)
target_include_directories(reame_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(reame_core
    PUBLIC reame_utils
    PRIVATE reame_flags
)

if(ZSTD_FOUND)
    target_link_libraries(reame_core PRIVATE PkgConfig::ZSTD)
    # PUBLIC: tests gate the zstd-specific expectations on this macro.
    target_compile_definitions(reame_core PUBLIC REAME_HAS_ZSTD=1)
endif()

if(REAME_HAS_LLAMA)
    target_sources(reame_core PRIVATE core/llama_backend_real.cpp)
    target_compile_definitions(reame_core PUBLIC REAME_HAS_LLAMA=1)
    # llama target only (PRIVATE): its headers never leak past the backend TU.
    target_link_libraries(reame_core PRIVATE llama)
    # llama.h is a system include so -Wconversion & co. don't fire on it.
    target_include_directories(reame_core SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    )
else()
    target_sources(reame_core PRIVATE core/llama_backend_stub.cpp)
endif()

if(REAME_BUILD_SERVER)
    add_library(reame_server STATIC
        server/http_parser.cpp
        server/api_handler.cpp
        server/http_server.cpp
        arca/server.cpp
    )
    target_include_directories(reame_server PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(reame_server
        PUBLIC reame_core nlohmann_json::nlohmann_json
        PRIVATE reame_flags Boost::headers Threads::Threads
    )
    target_compile_definitions(reame_server PUBLIC REAME_WITH_SERVER=1)
endif()

add_executable(reame main.cpp)
target_link_libraries(reame PRIVATE reame_core reame_utils reame_flags)

if(REAME_BUILD_SERVER)
    target_link_libraries(reame PRIVATE reame_server)
endif()
