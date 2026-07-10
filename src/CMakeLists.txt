# Module libraries. core/speculative/memory/server get real sources in later
# steps; for now only utils has code.

add_library(sovrano_utils STATIC
    utils/config.cpp
    utils/logger.cpp
)
target_include_directories(sovrano_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sovrano_utils PUBLIC sovrano_flags Threads::Threads)

# Core: LlamaModel wrapper. llama.cpp is confined to llama_backend_real.cpp;
# without the submodule a stub backend keeps the build green.
add_library(sovrano_core STATIC
    core/model.cpp
    core/sampler.cpp
    core/engine.cpp
    core/scheduler.cpp
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
target_include_directories(sovrano_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sovrano_core
    PUBLIC sovrano_utils
    PRIVATE sovrano_flags
)

if(ZSTD_FOUND)
    target_link_libraries(sovrano_core PRIVATE PkgConfig::ZSTD)
    # PUBLIC: tests gate the zstd-specific expectations on this macro.
    target_compile_definitions(sovrano_core PUBLIC SOVRANO_HAS_ZSTD=1)
endif()

if(SOVRANO_HAS_LLAMA)
    target_sources(sovrano_core PRIVATE core/llama_backend_real.cpp)
    target_compile_definitions(sovrano_core PUBLIC SOVRANO_HAS_LLAMA=1)
    # llama target only (PRIVATE): its headers never leak past the backend TU.
    target_link_libraries(sovrano_core PRIVATE llama)
    # llama.h is a system include so -Wconversion & co. don't fire on it.
    target_include_directories(sovrano_core SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    )
else()
    target_sources(sovrano_core PRIVATE core/llama_backend_stub.cpp)
endif()

if(SOVRANO_BUILD_SERVER)
    add_library(sovrano_server STATIC
        server/http_parser.cpp
        server/api_handler.cpp
        server/http_server.cpp
    )
    target_include_directories(sovrano_server PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(sovrano_server
        PUBLIC sovrano_core nlohmann_json::nlohmann_json
        PRIVATE sovrano_flags Boost::headers Threads::Threads
    )
    target_compile_definitions(sovrano_server PUBLIC SOVRANO_WITH_SERVER=1)
endif()

add_executable(sovrano main.cpp)
target_link_libraries(sovrano PRIVATE sovrano_core sovrano_utils sovrano_flags)

if(SOVRANO_BUILD_SERVER)
    target_link_libraries(sovrano PRIVATE sovrano_server)
endif()
