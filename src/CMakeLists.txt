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
    speculative/acceptance.cpp
    speculative/draft_generator.cpp
    speculative/batch_verifier.cpp
    speculative/speculative_decoder.cpp
    cache/cache_serializer.cpp
    cache/disk_cache_store.cpp
    cache/cache_manager.cpp
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

add_executable(sovrano main.cpp)
target_link_libraries(sovrano PRIVATE sovrano_core sovrano_utils sovrano_flags)

if(SOVRANO_BUILD_SERVER)
    target_link_libraries(sovrano PRIVATE
        Boost::system
        nlohmann_json::nlohmann_json
        PkgConfig::ZSTD
    )
    target_compile_definitions(sovrano PRIVATE SOVRANO_WITH_SERVER=1)
endif()
