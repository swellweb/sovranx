cmake_minimum_required(VERSION 3.16)

project(reame
    VERSION 0.1.4
    DESCRIPTION "Reame - CPU inference engine for GGUF LLMs (speculative decoding + memory optimizations)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(REAME_BUILD_TESTS   "Build unit tests"                     ON)
option(REAME_BUILD_SERVER  "Build HTTP server (needs Boost.Asio)" ON)
option(REAME_ENABLE_AVX2   "Enable AVX2 codegen (x86_64 only)"    ON)
option(REAME_ENABLE_AVX512 "Enable AVX-512 codegen (x86_64 only)" OFF)
option(REAME_WERROR        "Treat warnings as errors"             OFF)

include(cmake/CompilerFlags.cmake)

# ---------------------------------------------------------------------------
# llama.cpp (submodule). If missing, the build proceeds without it: the
# core library falls back to a stub backend that errors at model load.
# ---------------------------------------------------------------------------
set(REAME_HAS_LLAMA OFF)
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(REAME_HAS_LLAMA ON)
    # CPU-only engine: no GPU backends, no extra tooling.
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
    set(GGML_METAL           OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA            OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN          OFF CACHE BOOL "" FORCE)
    # ggml handles its own SIMD flags (AVX2/AVX512 on x86_64, NEON on arm64);
    # REAME_ENABLE_AVX* only affects Reame's own code.
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
else()
    message(WARNING
        "third_party/llama.cpp not found - building WITHOUT inference backend.\n"
        "Initialize the submodule with:\n"
        "  git submodule update --init --depth 1 third_party/llama.cpp")
endif()

# ---------------------------------------------------------------------------
# Dependencies. zstd is used by the KV-cache persistence (optional: without
# it, cache entries are stored uncompressed); Boost/json are server-only.
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD QUIET IMPORTED_TARGET libzstd)
endif()
if(NOT ZSTD_FOUND)
    message(STATUS "zstd not found - KV cache will be stored uncompressed")
endif()

if(REAME_BUILD_SERVER)
    # Boost.Asio/System are header-only in modern Boost: headers suffice.
    find_package(Boost 1.74 QUIET CONFIG)
    find_package(nlohmann_json 3.10 QUIET)

    if(NOT Boost_FOUND OR NOT nlohmann_json_FOUND)
        message(WARNING
            "Server dependencies missing (Boost: ${Boost_FOUND}, "
            "nlohmann_json: ${nlohmann_json_FOUND}). "
            "Server target disabled for this build.")
        set(REAME_BUILD_SERVER OFF)
    endif()
endif()

add_subdirectory(src)

if(REAME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "-----------------------------------------------")
message(STATUS "Reame ${PROJECT_VERSION}")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  llama.cpp  : ${REAME_HAS_LLAMA}")
message(STATUS "  Tests      : ${REAME_BUILD_TESTS}")
message(STATUS "  Server     : ${REAME_BUILD_SERVER}")
message(STATUS "  AVX2       : ${REAME_HAS_AVX2}")
message(STATUS "  AVX-512    : ${REAME_HAS_AVX512}")
message(STATUS "-----------------------------------------------")
