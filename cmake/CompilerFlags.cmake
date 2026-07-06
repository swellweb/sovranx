# Compiler flags shared by all Sovrano targets.
# SIMD flags are applied only on x86_64 and only if the compiler accepts them,
# so the project also configures cleanly on arm64 (Apple Silicon dev machines).

include(CheckCXXCompilerFlag)

add_library(sovrano_flags INTERFACE)

target_compile_options(sovrano_flags INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion
)

if(SOVRANO_WERROR)
    target_compile_options(sovrano_flags INTERFACE -Werror)
endif()

set(SOVRANO_HAS_AVX2 OFF)
set(SOVRANO_HAS_AVX512 OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    if(SOVRANO_ENABLE_AVX2)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(sovrano_flags INTERFACE -mavx2 -mfma -mf16c)
            set(SOVRANO_HAS_AVX2 ON)
        endif()
    endif()
    if(SOVRANO_ENABLE_AVX512)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        if(COMPILER_SUPPORTS_AVX512)
            target_compile_options(sovrano_flags INTERFACE
                -mavx512f -mavx512bw -mavx512vl -mavx512dq)
            set(SOVRANO_HAS_AVX512 ON)
        endif()
    endif()
endif()
