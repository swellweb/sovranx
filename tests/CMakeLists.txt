include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(sovrano_tests
    unit/test_config.cpp
    unit/test_logger.cpp
    unit/test_model.cpp
    unit/test_sampler.cpp
    unit/test_engine.cpp
)
target_link_libraries(sovrano_tests PRIVATE
    sovrano_core
    sovrano_utils
    Catch2::Catch2WithMain
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
# Catch2 exits with 4 when a test SKIPs (e.g. integration test without a
# downloaded model); tell ctest that means "skipped", not "failed".
catch_discover_tests(sovrano_tests PROPERTIES SKIP_RETURN_CODE 4)
