# ---------------------------------------------------------------------------
# retro-ai C++ tests  (GoogleTest)
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Vision OCR bug condition exploration test + preservation tests
add_executable(retro_ai_tests
    test_vision_ocr_bug.cpp
    test_vision_ocr_preservation.cpp
)

target_link_libraries(retro_ai_tests
    reward_systems
    retro_ai_core
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(retro_ai_tests)
