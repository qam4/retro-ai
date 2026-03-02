cmake_minimum_required(VERSION 3.20)
project(videopac_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect source files
# When the real emulator submodule is checked out, replace the stub with actual sources:
#   file(GLOB_RECURSE VIDEOPAC_SOURCES "src/*.cpp" "src/*.c")
set(VIDEOPAC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/videopac_stub.cpp
)

# Build as static library
add_library(videopac_core STATIC ${VIDEOPAC_SOURCES})

# Headless mode: disable SDL and graphics output
target_compile_definitions(videopac_core
    PUBLIC
        HEADLESS_MODE=1
        NO_SDL=1
)

# Expose include directories for consumers
target_include_directories(videopac_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# C++17 standard
target_compile_features(videopac_core PUBLIC cxx_std_17)

# Platform-specific settings
if(MSVC)
    target_compile_options(videopac_core PRIVATE /W3)
else()
    target_compile_options(videopac_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
