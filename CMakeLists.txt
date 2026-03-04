cmake_minimum_required(VERSION 3.20)
project(retro_ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Static libs linked into the Python .so need -fPIC on Linux
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find Python and pybind11 for bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build emulator cores as libraries
# Videopac emulator
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/emulators/videopac/CMakeLists.txt)
    # Build only the core library — disable standalone, libretro, tests, SDL
    set(BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
    set(BUILD_LIBRETRO OFF CACHE BOOL "" FORCE)
    set(ENABLE_SDL OFF CACHE BOOL "" FORCE)
    set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    # Disable the emulator's own test suite (we have our own)
    set(VIDEOPAC_SAVED_BUILD_TESTS ${BUILD_TESTS})
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(emulators/videopac)
    # Restore our BUILD_TESTS setting
    set(BUILD_TESTS ${VIDEOPAC_SAVED_BUILD_TESTS} CACHE BOOL "" FORCE)

    # The submodule's CMakeLists.txt uses CMAKE_SOURCE_DIR (which resolves to
    # our top-level dir) instead of CMAKE_CURRENT_SOURCE_DIR.  Fix the include
    # directories so videopac_core points at its own headers, not ours.
    # Use generator expressions to satisfy CMake's install export validation.
    set_target_properties(videopac_core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(videopac_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/emulators/videopac/include>
        $<INSTALL_INTERFACE:include>
    )

    set(HAVE_VIDEOPAC ON)
else()
    message(WARNING "Videopac emulator not found. Run: git submodule update --init")
    set(HAVE_VIDEOPAC OFF)
endif()

# MO5 emulator
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/emulators/mo5/CMakeLists.txt)
    add_subdirectory(emulators/mo5)
    set(HAVE_MO5 ON)
else()
    message(WARNING "MO5 emulator not found. Run: git submodule update --init")
    set(HAVE_MO5 OFF)
endif()

# Retro-AI core library (header-only)
add_library(retro_ai_core INTERFACE)
target_include_directories(retro_ai_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Reward systems library (defined before adapters that depend on it)
add_library(reward_systems
    src/reward_systems.cpp
    src/reward_systems/survival.cpp
    src/reward_systems/memory.cpp
    src/reward_systems/vision.cpp
    src/reward_systems/intrinsic.cpp
    src/reward_systems/custom.cpp
)
target_link_libraries(reward_systems PUBLIC retro_ai_core)

# RL adapters for each emulator
if(HAVE_VIDEOPAC)
    add_library(videopac_rl src/videopac_rl.cpp)
    target_link_libraries(videopac_rl
        PUBLIC retro_ai_core
        PRIVATE videopac_core reward_systems
    )
    # Use generator expressions to avoid source-dir-prefixed paths in exports
    target_include_directories(videopac_rl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/emulators/videopac/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

if(HAVE_MO5)
    add_library(mo5_rl src/mo5_rl.cpp)
    target_link_libraries(mo5_rl
        PUBLIC retro_ai_core
        PRIVATE mo5_core reward_systems
    )
    target_include_directories(mo5_rl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/emulators/mo5/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(retro_ai_native python/bindings.cpp)
    target_link_libraries(retro_ai_native PRIVATE retro_ai_core reward_systems)

    if(HAVE_VIDEOPAC)
        target_link_libraries(retro_ai_native PRIVATE videopac_rl)
        target_compile_definitions(retro_ai_native PRIVATE HAVE_VIDEOPAC)
    endif()

    if(HAVE_MO5)
        target_link_libraries(retro_ai_native PRIVATE mo5_rl)
        target_compile_definitions(retro_ai_native PRIVATE HAVE_MO5)
    endif()

    install(TARGETS retro_ai_native LIBRARY DESTINATION .)
endif()

# Tests
if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()

# Install headers
install(DIRECTORY include/retro_ai DESTINATION include)

# Install libraries — include all dependent targets in the export set
install(TARGETS retro_ai_core EXPORT retro_ai_targets)
install(TARGETS reward_systems EXPORT retro_ai_targets)

if(HAVE_VIDEOPAC)
    install(TARGETS videopac_rl videopac_core EXPORT retro_ai_targets)
endif()
if(HAVE_MO5)
    install(TARGETS mo5_rl mo5_core EXPORT retro_ai_targets)
endif()

install(EXPORT retro_ai_targets
    FILE retro_ai_targets.cmake
    NAMESPACE retro_ai::
    DESTINATION lib/cmake/retro_ai
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/retro_ai_config_version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/retro_ai_config_version.cmake"
    DESTINATION lib/cmake/retro_ai
)

# ---------------------------------------------------------------------------
# Python lint / format targets
# ---------------------------------------------------------------------------
if(Python3_FOUND)
    set(_PY_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")

    add_custom_target(format-python
        COMMAND ${Python3_EXECUTABLE} -m black "${_PY_SRC_DIR}"
        COMMENT "Formatting Python code with black"
        VERBATIM
    )

    add_custom_target(lint-python
        COMMAND ${Python3_EXECUTABLE} -m black --check "${_PY_SRC_DIR}"
        COMMAND ${Python3_EXECUTABLE} -m ruff check "${_PY_SRC_DIR}"
        COMMENT "Linting Python code (black --check + ruff)"
        VERBATIM
    )
endif()
