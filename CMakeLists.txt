cmake_minimum_required(VERSION 3.20)
project(retro_ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# Note: These assume the emulators expose their core as a library
# You may need to adjust based on your emulator's CMakeLists.txt

# Videopac emulator
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/emulators/videopac/CMakeLists.txt)
    add_subdirectory(emulators/videopac)
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

# Retro-AI core library
add_library(retro_ai_core INTERFACE)
target_include_directories(retro_ai_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# RL adapters for each emulator
if(HAVE_VIDEOPAC)
    add_library(videopac_rl src/videopac_rl.cpp)
    target_link_libraries(videopac_rl 
        PUBLIC retro_ai_core
        PRIVATE videopac_core  # Adjust based on your emulator's library name
    )
    target_include_directories(videopac_rl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/emulators/videopac/include
    )
endif()

if(HAVE_MO5)
    add_library(mo5_rl src/mo5_rl.cpp)
    target_link_libraries(mo5_rl 
        PUBLIC retro_ai_core
        PRIVATE mo5_core  # Adjust based on your emulator's library name
    )
    target_include_directories(mo5_rl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/emulators/mo5/include
    )
endif()

# Reward systems library
add_library(reward_systems src/reward_systems.cpp)
target_link_libraries(reward_systems PUBLIC retro_ai_core)

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
    
    # Install Python module
    install(TARGETS retro_ai_native LIBRARY DESTINATION .)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install headers
install(DIRECTORY include/retro_ai DESTINATION include)

# Install libraries
if(HAVE_VIDEOPAC)
    install(TARGETS videopac_rl EXPORT retro_ai_targets)
endif()
if(HAVE_MO5)
    install(TARGETS mo5_rl EXPORT retro_ai_targets)
endif()
install(TARGETS reward_systems EXPORT retro_ai_targets)

# Export targets
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
