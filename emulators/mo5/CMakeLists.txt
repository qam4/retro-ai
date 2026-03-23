# MO5 emulator core — wraps Crayon (Thomson MO5 emulator)

# Build Crayon's core library (headless, no SDL, no tests)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_LIBRETRO OFF CACHE BOOL "" FORCE)
set(BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
set(ENABLE_SDL OFF CACHE BOOL "" FORCE)

# Crayon uses include_directories(${CMAKE_SOURCE_DIR}/include) which resolves
# wrong when built as a subdirectory. Override before adding.
set(CRAYON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../crayon)
add_subdirectory(${CRAYON_DIR}
                 ${CMAKE_CURRENT_BINARY_DIR}/crayon
                 EXCLUDE_FROM_ALL)

# Fix Crayon's include directories (CMAKE_SOURCE_DIR pointed to retro-ai root)
set_target_properties(crayon_core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
target_include_directories(crayon_core PUBLIC
    $<BUILD_INTERFACE:${CRAYON_DIR}/include>
    $<BUILD_INTERFACE:${CRAYON_DIR}/third_party>
)

# Thin adapter library that maps mo5_core.h to Crayon's API
add_library(mo5_core STATIC src/mo5_crayon_adapter.cpp)
target_include_directories(mo5_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../crayon/include>
)
target_link_libraries(mo5_core PRIVATE crayon_core)
set_target_properties(mo5_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
