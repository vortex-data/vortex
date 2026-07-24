# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

cmake_minimum_required(VERSION 3.10)
project(VortexCXX
    VERSION 0.0.1
    LANGUAGES CXX)

include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(SCCACHE_PROGRAM sccache)
if (SCCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    message(STATUS "Sccache found: ${SCCACHE_PROGRAM}")
else ()
    message(STATUS "Sccache not found")
endif ()

option(BUILD_TESTS "Build tests" OFF)

set(SANITIZER "" CACHE STRING "Build with sanitizers")
set(TARGET_TRIPLE "" CACHE STRING "Rust target triple for FFI library")
set(RUST_BUILD_PROFILE "" CACHE STRING "Cargo profile name for Rust FFI library")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if (NOT SANITIZER STREQUAL "")
    message(NOTICE "Sanitizer: ${SANITIZER}")
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "Only debug build is supported for sanitizer builds")
    endif()

    if (SANITIZER STREQUAL "asan")
        set(SANITIZER_FLAGS "-fsanitize=address,undefined,leak")
    elseif (SANITIZER STREQUAL "tsan")
        set(SANITIZER_FLAGS "-fsanitize=thread")
    else()
        message(FATAL_ERROR "Unknown sanitizer ${SANITIZER}")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
endif()

if (RUST_BUILD_PROFILE STREQUAL "")
    if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(RUST_BUILD_PROFILE "release_debug")
    else()
        string(TOLOWER "${CMAKE_BUILD_TYPE}" RUST_BUILD_PROFILE)
    endif()
endif()

set(VORTEX_FFI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vortex-ffi")
set(VORTEX_FFI_LIB_DIR "${VORTEX_FFI_DIR}/../target/${TARGET_TRIPLE}/${RUST_BUILD_PROFILE}")
set(VORTEX_FFI_HEADERS "${VORTEX_FFI_DIR}/cinclude")

if(APPLE)
    set(VORTEX_FFI_STATIC "${VORTEX_FFI_LIB_DIR}/libvortex_ffi.a")
    set(VORTEX_FFI_SHARED "${VORTEX_FFI_LIB_DIR}/libvortex_ffi.dylib")
elseif(WIN32)
    set(VORTEX_FFI_STATIC "${VORTEX_FFI_LIB_DIR}/libvortex_ffi.lib")
    set(VORTEX_FFI_SHARED "${VORTEX_FFI_LIB_DIR}/libvortex_ffi.dll")
else()
    set(VORTEX_FFI_STATIC "${VORTEX_FFI_LIB_DIR}/libvortex_ffi.a")
    set(VORTEX_FFI_SHARED "${VORTEX_FFI_LIB_DIR}/libvortex_ffi.so")
endif()

if(NOT EXISTS "${VORTEX_FFI_STATIC}")
    message(FATAL_ERROR
        "vortex-ffi static library not found at ${VORTEX_FFI_STATIC}. "
        "Run: cargo build --profile <profile> -p vortex-ffi")
endif()

add_library(vortex_ffi STATIC IMPORTED)
set_target_properties(vortex_ffi PROPERTIES
    IMPORTED_LOCATION "${VORTEX_FFI_STATIC}"
    INTERFACE_INCLUDE_DIRECTORIES "${VORTEX_FFI_HEADERS}")

if(EXISTS "${VORTEX_FFI_SHARED}")
    add_library(vortex_ffi_shared SHARED IMPORTED)
    set_target_properties(vortex_ffi_shared PROPERTIES
        IMPORTED_LOCATION "${VORTEX_FFI_SHARED}"
        INTERFACE_INCLUDE_DIRECTORIES "${VORTEX_FFI_HEADERS}"
        INTERFACE_LINK_OPTIONS "LINKER:-rpath,${VORTEX_FFI_LIB_DIR}")
endif()

file(GLOB VORTEX_CXX_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(vortex_cxx STATIC ${VORTEX_CXX_SOURCES})
target_include_directories(vortex_cxx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../vortex-ffi/cinclude
)
target_link_libraries(vortex_cxx PUBLIC vortex_ffi)
add_library(vortex::cxx ALIAS vortex_cxx)
target_compile_features(vortex_cxx PUBLIC cxx_std_20)

set(APPLE_LINK_FLAGS "-framework CoreFoundation -framework Security")

if(APPLE)
    target_link_libraries(vortex_cxx PRIVATE ${APPLE_LINK_FLAGS})
endif()

if(TARGET vortex_ffi_shared)
    add_library(vortex_cxx_shared SHARED ${VORTEX_CXX_SOURCES})
    set_target_properties(vortex_cxx_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(vortex_cxx_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../vortex-ffi/cinclude
    )
    target_link_libraries(vortex_cxx_shared PUBLIC vortex_ffi_shared)
    add_library(vortex::cxx_shared ALIAS vortex_cxx_shared)
    target_compile_features(vortex_cxx_shared PUBLIC cxx_std_20)

    if(APPLE)
        target_link_libraries(vortex_cxx_shared PRIVATE ${APPLE_LINK_FLAGS})
    endif()
endif()

if (BUILD_TESTS OR BUILD_EXAMPLES)
    FetchContent_Declare(
      Nanoarrow
      GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow
      GIT_TAG        apache-arrow-nanoarrow-0.8.0
    )
    FetchContent_MakeAvailable(Nanoarrow)

    # arrow-nanoarrow creates a self-referential symlink
    # python/subprojects/arrow-nanoarrow -> ../.. that makes any recursive
    if (IS_SYMLINK "${nanoarrow_SOURCE_DIR}/python/subprojects/arrow-nanoarrow")
        file(REMOVE "${nanoarrow_SOURCE_DIR}/python/subprojects/arrow-nanoarrow")
    endif()
endif()

if (BUILD_TESTS)
    FetchContent_Declare(
        Catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1
    )
    FetchContent_MakeAvailable(Catch)
    include(Catch)
    target_compile_definitions(Catch2 PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)

    include(CTest)
    add_subdirectory(tests)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
