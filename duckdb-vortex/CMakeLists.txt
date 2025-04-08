cmake_minimum_required(VERSION 3.22)

set(TARGET_NAME vortex)
project(${TARGET_NAME}_project)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.1
)
FetchContent_Declare(
    vcpkg
    GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
    GIT_TAG 2025.03.19
)
FetchContent_MakeAvailable(Corrosion)
FetchContent_MakeAvailable(vcpkg)

# Install vcpkg dependencies.
execute_process(
    COMMAND bash ./bootstrap-vcpkg.sh -disableMetrics
    WORKING_DIRECTORY "${vcpkg_SOURCE_DIR}"
    RESULT_VARIABLE VCPKG_BOOTSTRAP_RESULT
)
set(VCPKG_INSTALL_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
file(MAKE_DIRECTORY "${VCPKG_INSTALL_DIR}")
execute_process(
    COMMAND "${vcpkg_SOURCE_DIR}/vcpkg" install --no-print-usage --x-manifest-root=${CMAKE_CURRENT_SOURCE_DIR}/vcpkg --x-install-root=${VCPKG_INSTALL_DIR}
    WORKING_DIRECTORY "${vcpkg_SOURCE_DIR}"
)

if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_PREFIX_PATH ${VCPKG_INSTALL_DIR}/arm64-osx)
elseif (APPLE)
    set(CMAKE_PREFIX_PATH ${VCPKG_INSTALL_DIR}/x64-osx)
elseif (UNIX AND NOT APPLE)
    set(CMAKE_PREFIX_PATH ${VCPKG_INSTALL_DIR}/x64-linux)
endif ()

find_package(Catch2 CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
if (APPLE)
    find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
    find_library(SECURITY_FRAMEWORK Security)
endif ()

corrosion_import_crate(MANIFEST_PATH ../Cargo.toml
    CRATES vortex-ffi
    FEATURES duckdb
    IMPORTED_CRATES vortex_ffi
)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(EXTENSION_SOURCES src/vortex_extension.cpp src/expr/expr.cpp src/vortex_write.cpp)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

# Generate C++ code from .proto files.
file(GLOB PROTO_FILES "../vortex-proto/proto/*.proto")
set(PROTO_GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gen)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES} PROTOC_OUT_DIR ${PROTO_GEN_DIR})

include_directories(src/include ${PROTO_GEN_DIR} ../vortex-ffi/cinclude)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES} ${PROTO_SRCS})
build_loadable_extension(${TARGET_NAME} ${EXTENSION_SOURCES} ${PROTO_SRCS})

target_link_libraries(${EXTENSION_NAME}
    vortex_ffi-static
    protobuf::libprotobuf
    ${CORE_FOUNDATION_FRAMEWORK}
    ${SECURITY_FRAMEWORK}
)
target_link_libraries(${LOADABLE_EXTENSION_NAME}
    vortex_ffi-shared
    protobuf::libprotobuf
    ${CORE_FOUNDATION_FRAMEWORK}
    ${SECURITY_FRAMEWORK}
)

add_subdirectory(test)

install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
