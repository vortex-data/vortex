cmake_minimum_required(VERSION 3.22)

set(TARGET_NAME vortex)
project(${TARGET_NAME}_project)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)

FetchContent_MakeAvailable(Corrosion)

find_package(Catch2 CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

if (APPLE)
    find_library(SECURITY_FRAMEWORK Security)
endif ()

corrosion_import_crate(MANIFEST_PATH ../Cargo.toml
    CRATES vortex-ffi
    FEATURES duckdb mimalloc
    CRATE_TYPES staticlib
    FLAGS --crate-type=staticlib
)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(EXTENSION_SOURCES src/vortex_extension.cpp src/expr/expr.cpp src/vortex_write.cpp src/vortex_scan.cpp)
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
    ${SECURITY_FRAMEWORK}
)

add_subdirectory(test)

install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
