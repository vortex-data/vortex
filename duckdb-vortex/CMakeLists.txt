cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TARGET_NAME vortex)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(EXTENSION_SOURCES src/vortex_extension.cpp src/expr/expr.cpp gen/dtype.pb.cc gen/expr.pb.cc gen/scalar.pb.cc)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

set(VORTEX_FFI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../vortex-ffi/cinclude)
include_directories(${VORTEX_FFI_INCLUDE_DIR})

include(FetchContent)
FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5.1
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml
        CRATES vortex-ffi
        FEATURES duckdb
        IMPORTED_CRATES vortex_ffi)

project(${TARGET_NAME}_project)
include_directories(src/include)
include_directories(gen/include)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

if (APPLE)
    find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
    find_library(SECURITY_FRAMEWORK Security)
endif ()

target_link_libraries(${EXTENSION_NAME}
        vortex_ffi-static
        ${CORE_FOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
)

target_link_libraries(${LOADABLE_EXTENSION_NAME}
        vortex_ffi-shared
        ${CORE_FOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
)

find_package(protobuf CONFIG REQUIRED)
target_link_libraries(${EXTENSION_NAME} protobuf::libprotobuf)
target_link_libraries(${LOADABLE_EXTENSION_NAME} protobuf::libprotobuf)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.0
)
FetchContent_MakeAvailable(Catch2)
add_subdirectory(test)

install(
        TARGETS ${EXTENSION_NAME}
        EXPORT "${DUCKDB_EXPORT_SET}"
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
