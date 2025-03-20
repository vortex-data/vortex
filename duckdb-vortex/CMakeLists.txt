cmake_minimum_required(VERSION 3.22)

set(TARGET_NAME vortex)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(EXTENSION_SOURCES src/vortex_extension.cpp)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

set(VORTEX_FFI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../vortex-ffi/cinclude)
include_directories(${VORTEX_FFI_INCLUDE_DIR})

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be
# added in ./vcpkg.json and then used in cmake with find_package. Feel free to
# remove or replace with other dependencies. Note that it should also be removed
# from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.1
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml
                       CRATES vortex-duckdb vortex-ffi
                       IMPORTED_CRATES vortex_ffi vortex_duckdb)

project(${TARGET_NAME}_project)
include_directories(src/include)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# if(APPLE)
#     find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
#     find_library(SECURITY_FRAMEWORK Security)
# endif()

target_link_libraries(${EXTENSION_NAME}
    OpenSSL::SSL
    OpenSSL::Crypto
    vortex_duckdb-static
    vortex_ffi-static
    ${CORE_FOUNDATION_FRAMEWORK}
    ${SECURITY_FRAMEWORK}
)

target_link_libraries(${LOADABLE_EXTENSION_NAME}
    OpenSSL::SSL
    OpenSSL::Crypto
    vortex_duckdb-shared
    vortex_ffi-static
    ${CORE_FOUNDATION_FRAMEWORK}
    ${SECURITY_FRAMEWORK}
)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
