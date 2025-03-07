cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME vortex_duckdb)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
include_directories(gen/include)

set(EXTENSION_SOURCES src/vortex_duckdb_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link OpenSSL in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)

## Define and build rust

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(RUST_PROJECT_DIR "${CMAKE_SOURCE_DIR}/../..")
# The profile to use when building the crate
set(RUST_BUILD_MODE "debug")
# The shared object (so) we're creating
set(RUST_SO_NAME "libvortex_duckdb.dylib")
# Where is the object initially created?
set(RUST_SO "${RUST_PROJECT_DIR}/target/${RUST_BUILD_MODE}/${RUST_SO_NAME}")
# The common location it will live
set(RUST_SO_COMMON "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${RUST_SO_NAME}")


message("${RUST_SO_NAME}")
message("${RUST_SO}")
message("${RUST_SO_COMMON}")
message(${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${RUST_SO_NAME})

add_custom_command(
        OUTPUT ${RUST_SO_COMMON}
        # Build the rust code
        COMMAND ${CMAKE_COMMAND} -E env cargo build -p vortex-duckdb
        --manifest-path ${RUST_PROJECT_DIR}/Cargo.toml
        # Copy the library to a common location
        COMMAND ${CMAKE_COMMAND} -E copy ${RUST_SO} ${RUST_SO_COMMON}
        WORKING_DIRECTORY ${RUST_PROJECT_DIR}
        COMMENT "Building Rust project with cargo"
        VERBATIM
)

add_custom_target(rust_build ALL
        DEPENDS ${RUST_SO_COMMON}
)

add_library(vortex_duckdb SHARED IMPORTED)
set_target_properties(vortex_duckdb PROPERTIES
        IMPORTED_LOCATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${RUST_SO_NAME}
)

add_dependencies(vortex_duckdb rust_build)


### end rust

target_link_libraries(${EXTENSION_NAME} vortex_duckdb)
target_link_libraries(${LOADABLE_EXTENSION_NAME} vortex_duckdb)


install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")