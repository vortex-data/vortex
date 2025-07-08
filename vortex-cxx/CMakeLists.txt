cmake_minimum_required(VERSION 3.16)

if(DEFINED ENV{VCPKG_TOOLCHAIN_PATH})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_TOOLCHAIN_PATH}")
endif()

project(vortex-cxx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(VORTEX_ENABLE_TESTING "Enable building test binary for vortex-cxx" OFF)
option(VORTEX_DEV "Enable dev mode" OFF)

if (VORTEX_DEV)
    set(VORTEX_ENABLE_TESTING ON)
endif()

# Get cargo target dir
execute_process(COMMAND cargo locate-project --workspace --message-format plain
    OUTPUT_VARIABLE CARGO_TARGET_DIR
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
string(REGEX REPLACE "/Cargo.toml\n$" "/target" CARGO_TARGET_DIR "${CARGO_TARGET_DIR}")
set(CARGO_MANIFEST ${PROJECT_SOURCE_DIR}/Cargo.toml)
set(RUST_SOURCE_FILE ${PROJECT_SOURCE_DIR}/src/lib.rs)
set(RUST_BRIDGE_CPP ${CARGO_TARGET_DIR}/cxxbridge/vortex-cxx/src/lib.rs.cc)
set(RUST_HEADER_FILE ${CARGO_TARGET_DIR}/cxxbridge/vortex-cxx/src/lib.rs.h)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUST_LIB ${CARGO_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}vortex_cxx${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(RUST_LIB ${CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}vortex_cxx${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

find_package(Arrow CONFIG REQUIRED)

set(CPP_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/src
                    ${CARGO_TARGET_DIR}/cxxbridge
                    ${CARGO_TARGET_DIR}/cxxbridge/vortex-cxx/src
                    ${Arrow_INCLUDE_DIRS})

file(GLOB CPP_SOURCE_FILE "src/*.cpp")
file(GLOB CPP_HEADER_FILE "include/*.hpp" "src/*.hpp")

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CARGO_BUILD_FLAGS "--release")
endif()

add_custom_target(cargo_build
    COMMAND cargo build --manifest-path ${CARGO_MANIFEST} ${CARGO_BUILD_FLAGS}
    BYPRODUCTS ${RUST_BRIDGE_CPP} ${RUST_LIB} ${RUST_HEADER_FILE}
    DEPENDS ${RUST_SOURCE_FILE}
    USES_TERMINAL
    COMMENT "Running cargo build for vortex-cxx..."
)

add_library(vortex_cxx STATIC ${CPP_SOURCE_FILE} ${RUST_BRIDGE_CPP})
target_sources(vortex_cxx PUBLIC ${CPP_HEADER_FILE})
target_sources(vortex_cxx PRIVATE ${RUST_HEADER_FILE})
target_include_directories(vortex_cxx PUBLIC ${CPP_INCLUDE_DIR})
target_link_libraries(vortex_cxx PUBLIC ${RUST_LIB})
target_link_libraries(vortex_cxx PRIVATE ${CMAKE_DL_LIBS} Arrow::arrow_static)
set_target_properties(vortex_cxx
    PROPERTIES ADDITIONAL_CLEAN_FILES ${CARGO_TARGET_DIR}
)
add_dependencies(vortex_cxx cargo_build)

# Platform-specific configuration
if(WIN32)
    target_link_libraries(vortex_cxx userenv ws2_32 bcrypt)
endif()

if(APPLE)
    target_link_libraries(vortex_cxx "-framework CoreFoundation -framework Security")
endif()

# Tests
if (VORTEX_ENABLE_TESTING)
    find_package(GTest CONFIG REQUIRED)

    enable_testing()
    add_executable(vortex_cxx_test tests/basic_test.cpp)
    target_include_directories(vortex_cxx_test PUBLIC ${CPP_INCLUDE_DIR})
    target_link_libraries(vortex_cxx_test GTest::gtest_main vortex_cxx Arrow::arrow_static)

    include(GoogleTest)
    gtest_discover_tests(vortex_cxx_test)
endif()