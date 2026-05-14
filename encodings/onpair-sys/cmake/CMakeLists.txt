cmake_minimum_required(VERSION 3.21)
project(onpair_shim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/onpair_pin.cmake")

# Skip onpair_cpp's own tests/examples and tell it not to fetch Boost.
set(ONPAIR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONPAIR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ONPAIR_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(ONPAIR_NATIVE_ARCH OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    onpair_cpp
    GIT_REPOSITORY ${ONPAIR_CPP_REPO}
    GIT_TAG        ${ONPAIR_CPP_TAG}
    PATCH_COMMAND  ${CMAKE_COMMAND}
                   -DSRC_DIR=<SOURCE_DIR>
                   -P "${CMAKE_CURRENT_LIST_DIR}/strip_boost.cmake"
)
FetchContent_MakeAvailable(onpair_cpp)

add_library(onpair_shim STATIC
    "${CMAKE_CURRENT_LIST_DIR}/../cxx/onpair_shim.cpp"
)
target_include_directories(onpair_shim
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../cxx"
)
target_link_libraries(onpair_shim PUBLIC OnPair::onpair)
set_target_properties(onpair_shim PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS onpair_shim onpair
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
