include_directories(${CMAKE_SOURCE_DIR}/../src/include)

add_executable(vortex_tests src/vortex_expr_test.cpp)
target_include_directories(vortex_tests PRIVATE
    ${catch2_SOURCE_DIR}/single_include
)

target_link_libraries(vortex_tests PRIVATE
    ${EXTENSION_NAME}
    Catch2::Catch2WithMain)

enable_testing()
add_test(NAME AllTests COMMAND vortex_tests)
