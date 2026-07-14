#!/bin/sh
set -eu
cmake -Bbuild -DBUILD_TESTS=1 -DCMAKE_CXX_FLAGS='-fprofile-arcs -ftest-coverage'
cmake --build build -j
ctest --test-dir build --output-on-failure

geninfo build/CMakeFiles/vortex_cxx_shared.dir/ \
    build/tests/CMakeFiles/vortex_cxx_test.dir/ \
    --rc geninfo_unexecuted_blocks=1 \
    --exclude /usr --exclude build/_deps --exclude tests \
    -j -b src -o coverage.info
if [ $# -gt 0 ]; then
    genhtml coverage.info -o coverage
fi
