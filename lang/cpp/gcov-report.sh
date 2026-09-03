#!/bin/sh

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -eu
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DVORTEX_BUILD_TESTING=ON \
    -DCMAKE_CXX_FLAGS='-fprofile-arcs -ftest-coverage'
cmake --build build --parallel
ctest --test-dir build --output-on-failure

geninfo build/CMakeFiles/vortex_cxx.dir/ \
    build/tests/CMakeFiles/vortex_cxx_test.dir/ \
    --rc geninfo_unexecuted_blocks=1 \
    --exclude /usr --exclude build/_deps --exclude tests \
    -j -b src -o coverage.info
if [ $# -gt 0 ]; then
    genhtml coverage.info -o coverage
fi
