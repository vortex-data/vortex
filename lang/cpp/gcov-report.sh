#!/bin/sh

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Builds the C++ tests with gcov instrumentation, runs them, and writes
# coverage.info next to this script; pass `html` to also render coverage/.
set -eu
cd "$(dirname "$0")"

cmake -S . -B build \
    -DVORTEX_BUILD_TESTING=ON \
    -DCMAKE_CXX_FLAGS=--coverage
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# lcov matches exclude globs against full source paths.
geninfo build/CMakeFiles/vortex_cxx.dir/ \
    build/tests/CMakeFiles/vortex_cxx_test.dir/ \
    --rc geninfo_unexecuted_blocks=1 \
    --exclude '/usr/*' --exclude '*/_deps/*' --exclude '*/tests/*' \
    -j -b src -o coverage.info
if [ "${1:-}" = html ]; then
    genhtml coverage.info -o coverage
fi
