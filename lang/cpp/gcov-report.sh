#!/bin/sh

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Builds the C++ tests with gcov instrumentation, runs them, and writes
# coverage.info next to this script; pass `html` to also render coverage/.
# CMAKE_BUILD_PARALLEL_LEVEL overrides the detected CPU count.
set -eu
cd "$(dirname "$0")"

coverage_key='$<TARGET_PROPERTY:BINARY_DIR>/$<TARGET_PROPERTY:NAME>'
coverage_launcher="cmake;-E;env;SCCACHE_C_CUSTOM_CACHE_BUSTER=${SCCACHE_C_CUSTOM_CACHE_BUSTER:-}:$coverage_key"
coverage_launcher="$coverage_launcher${CMAKE_CXX_COMPILER_LAUNCHER:+;$CMAKE_CXX_COMPILER_LAUNCHER}"

# CMAKE_SHARED_LINKER_FLAGS links gcov into C-linked shared libraries.
# CMAKE_CXX_COMPILER_LAUNCHER keeps cached .gcda paths target-specific.
cmake -S . -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DVORTEX_BUILD_TESTS=ON \
    -DCMAKE_CXX_FLAGS=--coverage \
    -DCMAKE_SHARED_LINKER_FLAGS=--coverage \
    -DCMAKE_CXX_COMPILER_LAUNCHER="$coverage_launcher"

# getconf works on Linux and macOS; nproc is not installed on stock macOS.
cmake --build build \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
ctest --test-dir build --output-on-failure

# lcov matches exclude globs against full source paths.
geninfo build/CMakeFiles/vortex_cxx_shared.dir/ \
    build/tests/CMakeFiles/vortex_cxx_test.dir/ \
    --rc geninfo_unexecuted_blocks=1 \
    --exclude '/usr/*' --exclude '*/_deps/*' --exclude '*/tests/*' \
    -j -b src -o coverage.info
if [ "${1:-}" = html ]; then
    genhtml coverage.info -o coverage
fi
