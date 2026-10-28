#!/bin/sh

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Produce $ARCHIVE for upload to R2: either download DuckDB release for
# $REF_DIR or build DuckDB from source at commit $REF_DIR and pack
# libraries and headers.
#
# Required env vars: ARCHIVE, REF_DIR, RELEASE, PLATFORM_OS

set -eu

if [ "$RELEASE" = "true" ]; then
    suffix="${ARCHIVE#libduckdb-}"
    suffix="${suffix%.zip}"
    upstream="duckdb-shared-libs-${suffix}.tar.gz"

    echo "Mirroring DuckDB release ${REF_DIR}/${upstream} as ${ARCHIVE}"
    curl -fSL --retry 3 -o "$upstream" \
        "https://github.com/duckdb/duckdb/releases/download/${REF_DIR}/${upstream}"

    stage="stage"
    mkdir -p "$stage"
    tar -xzf "$upstream" -C "$stage"
    ( cd "$stage" && zip -r "../${ARCHIVE}" . )
else
    echo "Building DuckDB commit ${REF_DIR} from source"

    curl -fSL --retry 3 -o duckdb-src.zip \
        "https://github.com/duckdb/duckdb/archive/${REF_DIR}.zip"

    # macos zip extract error: cannot create
    # <...>/issue2628_������.csv Illegal byte sequence
    if [ "$PLATFORM_OS" = "osx" ]; then
        7z x duckdb-src.zip
    else
        unzip -q duckdb-src.zip
    fi

    src_dir="duckdb-${REF_DIR}"
    extra=""
    if [ "$PLATFORM_OS" = "osx" ]; then
        extra="OSX_BUILD_UNIVERSAL=1"
        extensions="parquet;tpch;tpcds;icu"
    else
        make -C "$src_dir" setup-vcpkg
        VCPKG_TOOLCHAIN_PATH="$PWD/${src_dir}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        export VCPKG_TOOLCHAIN_PATH
        extensions="parquet;tpch;tpcds;icu;httpfs"
    fi

    make -C "$src_dir" \
        GEN=ninja \
        DISABLE_SANITIZER=1 \
        THREADSAN=0 \
        BUILD_SHELL=false \
        BUILD_UNITTESTS=false \
        ENABLE_UNITTEST_CPP_TESTS=false \
        BUILD_EXTENSIONS="$extensions" \
        LINK_CORE_EXTENSIONS=1 \
        $extra

    lib_dir="${src_dir}/build/release/src"
    stage="stage"
    mkdir -p "$stage"

    cp -a "${lib_dir}/libduckdb.so" "$stage/" 2>/dev/null || true
    cp -a "${lib_dir}/libduckdb.dylib" "$stage/" 2>/dev/null || true
    cp -a "${lib_dir}/libduckdb_static.a" "$stage/"
    cp -a "${src_dir}/src/include/duckdb.h" "$stage/" 2>/dev/null || true
    cp -a "${src_dir}/src/include/duckdb.hpp" "$stage/" 2>/dev/null || true

    ( cd "$stage" && zip -r "../${ARCHIVE}" . )
fi

ls -la "$ARCHIVE"
