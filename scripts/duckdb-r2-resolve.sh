#!/bin/sh

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Resolve DuckDB version in vortex-duckdb/build.rs and check which archives
# are present in R2. Output "version", "ref_dir", "release", "matrix", and
# "any_missing".
#
# Required env vars: PUBLIC_BASE_URL, GITHUB_OUTPUT

set -eu

version=$(grep -oP 'DEFAULT_DUCKDB_VERSION:\s*&str\s*=\s*"\K[^"]+' \
    vortex-duckdb/build.rs)

# vortex-duckdb/build.rs: >=2 dot-separated numbers are a
# tagged release (ref dir "vX.Y.Z"), anything else is a commit.
ref=${version#v}
if printf '%s' "$ref" | grep -Eq '^[0-9]+(\.[0-9]+)+$'; then
    release=true
    ref_dir="v$ref"
else
    release=false
    ref_dir="$ref"
fi

echo "DuckDB $version release=$release"

entries=$(mktemp)
trap 'rm -f "$entries"' EXIT

for archive in \
    libduckdb-linux-amd64.zip \
    libduckdb-linux-arm64.zip \
    libduckdb-osx-universal.zip; do

    url="${PUBLIC_BASE_URL}/${ref_dir}/${archive}"
    code=$(curl -o /dev/null -s -w '%{http_code}' --head "$url" || echo 000)
    if [ "$code" = "200" ]; then
        echo "present in R2: $archive"
        continue
    fi

    echo "missing in R2 (HTTP $code): $archive"
    case "$archive" in
        *linux-amd64*) runner=ubuntu-latest; os=linux; arch=amd64 ;;
        *linux-arm64*) runner=ubuntu-24.04-arm; os=linux; arch=arm64 ;;
        *osx-universal*) runner=macos-14; os=osx; arch=universal ;;
    esac
    jq -nc \
        --arg archive "$archive" \
        --arg runner "$runner" \
        --arg os "$os" \
        --arg arch "$arch" \
        '{archive: $archive, runner: $runner, os: $os, arch: $arch}' >> "$entries"
done

if [ -s "$entries" ]; then
    include=$(jq -sc '.' < "$entries")
    matrix=$(jq -nc --argjson include "$include" '{include: $include}')
    any_missing=true
else
    matrix='{"include":[]}'
    any_missing=false
fi

echo "any_missing=$any_missing"

{
    echo "version=$version"
    echo "ref_dir=$ref_dir"
    echo "release=$release"
    echo "matrix=$matrix"
    echo "any_missing=$any_missing"
} >> "$GITHUB_OUTPUT"
