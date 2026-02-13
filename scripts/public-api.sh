#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail

# Regenerate public-api.lock files for all published crates.
# Uses cargo-public-api to dump the public API surface of each crate.
#
# Usage:
#   bash scripts/public-api.sh           # regenerate all lock files
#   cargo +nightly public-api -p <crate> -ss > <path>/public-api.lock  # single crate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract published crate lines from Cargo.toml between BEGIN/END markers.
# Each line looks like:
#   vortex-alp = { version = "0.1.0", path = "./encodings/alp", ... }
# Build parallel arrays of crate names and paths.
crate_names=()
crate_paths=()
pkg_flags=()
while IFS= read -r line; do
    name=$(echo "$line" | sed 's/ *=.*//')
    path=$(echo "$line" | sed 's/.*path *= *"\([^"]*\)".*/\1/')
    path="${path#./}"
    crate_names+=("$name")
    crate_paths+=("$path")
    pkg_flags+=("-p" "$name")
done < <(sed -n '/^# BEGIN crates published/,/^# END crates published/{ /^#/d; /^$/d; p; }' "$REPO_ROOT/Cargo.toml")

echo "Found ${#crate_names[@]} published crates."

# Step 1: Pre-build all crates in one cargo invocation so dependencies are compiled
# in parallel using cargo's built-in parallelism.
echo "Pre-building all crates..."
cargo +nightly check "${pkg_flags[@]}"

# Insert blank lines between every item to reduce git conflicts
format_api() {
    awk 'NR > 1 { print "" } { print }'
}
export -f format_api

# Step 2: Generate public-api.lock files in parallel.
# Each invocation only needs to run rustdoc on a single (already-compiled) crate.
if command -v parallel &>/dev/null; then
    echo "Generating public API lock files in parallel..."
    export REPO_ROOT
    parallel --bar \
        'cargo +nightly public-api -p {1} -ss | format_api > "$REPO_ROOT/{2}/public-api.lock"' \
        ::: "${crate_names[@]}" :::+ "${crate_paths[@]}"
else
    echo "GNU parallel not found, falling back to sequential generation..."
    echo "  hint: brew install parallel (macOS) or apt install parallel (Linux)"
    for i in "${!crate_names[@]}"; do
        echo "  ${crate_names[$i]} -> ${crate_paths[$i]}/public-api.lock"
        cargo +nightly public-api -p "${crate_names[$i]}" -ss | format_api > "$REPO_ROOT/${crate_paths[$i]}/public-api.lock"
    done
fi

echo "Done. All public-api.lock files regenerated."
