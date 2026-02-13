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

# Extract published crates from Cargo.toml between BEGIN/END markers.
# Each line looks like:
#   vortex-alp = { version = "0.1.0", path = "./encodings/alp", ... }
while IFS= read -r line; do
    # Extract crate name (everything before the first ' =')
    crate_name=$(echo "$line" | sed 's/ *=.*//')

    # Extract path value from the line
    crate_path=$(echo "$line" | sed 's/.*path *= *"\([^"]*\)".*/\1/')
    # Strip leading ./
    crate_path="${crate_path#./}"

    echo "Generating public API for $crate_name -> $crate_path/public-api.lock"
    cargo +nightly public-api -p "$crate_name" -ss > "$REPO_ROOT/$crate_path/public-api.lock"
done < <(sed -n '/^# BEGIN crates published/,/^# END crates published/{ /^#/d; /^$/d; p; }' "$REPO_ROOT/Cargo.toml")

echo "Done. All public-api.lock files regenerated."
