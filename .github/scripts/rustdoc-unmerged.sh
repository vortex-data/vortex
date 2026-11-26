#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# A RUSTDOC wrapper that writes each crate's documentation to its own directory.
#
# Every rustdoc invocation rereads and rewrites the shared files of its output directory (the
# search index, source index, and crate list) to merge in its crate, under a lock. With dozens of
# workspace crates documented into one `target/doc`, that merge dominates `cargo doc`: it grows with
# every crate already documented, so it scales quadratically, and it serializes rustdoc. CI only
# checks that documentation builds, so nothing needs the merged output. Stable rustdoc has no way to
# defer merging (`--merge` is unstable), so this gives each crate a separate output directory.
#
# Use as `RUSTDOC=.github/scripts/rustdoc-unmerged.sh cargo doc ...`. The real rustdoc is
# `$REAL_RUSTDOC` if set, and otherwise `rustdoc` from PATH.
set -euo pipefail

args=("$@")
crate_name=""
out_index=""
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --crate-name) crate_name="${args[i + 1]:-}" ;;
    -o | --out-dir) out_index=$((i + 1)) ;;
  esac
done

# Only rewrite documentation builds; pass anything else (such as doctests or version probes)
# through unchanged.
if [[ -n "$crate_name" && -n "$out_index" && " ${args[*]} " != *" --test "* ]]; then
  args[out_index]="${args[out_index]}/../doc-unmerged/${crate_name}"
fi

exec "${REAL_RUSTDOC:-rustdoc}" "${args[@]}"
