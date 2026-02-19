#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# Merge vortex-dtype into vortex-array, then delete vortex-dtype entirely.
#
# Run from the repo root:
#   bash scripts/merge-dtype-into-array.sh

# ─── Step 1: Move source files into vortex-array/src/dtype/ ─────────────────
mkdir -p vortex-array/src/dtype
cp -r vortex-dtype/src/* vortex-array/src/dtype/
mv vortex-array/src/dtype/lib.rs vortex-array/src/dtype/mod.rs

# ─── Step 2: Handle dtype.rs module inception ───────────────────────────────
# Rename dtype.rs → dtype_impl.rs to avoid having a file named the same as its
# parent directory module, and update references in mod.rs.
mv vortex-array/src/dtype/dtype.rs vortex-array/src/dtype/dtype_impl.rs
sed -i '' 's/^mod dtype;$/mod dtype_impl;/' vortex-array/src/dtype/mod.rs
sed -i '' 's/^pub use dtype::NativeDType;$/pub use dtype_impl::NativeDType;/' \
  vortex-array/src/dtype/mod.rs

# Move the DType enum definition from dtype_impl.rs into mod.rs so that the
# canonical definition lives at the module root.

# 2a: Extract the DType enum (doc comment through closing brace) to a temp file.
sed -n '/^\/\/\/ The logical types of elements in Vortex arrays\./,/^}$/p' \
  vortex-array/src/dtype/dtype_impl.rs > /tmp/vortex_dtype_enum.rs

# 2b: Remove the enum from dtype_impl.rs.
sed -i '' '/^\/\/\/ The logical types of elements in Vortex arrays\./,/^}$/d' \
  vortex-array/src/dtype/dtype_impl.rs

# 2c: In dtype_impl.rs, DType is now in the parent module. Import it and bring
#     all variants into scope for the impl blocks.
sed -i '' 's/^use DType::\*;$/use super::DType;\
use DType::*;/' vortex-array/src/dtype/dtype_impl.rs

# 2d: Build a new mod.rs with the DType enum inserted before the `pub use`
#     re-exports, and add the `use std::sync::Arc;` import that the enum needs.
INSERT_LINE=$(rg -n '^pub use bigint::\*;' vortex-array/src/dtype/mod.rs | head -1 | cut -d: -f1)
{
  # Everything before the first `pub use` line.
  head -n $((INSERT_LINE - 1)) vortex-array/src/dtype/mod.rs
  echo "use std::sync::Arc;"
  echo ""
  cat /tmp/vortex_dtype_enum.rs
  echo ""
  # Everything from the first `pub use` line onward.
  tail -n +"$INSERT_LINE" vortex-array/src/dtype/mod.rs
} > /tmp/vortex_dtype_mod.rs
mv /tmp/vortex_dtype_mod.rs vortex-array/src/dtype/mod.rs

# 2e: Remove the old re-export of DType (it is now defined directly in mod.rs).
sed -i '' '/^pub use dtype::DType;$/d' vortex-array/src/dtype/mod.rs
sed -i '' '/^pub use dtype_impl::DType;$/d' vortex-array/src/dtype/mod.rs

rm -f /tmp/vortex_dtype_enum.rs

# ─── Step 3: Strip crate-level attributes from dtype/mod.rs ─────────────────
# Inner #![...] attributes are only valid at the crate root. Drop them.
sed -i '' '/^#!\[cfg(target_endian/d' vortex-array/src/dtype/mod.rs
sed -i '' '/^#!\[deny/d; /^#!\[warn/d' vortex-array/src/dtype/mod.rs

# ─── Step 4: Add `pub mod dtype;` to vortex-array/src/lib.rs ────────────────
# Insert alphabetically (between display and executor).
sed -i '' 's/^mod executor;$/pub mod dtype;\nmod executor;/' \
  vortex-array/src/lib.rs

# ─── Step 5: Fix imports in moved dtype files ───────────────────────────────
# 5a: crate:: → crate::dtype:: (all internal references in the moved files).
fd -e rs . vortex-array/src/dtype -x sed -i '' 's/crate::/crate::dtype::/g'

# 5b: Fix double dtype:: caused by the internal `dtype` module (now `dtype_impl`)
#     being referenced as crate::dtype:: in the original, which became crate::dtype::dtype::.
fd -e rs . vortex-array/src/dtype -x sed -i '' 's/crate::dtype::dtype::/crate::dtype::/g'

# 5c: vortex_dtype:: → vortex_array::dtype:: (doc examples in the moved files).
fd -e rs . vortex-array/src/dtype -x sed -i '' 's/vortex_dtype::/vortex_array::dtype::/g'

# 5d: Fix #[macro_export] macro bodies. These macros used literal `vortex_dtype::`
#     references which step 5c turned into `vortex_array::dtype::`. But you can't
#     use a crate's own name from within that crate — macros need `$crate::`.
#     Only apply to non-comment lines to preserve doc examples.
fd -e rs . vortex-array/src/dtype \
  -x sed -i '' '/^[[:space:]]*\/\//!s/vortex_array::/$crate::/g'

# ─── Step 6: Fix imports in existing vortex-array files ─────────────────────
# 6a: vortex_dtype:: → crate::dtype:: in vortex-array/src/ EXCEPT the dtype/ subdirectory.
fd -e rs . vortex-array/src --exclude dtype \
  -x sed -i '' 's/vortex_dtype::/crate::dtype::/g'

# 6b: vortex_dtype:: → vortex_array::dtype:: in bench/test files (these compile as
#     separate binaries and use vortex_array::, not crate::).
fd -e rs . vortex-array --exclude src \
  -x sed -i '' 's/vortex_dtype::/vortex_array::dtype::/g'

# ─── Step 7: Fix imports in all other crates ────────────────────────────────
# vortex_dtype:: → vortex_array::dtype:: across the entire workspace,
# excluding vortex-array (handled above) and vortex-dtype (being deleted).
fd -e rs . \
  --exclude vortex-array \
  --exclude vortex-dtype \
  -x sed -i '' 's/vortex_dtype::/vortex_array::dtype::/g'


# ─── Step 7b: Fix #[macro_export] macro imports ─────────────────────────────
# Macros with #[macro_export] are exported at the crate root, not at the module
# where they are defined. The complete list from vortex-dtype:
#   field_path, match_each_native_ptype, match_each_integer_ptype,
#   match_each_unsigned_integer_ptype, match_each_signed_integer_ptype,
#   match_each_float_ptype, match_each_native_simd_ptype,
#   match_smallest_offset_type, match_each_decimal_value, match_each_decimal_value_type
MACRO_SED='s/::dtype::match_each_/::match_each_/g; s/::dtype::match_smallest_/::match_smallest_/g; s/::dtype::field_path/::field_path/g'

# Within vortex-array/src (uses crate::dtype:: → crate::)
fd -e rs . vortex-array/src -x sed -i '' "$MACRO_SED"

# Within vortex-array bench/test files (uses vortex_array::dtype:: → vortex_array::)
fd -e rs . vortex-array --exclude src -x sed -i '' "$MACRO_SED"

# In all other crates (including vortex-duckdb, vortex-python, etc.)
fd -e rs . --exclude vortex-array --exclude vortex-dtype -x sed -i '' "$MACRO_SED"

# Also fix macro imports in vortex-duckdb and vortex-python specifically for any
# non-.rs files (e.g. build scripts, pyo3 bindings) that may use these macros.
fd -e rs . vortex-duckdb -x sed -i '' "$MACRO_SED"
fd -e rs . vortex-python -x sed -i '' "$MACRO_SED"

# ─── Step 8: Update vortex-array/Cargo.toml ─────────────────────────────────
# 8a: Add cudarc (optional) — after cfg-if alphabetically.
sed -i '' '/^cfg-if = { workspace = true }/a\
cudarc = { workspace = true, optional = true }
' vortex-array/Cargo.toml

# 8b: Add half with num-traits feature — after goldenfile.
sed -i '' '/^goldenfile = /a\
half = { workspace = true, features = ["num-traits"] }
' vortex-array/Cargo.toml

# 8c: Add jiff — after itertools.
sed -i '' '/^itertools = { workspace = true }/a\
jiff = { workspace = true }
' vortex-array/Cargo.toml

# 8d: Add primitive-types (optional) — after pin-project-lite.
sed -i '' '/^pin-project-lite = { workspace = true }/a\
primitive-types = { workspace = true, optional = true, features = ["arbitrary"] }
' vortex-array/Cargo.toml

# 8e: Add "dtype" feature to vortex-flatbuffers.
sed -i '' 's/vortex-flatbuffers = { workspace = true, features = \["array"\] }/vortex-flatbuffers = { workspace = true, features = ["array", "dtype"] }/' \
  vortex-array/Cargo.toml

# 8f: Add "dtype" feature to vortex-proto.
sed -i '' 's/vortex-proto = { workspace = true, features = \["expr", "scalar"\] }/vortex-proto = { workspace = true, features = ["dtype", "expr", "scalar"] }/' \
  vortex-array/Cargo.toml

# 8g: Add "flatbuffers" feature to vortex-error.
sed -i '' 's/vortex-error = { workspace = true }/vortex-error = { workspace = true, features = ["flatbuffers"] }/' \
  vortex-array/Cargo.toml

# 8h: Add "rc" feature to serde (dtype needs serde with "rc" + "derive").
sed -i '' 's/serde = { workspace = true, optional = true, features = \["derive"\] }/serde = { workspace = true, optional = true, features = ["derive", "rc"] }/' \
  vortex-array/Cargo.toml

# 8i: Remove vortex-dtype dependency.
sed -i '' '/^vortex-dtype = /d' vortex-array/Cargo.toml

# 8j: Update arbitrary feature — replace vortex-dtype/arbitrary with dep:primitive-types.
sed -i '' 's/arbitrary = \["dep:arbitrary", "vortex-dtype\/arbitrary"\]/arbitrary = ["dep:arbitrary", "dep:primitive-types"]/' \
  vortex-array/Cargo.toml

# 8k: Add cudarc feature (after canonical_counter).
sed -i '' '/^canonical_counter = \[\]/a\
cudarc = ["dep:cudarc"]
' vortex-array/Cargo.toml

# 8l: Remove "vortex-dtype/serde" from serde feature list.
sed -i '' '/"vortex-dtype\/serde",/d' vortex-array/Cargo.toml

# 8m: Add serde_json and serde_test to dev-dependencies.
sed -i '' '/^rstest = { workspace = true }$/a\
serde_json = { workspace = true }\
serde_test = { workspace = true }
' vortex-array/Cargo.toml

# ─── Step 9: Remove vortex-dtype dep from all other Cargo.toml files ────────
# First, add vortex-array to any Cargo.toml that has vortex-dtype but not
# vortex-array (e.g. vortex-cuda/cub).
for f in $(fd -g Cargo.toml . --exclude vortex-array --exclude vortex-dtype); do
  if rg -q '^vortex-dtype = ' "$f" && ! rg -q '^vortex-array = ' "$f"; then
    sed -i '' '/^vortex-dtype = /a\
vortex-array = { workspace = true }
' "$f"
  fi
done

# Propagate cudarc feature to vortex-cuda's vortex-array dependency.
sed -i '' 's/^vortex-array = { workspace = true }$/vortex-array = { workspace = true, features = ["cudarc"] }/' \
  vortex-cuda/Cargo.toml

# Remove vortex-dtype dependency from all other Cargo.toml files.
fd -g Cargo.toml . \
  --exclude vortex-array \
  --exclude vortex-dtype \
  -x sed -i '' '/^vortex-dtype = /d'

# Remove "vortex-dtype/serde" from the umbrella vortex crate's serde feature.
sed -i '' '/"vortex-dtype\/serde",/d' vortex/Cargo.toml

# ─── Step 10: Update vortex/src/lib.rs (umbrella crate) ─────────────────────
# Point the dtype re-export at vortex-array::dtype instead of vortex_dtype.
sed -i '' 's/pub use vortex_dtype::\*;/pub use vortex_array::dtype::*;/' \
  vortex/src/lib.rs

# Fix the DTypeSession import.
sed -i '' 's/use vortex_dtype::session::DTypeSession;/use vortex_array::dtype::session::DTypeSession;/' \
  vortex/src/lib.rs

# ─── Step 11: Remove vortex-dtype from workspace root Cargo.toml ────────────
# Remove from members list.
sed -i '' '/"vortex-dtype",/d' Cargo.toml

# Remove workspace dependency definition.
sed -i '' '/^vortex-dtype = .*path = /d' Cargo.toml

# ─── Step 12: Delete the vortex-dtype crate entirely ────────────────────────
rm -rf vortex-dtype

# ─── Step 13: Format files ──────────────────────────────────────────────────
cargo +nightly fmt --all
taplo fmt

# ─── Step 14: Update public API lockfile ─────────────────────────────────────
bash ./scripts/public-api.sh
cargo update --manifest-path java/testfiles/Cargo.toml

echo ""
echo "Done! Run these to verify:"
echo "  cargo clippy --all-targets --all-features"
echo "  cargo nextest run --all-features --no-fail-fast"
