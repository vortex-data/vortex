// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registry for per-encoding row-encode fast paths from downstream crates.
//!
//! Encodings that live outside `vortex-array` (such as `RunEnd` in `encodings/runend` or the
//! FastLanes encodings in `encodings/fastlanes`) cannot be directly downcast from inside the
//! variadic [`RowSize`] / [`RowEncode`] dispatch loops. Instead, they submit a
//! [`RowEncodeRegistration`] via the `inventory` crate, and the dispatch loop looks them up
//! by [`ArrayId`].
//!
//! [`RowSize`]: crate::RowSize
//! [`RowEncode`]: crate::RowEncode

use std::sync::OnceLock;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::options::RowSortField;

/// Function pointer signature for an encoding's per-row size contribution.
///
/// Returns `Ok(Some(()))` when the kernel handled the column, or `Ok(None)` to decline and
/// fall back to the canonical path.
pub type DynSizeFn =
    fn(&ArrayRef, RowSortField, &mut [u32], &mut ExecutionCtx) -> VortexResult<Option<()>>;

/// Function pointer signature for an encoding's per-row byte encoding (cursor path).
///
/// Returns `Ok(Some(()))` when the kernel handled the column, or `Ok(None)` to decline and
/// fall back to the canonical path.
pub type DynEncodeFn = fn(
    &ArrayRef,
    RowSortField,
    &[u32],
    &mut [u32],
    &mut [u8],
    &mut ExecutionCtx,
) -> VortexResult<Option<()>>;

/// Function pointer signature for an encoding's fixed-width arithmetic encode path.
///
/// Used for fixed-width columns that appear before any variable-length column, where the
/// within-row write position of row `i` is `i * row_stride + col_prefix (+ var_prefix[i])`.
/// This lets a fixed-width encoding (e.g. FastLanes BitPacked) fuse decompression with the
/// row write and skip both the intermediate canonical array and the per-row cursor traffic.
///
/// Returns `Ok(Some(()))` when the kernel handled the column, or `Ok(None)` to decline.
pub type DynEncodeFixedArithFn = fn(
    &ArrayRef,
    RowSortField,
    u32,            // col_prefix
    u32,            // row_stride
    Option<&[u32]>, // var_prefix (exclusive cumsum of varlen lengths), None for pure-fixed
    usize,          // nrows
    &mut [u8],
    &mut ExecutionCtx,
) -> VortexResult<Option<()>>;

/// A registration submitted by an encoding crate to plug into the row encoder.
///
/// Because [`ArrayId`] requires runtime string interning, the encoding id is passed as a
/// function pointer that is called once at registry initialization time.
pub struct RowEncodeRegistration {
    /// Returns the [`ArrayId`] of the encoding this registration applies to.
    pub id: fn() -> ArrayId,
    /// Per-row size contribution function.
    pub size: DynSizeFn,
    /// Per-row encoding function (cursor path).
    pub encode: DynEncodeFn,
    /// Optional fixed-width arithmetic encode path. Set to `None` to always take the cursor
    /// path (or the canonical fallback) for fixed-before-varlen columns.
    pub encode_fixed_arith: Option<DynEncodeFixedArithFn>,
}

inventory::collect!(RowEncodeRegistration);

type Entry = (DynSizeFn, DynEncodeFn, Option<DynEncodeFixedArithFn>);

/// Look up the registered (size, encode, encode_fixed_arith) functions for an encoding id.
pub(crate) fn lookup(id: &ArrayId) -> Option<Entry> {
    static MAP: OnceLock<HashMap<ArrayId, Entry>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        inventory::iter::<RowEncodeRegistration>
            .into_iter()
            .map(|r| ((r.id)(), (r.size, r.encode, r.encode_fixed_arith)))
            .collect()
    });
    map.get(id).copied()
}
