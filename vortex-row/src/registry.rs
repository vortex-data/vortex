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

/// Function pointer signature for an encoding's per-row byte encoding.
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

/// A registration submitted by an encoding crate to plug into the row encoder.
///
/// Because [`ArrayId`] requires runtime string interning, the encoding id is passed as a
/// function pointer that is called once at registry initialization time.
pub struct RowEncodeRegistration {
    /// Returns the [`ArrayId`] of the encoding this registration applies to.
    pub id: fn() -> ArrayId,
    /// Per-row size contribution function.
    pub size: DynSizeFn,
    /// Per-row encoding function.
    pub encode: DynEncodeFn,
}

inventory::collect!(RowEncodeRegistration);

/// Look up a (size, encode) pair for the given encoding id.
pub(crate) fn lookup(id: &ArrayId) -> Option<(DynSizeFn, DynEncodeFn)> {
    static MAP: OnceLock<HashMap<ArrayId, (DynSizeFn, DynEncodeFn)>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        inventory::iter::<RowEncodeRegistration>
            .into_iter()
            .map(|r| ((r.id)(), (r.size, r.encode)))
            .collect()
    });
    map.get(id).copied()
}
