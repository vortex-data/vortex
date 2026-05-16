// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding-aware dispatch table.
//!
//! Routes `PointDispatch::scalar_at` (and friends) to encoding-specific
//! implementations when one exists, falling back to the existing
//! `OperationsVTable::scalar_at` otherwise.
//!
//! ## Phase 1c
//!
//! This module is a pragmatic in-crate dispatch table: it checks each opt-in
//! encoding via [`ArrayRef::as_typed`] in priority order. As more in-crate
//! encodings are ported, they get an `as_typed::<X>()` arm here.
//!
//! Phase 2 will replace this with a proper cross-crate kernel registry (likely
//! an additional method on `OperationsVTable` with a default that falls back to
//! `scalar_at`), so encodings in external crates (PCO, FSST, ZSTD, ALP, …)
//! can opt in without vortex-array knowing about them.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::Slice;
use crate::arrays::slice::SliceArrayExt;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Dispatch a `scalar_at` call to the encoding-specific point-fn kernel, or
/// fall back to `arr.execute_scalar` (the legacy OperationsVTable path).
///
/// Called by `PointRuntime::scalar_at` and `PointSession::scalar_at`. Note that
/// `PointSession::scalar_at` *also* wraps this in a scalar-cache lookup; this
/// function is the kernel invocation step that produces a value on cache miss.
pub(crate) fn dispatch_scalar_at<D: PointDispatch + ?Sized>(
    arr: &ArrayRef,
    idx: usize,
    d: &mut D,
) -> VortexResult<Scalar> {
    // Slice: rewrite to (child, idx + offset) and recurse via d. The recursion
    // ensures any session cache is consulted at the child level too.
    if let Some(slice) = arr.as_typed::<Slice>() {
        let offset = slice.slice_range().start;
        let child = slice.child().clone();
        return d.scalar_at(&child, offset + idx);
    }
    // Fallback: legacy OperationsVTable::scalar_at.
    arr.execute_scalar(idx, d.ctx())
}

/// Dispatch a `search_sorted` call. Phase 1c has no encoding-specific overrides
/// yet, so this is unconditionally `generic_search_sorted`. The hook exists so
/// later phases can intercept (Dict, RunEnd, Chunked, FoR, Constant, Sequence).
pub(crate) fn dispatch_search_sorted<D: PointDispatch + ?Sized>(
    arr: &ArrayRef,
    value: &Scalar,
    side: SearchSortedSide,
    d: &mut D,
) -> VortexResult<SearchResult> {
    super::algorithms::generic_search_sorted(arr, value, side, d)
}
