// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Thin shims that route point-fn dispatch through the encoding vtable.
//!
//! Encodings opt into point-fn awareness by overriding
//! [`OperationsVTable::point_scalar_at`](crate::array::OperationsVTable::point_scalar_at);
//! the default delegates to the legacy `scalar_at`, so unported encodings keep
//! working unchanged.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::point_fn::PointDispatch;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Dispatch `scalar_at` through the encoding's `point_scalar_at` vtable hook.
pub(crate) fn dispatch_scalar_at<D: PointDispatch>(
    arr: &ArrayRef,
    idx: usize,
    d: &mut D,
) -> VortexResult<Scalar> {
    arr.point_execute_scalar(idx, d)
}

/// Dispatch `search_sorted` through the encoding's `point_search_sorted` vtable
/// hook (default = generic binary search via `d.scalar_at`).
pub(crate) fn dispatch_search_sorted<D: PointDispatch>(
    arr: &ArrayRef,
    value: &Scalar,
    side: SearchSortedSide,
    d: &mut D,
) -> VortexResult<SearchResult> {
    arr.point_execute_search_sorted(value, side, d)
}
