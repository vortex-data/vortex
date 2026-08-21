// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed decoding and row access for one input column.
//!
//! [`InputElement`] separates invocation-wide decoding from the checked and unchecked access paths
//! used by row kernels.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::ViewLen;

/// An element type that can be read row-wise out of an input column.
///
/// # Safety
///
/// - For each view returned by [`view`](Self::view), every index below [`ViewLen::len`] **must**
///   satisfy the contract of [`get_from_view_unchecked`](Self::get_from_view_unchecked).
/// - The view length and its addressable indices **must** remain stable while the view exists.
///   Interior mutability exposed through an element **must not** change either property.
/// - Shared execution checks the length once before unchecked reads. Violating these requirements
///   can cause undefined behavior.
pub unsafe trait InputElement: 'static {
    /// The decoded column representation supporting `O(1)` row access.
    type Column;

    /// The row-loop view of a decoded column.
    ///
    /// This can borrow a cheaper representation than [`Column`](Self::Column). Primitive elements,
    /// for example, expose a slice so its pointer and length are loop invariants rather than
    /// re-reading a [`Buffer`](vortex_buffer::Buffer) descriptor for every row.
    type View<'a>: ViewLen;

    /// The borrowed element value handed to a row closure.
    type Elem<'a>;

    /// Whether every dense decode and access path tolerates rows that are null in the input.
    ///
    /// Arrays guarantee payloads only for valid rows. Set this to `true` only when every decode and
    /// access method remains safe for null rows. Dense execution can pass unspecified values from
    /// null rows to the row closure.
    const DENSE_SAFE: bool;

    /// Whether [`decode`](Self::decode) is infallible for _legal_ input data.
    ///
    /// This excludes infrastructural failures such as IO or allocation.
    const DECODE_INFALLIBLE: bool;

    /// Validate that `dtype` is an acceptable input column dtype for this element type.
    fn validate(dtype: &DType) -> VortexResult<()>;

    /// Decode `array` into its column representation.
    ///
    /// Called once per row-kernel invocation. Retrying a partially valid batch after a dense
    /// deferred error starts a second invocation over valid rows. Hoist dtype checks, downcasts,
    /// and other invocation-invariant work into this method.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;

    /// Whether [`decode_null_tolerant`](Self::decode_null_tolerant) can decode this array.
    ///
    /// The conservative default declines. An implementation whose ordinary decode is safe and
    /// infallible over null payloads can return `true`. Other implementations can inspect `array`
    /// and opt in only for supported representations.
    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(false)
    }

    /// Decode `array` _without_ assuming every row is valid, or return `Ok(None)` when this element
    /// cannot decode this particular array.
    ///
    /// Override this for a non-dense-safe representation that can still place safe placeholders in
    /// null slots. Valid-row execution never reads those slots.
    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        if Self::can_decode_null_tolerant(&array)? {
            Self::decode(array, ctx).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Read one row without repeating batch-constant work from [`decode`](Self::decode).
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;

    /// Borrow the representation used when this argument varies within the batch.
    ///
    /// Called once before the hot loop. Constants do not use this view because the tuple adapter
    /// keeps their one-row decoded representation separate.
    fn view(column: &Self::Column) -> Self::View<'_>;

    /// Read one row from a [`View`](Self::View).
    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a;

    /// Read one row without checking that `index` is in bounds.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`ViewLen::len`] for `view`.
    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        Self::get_from_view(view, index)
    }
}
