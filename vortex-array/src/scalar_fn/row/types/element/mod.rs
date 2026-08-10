// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce.
//!
//! [`InputElement::Elem`] may borrow from its decoded column. [`OutputElement`] is returned by an
//! owned row computation; runtime-shaped output uses an
//! [`OutputSink`](crate::scalar_fn::OutputSink).

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;

mod bool;

mod primitive;

mod tuple;
pub use tuple::ElementTuple;
pub use tuple::IndexedElementTuple;
pub use tuple::batch_constant;

/// An element type that can be read row-wise out of an input column.
///
/// # Safety
///
/// For every view returned by [`varying`](Self::varying), every index below
/// [`varying_len`](Self::varying_len) **must** satisfy the safety contract of
/// [`get_varying_unchecked`](Self::get_varying_unchecked). Shared execution relies on this proof to
/// perform unchecked reads after one pre-loop length check.
pub unsafe trait InputElement: 'static {
    /// The decoded column representation supporting `O(1)` row access.
    type Column;

    /// The view of a varying decoded column read by the hot row loop.
    ///
    /// This may borrow a cheaper representation than [`Column`](Self::Column). Primitive elements,
    /// for example, expose a slice so its pointer and length are loop invariants rather than
    /// re-reading a [`Buffer`](vortex_buffer::Buffer) descriptor for every row.
    type Varying<'a>;

    /// The borrowed element value handed to a row closure.
    type Elem<'a>;

    /// Whether every dense decode and access path tolerates rows that are null in the input.
    ///
    /// Arrays only guarantee payloads for valid rows. This is `false` when a null row's stored
    /// offset or pointer may not address anything, and `true` only when [`decode`](Self::decode),
    /// [`get`](Self::get), [`varying`](Self::varying), [`varying_len`](Self::varying_len), and
    /// [`get_varying`](Self::get_varying) remain safe and correct for null rows.
    ///
    /// Dense execution requires this of every argument; otherwise the row layer executes only
    /// valid rows.
    ///
    /// Dense execution can pass unspecified values from null rows. The closure must be total over
    /// every stored value: it cannot panic or cause side effects beyond its declared output.
    const DENSE_SAFE: bool = false;

    /// Whether [`decode`](Self::decode) can fail on _legal_ input data.
    ///
    /// This excludes infrastructural failures such as IO or allocation. Set it when legal input may
    /// contain a value that the decoder rejects.
    const DECODE_FALLIBLE: bool = true;

    /// Validate that `dtype` is an acceptable input column dtype for this element type.
    fn validate(dtype: &DType) -> VortexResult<()>;

    /// Decode `array` into its column representation. Called once per batch.
    ///
    /// Hoist dtype checks, downcasts, and other batch-invariant work into this method.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;

    /// Decode `array` _without_ assuming every row is valid, or `Ok(None)` when this element
    /// cannot for this particular array.
    ///
    /// An element with [`DENSE_SAFE`](Self::DENSE_SAFE) set **should not** override this: its
    /// ordinary decode already tolerates null payloads, so the default is already correct and an
    /// override just restates it. Overriding is for an element that is _not_ dense-safe but can
    /// still write an arbitrary placeholder into null slots; the caller guarantees
    /// [`get`](Self::get) is never called for such a row. The skip-invalid strategy uses this
    /// representation to avoid filtering the input.
    ///
    /// Return `Ok(None)` rather than an error when an array has no null-tolerant decode; the
    /// batch execution falls back to the filter strategy.
    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        if Self::DENSE_SAFE {
            Self::decode(array, ctx).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Read the element at `index`, the one function called once per row.
    ///
    /// This must not repeat work that is constant across the batch; do that work in
    /// [`decode`](Self::decode).
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;

    /// Borrow the representation used when this argument varies within the batch.
    ///
    /// Called once before the hot loop. Constants do not use this view because the tuple adapter
    /// keeps their one-row decoded representation separate.
    fn varying(column: &Self::Column) -> Self::Varying<'_>;

    /// Number of rows addressable through a [`Varying`](Self::Varying) view.
    ///
    /// Every index below this length must be valid for
    /// [`get_varying_unchecked`](Self::get_varying_unchecked).
    fn varying_len(column: &Self::Varying<'_>) -> usize;

    /// Read one row from a [`Varying`](Self::Varying) view.
    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a;

    /// Read one row without checking that `index` is in bounds.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`varying_len`](Self::varying_len) for `column`.
    unsafe fn get_varying_unchecked<'a>(column: &Self::Varying<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        Self::get_varying(column, index)
    }
}

/// An owned row value that can be built into an all-valid column.
pub trait OutputElement: 'static + Sized {
    /// The dtype of columns built from this element type. **Must** be non-nullable: nullability is
    /// derived from the inputs by batch execution.
    ///
    /// Taking no arguments confines an element's dtype to a property of its Rust type, so an output
    /// whose dtype depends on runtime data (a tensor, whose dtype carries its shape) cannot be an
    /// element. Such an output uses an [`OutputSink`](crate::scalar_fn::OutputSink), whose
    /// [`sink_dtype`](crate::scalar_fn::OutputSink::sink_dtype) does see the input dtypes.
    fn element_dtype() -> DType;

    /// Build a column from one value per row. Called once per batch.
    fn build(values: Vec<Self>) -> ArrayRef;
}
