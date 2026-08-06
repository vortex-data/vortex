// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce.
//!
//! Both traits are open, and this module holds one file per type family, so covering a new one is a
//! sibling file and every row function gains it. The families are not confined to this crate:
//! `vortex-tensor`'s `TensorRow` drills through an extension wrapper into its storage.
//!
//! The two directions are deliberately asymmetric. [`InputElement::Elem`] is a GAT, so an input row
//! can borrow out of the decoded column, while an [`OutputElement`] is one owned value written into
//! an [`ElementSink`](crate::scalar_fn::ElementSink). Runtime-shaped output uses a custom
//! [`OutputSink`](crate::scalar_fn::OutputSink) instead.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;

mod bool;

#[cfg(any(test, feature = "_test-harness"))]
mod conformance;
#[cfg(any(test, feature = "_test-harness"))]
pub use conformance::assert_element_conforms;

mod primitive;

mod tuple;
pub use tuple::ElementTuple;
pub(super) use tuple::batch_constant;

/// An element type that can be read row-wise out of an input column.
pub trait InputElement: 'static {
    /// The decoded column representation supporting `O(1)` row access.
    type Column;

    /// The view of a varying decoded column read by the hot row loop.
    ///
    /// This may borrow a cheaper representation than [`Column`](Self::Column). Primitive elements,
    /// for example, expose a slice so its pointer and length are loop invariants rather than
    /// re-reading a [`Buffer`](vortex_buffer::Buffer) descriptor for every row.
    type Varying<'a>;

    /// The borrowed element value handed to the row closure a [`RowFn`](crate::scalar_fn::RowFn)
    /// visits with.
    type Elem<'a>;

    /// Whether [`decode`](Self::decode) and [`get`](Self::get) tolerate rows that are null in the
    /// input.
    ///
    /// Arrays only guarantee their contents for _valid_ rows, so this is `false` for any element
    /// that follows an offset or pointer stored in the array: behind a null row that value is
    /// arbitrary and may not address anything. Reading a whole value out of a flat buffer is `true`,
    /// since the value is garbage but the read cannot fault.
    ///
    /// Dense execution requires this of every argument; otherwise the row layer executes only
    /// valid rows.
    const DENSE_SAFE: bool = false;

    /// Whether [`decode`](Self::decode) can fail on _legal_ input data.
    ///
    /// `false` for an element read straight out of a buffer: decoding can still fail for
    /// infrastructural reasons (IO, allocation), but never because of the values. `true` for an
    /// element that parses its bytes, since a malformed WKB geometry in a _valid_ row is a domain
    /// error, which makes a function over that element
    /// [fallible](crate::scalar_fn::ScalarFnVTable::is_fallible) however infallible its own row
    /// computation is.
    const DECODE_FALLIBLE: bool = true;

    /// A relative unit count for per-row decode work avoided by filtering this argument first.
    ///
    /// Use `1` for an element whose decode _parses_ every row (a geometry built from coordinate
    /// storage): decoding only the survivors of a sparse validity mask is genuinely cheaper than
    /// decoding everyone. Keep the default `0` for a bulk canonicalization (bytes, bools,
    /// primitives), whose decode is a memcpy-shaped pass that filtering barely shrinks. Larger
    /// values may express a proportionally more expensive decode.
    ///
    /// The lifting reads this when it picks a null strategy for a batch with a mixed
    /// validity mask: filtering the inputs first only pays off when it shrinks a per-row decode,
    /// so elements that leave this at zero always take the cheaper branch-and-skip strategy.
    /// Getting it wrong is a performance bug, never a correctness bug.
    const FILTERED_DECODE_COST: usize = 0;

    /// Validate that `dtype` is an acceptable input column dtype for this element type.
    fn validate(dtype: &DType) -> VortexResult<()>;

    /// Decode `array` into its column representation. Called once per batch.
    ///
    /// This is where every per-batch cost belongs: resolving the dtype, downcasting the buffer,
    /// checking the ptype, and anything else that does not vary by row. [`Column`](Self::Column) is
    /// the type to widen if that means carrying more, since it is chosen by the element.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;

    /// Decode `array` _without_ assuming every row is valid, or `Ok(None)` when this element
    /// cannot for this particular array.
    ///
    /// An element with [`DENSE_SAFE`](Self::DENSE_SAFE) set **should not** override this: its
    /// ordinary decode already tolerates null payloads, so the default is already correct and an
    /// override just restates it. Overriding is for an element that is *not* dense-safe but can
    /// still write an arbitrary placeholder into null slots; the caller guarantees
    /// [`get`](Self::get) is never called for such a row. It is what the branch-and-skip null
    /// strategy decodes with.
    ///
    /// Return `Ok(None)` rather than an error when an array has no null-tolerant decode; the lifting
    /// falls back to the filter strategy.
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
    /// `O(1)` is necessary but **not sufficient**: this must not repeat work that is constant across
    /// the batch, however cheap that work looks per call. An `O(1)` ptype check and buffer downcast
    /// per row cost `l2_norm` 2x at width 2, invisible in the call because it read like a getter. Do
    /// that work in [`decode`](Self::decode) and leave this an offset computation.
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;

    /// Borrow the representation used when this argument varies within the batch.
    ///
    /// Called once before the hot loop. Constants do not use this view because the tuple adapter
    /// keeps their one-row decoded representation separate.
    fn varying(column: &Self::Column) -> Self::Varying<'_>;

    /// Number of rows addressable through a [`Varying`](Self::Varying) view.
    fn varying_len(column: &Self::Varying<'_>) -> usize;

    /// Read one row from a [`Varying`](Self::Varying) view.
    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a;
}

/// An element type that a row computation can produce, buildable into an all-valid column.
///
/// [`Clone`] is required so [`ElementSink`](crate::scalar_fn::ElementSink) can allocate through
/// `vec![placeholder; rows]`, which is what lets a zero placeholder reach the allocator's zeroed
/// path instead of costing a write pass over the output.
pub trait OutputElement: 'static + Sized + Clone {
    /// The dtype of columns built from this element type. Must be non-nullable: nullability is
    /// derived from the inputs by the lifting.
    ///
    /// Taking no arguments confines an element's dtype to a property of its Rust type, so an output
    /// whose dtype depends on runtime data (a tensor, whose dtype carries its shape) cannot be an
    /// element. Such an output uses an [`OutputSink`](crate::scalar_fn::OutputSink), whose
    /// [`sink_dtype`](crate::scalar_fn::OutputSink::sink_dtype) does see the input dtypes.
    fn element_dtype() -> DType;

    /// Build a column from one value per row. Called once per batch.
    fn build(values: Vec<Self>) -> ArrayRef;

    /// An arbitrary value of this element, pre-filled into the output slots that the
    /// branch-and-skip null strategy skips.
    ///
    /// The value is never observable: the lifting masks every slot holding it before the
    /// result escapes. It only has to be cheap to construct and legal to
    /// [`build`](Self::build) with.
    fn placeholder() -> Self;
}
