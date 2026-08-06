// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function vtable machinery.
//!
//! This module contains the [`ScalarFnVTable`] trait and all built-in scalar function
//! implementations. Expressions ([`crate::expr::Expression`]) reference scalar functions
//! at each node.
//!
//! # Choosing a trait
//!
//! Two traits reach this vtable, and [`RowFn`] derives the whole of [`ScalarFnVTable`] from a row
//! closure. Implement `RowFn` when the function fits it, and `ScalarFnVTable` when it does not.
//!
//! [`RowFn`] is for a kernel whose value at a row is determined by that row alone, and which has to
//! read every row anyway: the arithmetic operators over primitive columns, `vortex.tensor.l2_norm`,
//! `vortex.tensor.inner_product`, `vortex.tensor.cosine_similarity`, `vortex.geo.distance`,
//! `vortex.geo.contains`. Name the element types and write the row closure, and the rest is
//! derived, including which rows get visited.
//!
//! Its *input* side is open. [`InputElement::Elem`] is a GAT, so an element can hand the closure
//! borrowed variable-length data (a byte-string element yielding `&[u8]`) or drill through a wrapper
//! (`vortex-tensor`'s `TensorRow` yields a slice of an extension array's storage). Covering a new
//! type family, a list row included, is one impl.
//!
//! Its output is always an [`OutputSink`], allocated once per batch and handing the closure one row
//! to write. [`ElementSink`] is the standard sink for one owned [`OutputElement`] per row. A custom
//! sink carries runtime-shaped output, such as a tensor whose width comes from its input dtype, or a
//! future string transform appending every row into one shared byte buffer.
//!
//! When part of the kernel's work depends only on an operand that is constant for the batch (the
//! norm of a broadcast query vector, a prepared form of a constant geometry), do that work in
//! [`RowVisitor::visit_prepared_into`]'s once-per-batch prepare step. Pass `|_| ()` when there is
//! nothing to prepare. Prepare **must not** be load-bearing for validation: an empty batch decodes
//! every operand as non-constant, so a prepare that validated its constant would silently not run.
//!
//! Null handling is derived too, null-strategy selection included: a nullable batch runs densely
//! (compute every row, mask after), by branch-and-skip (decode full length, compute only the
//! conjoined-valid rows, mask after), or by filtering (shrink the inputs to the valid rows,
//! compute, scatter back), and the framework picks per batch. Function authors do nothing. The one
//! input to that choice an element controls is [`InputElement::FILTERED_DECODE_COST`]: set it when
//! decoding a column does expensive per-row work (parsing a geometry), so sparse batches keep the
//! filter strategy's shrunken decode. Costs from separate arguments are additive.
//!
//! Two things no output sink covers, and they are what actually send a function to
//! [`ScalarFnVTable`]:
//!
//! - **A result that aliases an input.** Sinks own their output bytes. Trimming strings is the
//!   example, where the ideal kernel keeps the input's data buffer and writes new views over it,
//!   copying no bytes, which only a columnar kernel can express.
//! - **A null result for a non-null row.** Sinks build an all-valid column, so
//!   `vortex.list.sum` cannot be a row function: a valid empty list sums to null.
//!
//! [`ScalarFnVTable`] takes the whole column instead, and everything a row function gets derived is
//! then hand-written: null propagation, constant folding, nullability, validity, and options serde.
//! Besides the two cases above and the functions that are simply not strict (Kleene logic, or a
//! strictness that depends on the options), reach for it when a row loop *could* express the
//! function but would do avoidable work:
//!
//! - **The answer is already an array, or is one value for the whole column.**
//!   `vortex.list.length` hands back a `ListViewArray`'s sizes child, and a single `ConstantArray`
//!   for a `FixedSizeListArray`. A row loop would rebuild that one `u64` at a time, even given a
//!   list-length element that reads the size out of the layout rather than the list.
//! - **A row is not the natural unit of work.** `vortex.not` is one `!` per 64-bit word, in place
//!   when the bit buffer is unshared, against 64 loop iterations and 64 bit writes, and its
//!   encoding-aware fallback pushes the inversion down instead of canonicalizing.
//! - **The row's value is cheaper to read than the row.** `vortex.byte_length` was tried as a row
//!   function and measured 7.6x slower than its columnar implementation, because the length is a
//!   field of the view and the row loop paid to resolve the bytes it never looked at. Being
//!   row-determined is necessary but not sufficient.

use vortex_session::registry::Id;

use crate::scalar_fn::fns::byte_length::ByteLength;
use crate::scalar_fn::fns::ext_storage::ExtStorage;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;

mod vtable;
pub use vtable::*;

mod plugin;
pub use plugin::*;

mod foreign;
pub use foreign::*;

mod typed;
pub use typed::*;

mod erased;
pub use erased::*;

mod options;
pub use options::*;

mod signature;
pub use signature::*;

mod row;
pub use row::*;

pub mod fns;
pub mod internal;
pub mod session;

/// A unique identifier for a scalar function.
pub type ScalarFnId = Id;

/// Private module to seal [`typed::DynScalarFn`].
mod sealed {
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::typed::TypedScalarFnInstance;

    /// Marker trait to prevent external implementations of [`super::typed::DynScalarFn`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynScalarFn`].
    impl<V: ScalarFnVTable> Sealed for TypedScalarFnInstance<V> {}
}

/// A scalar function has a negative cost if applying it to an array and
/// canonicalizing is cheaper than canonicalizing an array and applying it.
///
/// Example of negative cost expressions are byte_length(), ext_storage(), and get_item() since
/// they don't depend on input size.
///
/// Example of non-negative cost expression is like() as it's linear over
/// individual input.
pub fn is_negative_cost(id: ScalarFnId) -> bool {
    id == ScalarFnVTable::id(&ByteLength)
        || id == ScalarFnVTable::id(&ExtStorage)
        || id == ScalarFnVTable::id(&GetItem)
        || id == ScalarFnVTable::id(&Literal)
}
