// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-oriented byte encoder, analogous to Apache Arrow's `arrow-row` crate.
//!
//! The encoder converts N columnar arrays into a single `List<u8>` array where each row's
//! bytes are lexicographically comparable in the same order as a tuple comparison of the
//! original values. This is useful for sorting, hashing into row containers, and other
//! operations that benefit from a sort-friendly opaque byte representation of a multi-column
//! key.
//!
//! Two variadic scalar functions drive the implementation:
//! - [`RowSize`] computes per-row byte sizes across all N input columns.
//! - [`RowEncode`] writes the row-encoded bytes into a single `ListView<u8>` accumulator
//!   in one left-to-right pass.
//!
//! Each scalar function exposes a per-encoding fast-path trait
//! ([`RowSizeKernel`] / [`RowEncodeKernel`]) for downstream encodings to plug into; PR 3
//! adds in-crate impls for `Constant`, `Dict`, and `Patched` and an inventory-based
//! registry for external encodings.
//!
//! The user-facing entry point is [`convert_columns`].
//!
//! Row-encoding scalar functions are not registered in the default
//! [`VortexSession`]. Call [`initialize`] on a session to make `RowSize` and `RowEncode`
//! available via the expression layer.

pub mod codec;
pub mod convert;
pub mod encode;
pub mod options;
pub mod size;

#[cfg(test)]
mod tests;

pub use convert::compute_row_sizes;
pub use convert::convert_columns;
pub use encode::RowEncode;
pub use encode::RowEncodeKernel;
pub use options::RowEncodeOptions;
pub use options::SortField;
pub use size::RowSize;
pub use size::RowSizeKernel;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

/// Register the row-encoding scalar functions ([`RowSize`] and [`RowEncode`]) on the given
/// session.
///
/// Call once on session construction if you want row encoding available via the expression
/// layer or via [`convert_columns`].
pub fn initialize(session: &VortexSession) {
    session.scalar_fns().register(RowSize);
    session.scalar_fns().register(RowEncode);
}
