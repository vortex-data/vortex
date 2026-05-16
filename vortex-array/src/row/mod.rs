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
//! Each scalar function supports per-encoding fast paths: encodings such as `Constant`,
//! `Dict`, and `RunEnd` can short-circuit canonicalization by implementing
//! [`RowSizeKernel`] / [`RowEncodeKernel`] that write directly into the shared output
//! buffers.
//!
//! The user-facing entry point is [`convert_columns`].

pub mod codec;
pub mod convert;
pub mod encode;
pub mod options;
pub mod registry;
pub mod size;

#[cfg(test)]
mod tests;

pub use convert::compute_row_sizes;
pub use convert::convert_columns;
pub use encode::RowEncode;
pub use encode::RowEncodeKernel;
pub use options::RowEncodeOptions;
pub use options::SortField;
pub use registry::RowEncodeRegistration;
pub use size::RowSize;
pub use size::RowSizeKernel;
