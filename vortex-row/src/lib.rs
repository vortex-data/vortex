// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-oriented byte encoding for Vortex arrays.
//!
//! This crate converts one or more columnar arrays into a single `ListView<u8>` array whose
//! row byte slices can be compared lexicographically. The byte ordering matches tuple
//! ordering of the input values under the requested [`RowSortField`] settings, making the
//! representation useful for sort keys and other row-key operations.
//!
//! The public entry points are:
//! - [`RowEncoder`], the primary API for encoding columns into row bytes.
//! - [`RowEncoder::row_sizes`], which computes the fixed and variable byte contributions
//!   without materializing the encoded rows.
//! - [`convert_columns`] and [`compute_row_sizes`], compatibility helpers around
//!   [`RowEncoder`].
//! - [`initialize`], which registers the [`RowSize`] and [`RowEncode`] scalar functions on a
//!   [`VortexSession`].
//!
//! Internally, encoding is split into two scalar functions. [`RowSize`] performs the sizing
//! pass and classifies fixed-width versus variable-width input columns. [`RowEncode`] uses
//! those sizes to allocate one contiguous elements buffer, then writes each column's bytes
//! into the per-row slots from left to right.
//!
//! Supported logical types are nulls, booleans, primitive integers and floats, decimals up to
//! 128 bits, UTF-8 and binary values, structs, and fixed-size lists. Extension, variant,
//! union, and variable-size list arrays are rejected because this crate does not define an
//! ordering for them.
//!
//! The byte-level format is documented in the row encoding spec:
//! <https://docs.vortex.dev/specs/row-encoding.html>.

mod codec;
mod encode;
mod encoder;
mod options;
mod size;

#[cfg(test)]
mod tests;

pub use encode::RowEncode;
pub use encoder::RowEncoder;
pub use encoder::compute_row_sizes;
pub use encoder::compute_row_sizes_with_options;
pub use encoder::convert_columns;
pub use encoder::convert_columns_with_options;
pub use options::RowEncodingOptions;
pub use options::RowSortField;
pub use size::RowSize;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

/// Register the row-encoding scalar functions ([`RowSize`] and [`RowEncode`]) on the given
/// session.
///
/// Call this during session construction when row encoding must be available through the
/// expression layer. The direct [`RowEncoder`] API constructs the scalar-function calls
/// itself and does not require global registration.
pub fn initialize(session: &VortexSession) {
    session.scalar_fns().register(RowSize);
    session.scalar_fns().register(RowEncode);
}
