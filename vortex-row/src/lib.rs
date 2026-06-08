// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-oriented byte encoding for Vortex arrays.
//!
//! This crate converts one or more columnar arrays into a single `ListView<u8>` array whose
//! row byte slices can be compared lexicographically. The byte ordering matches tuple
//! ordering of the input values under the requested [`RowSortFieldOptions`] settings, making the
//! representation useful for sort keys and other row-key operations. It is the Vortex analogue
//! of `arrow-row`.
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
//! <div class="warning">
//!
//! The row encoding format is **experimental**. Its byte layout, supported type set, and
//! edge-case semantics may change between Vortex releases. Do not persist these bytes or
//! depend on them as a stable interchange format.
//!
//! </div>
//!
//! # Byte-layout reference
//!
//! This is a schema-aware row-key format: the bytes carry no type tags, field names, or sort
//! options, so two encoded rows are comparable only when produced from the same schema and the
//! same per-column [`RowSortFieldOptions`].
//!
//! ## Order property
//!
//! For a fixed schema with columns `c0..cn` and per-column sort fields `f0..fn`:
//!
//! ```text
//! encode(row_a) < encode(row_b)
//!   <=>  (row_a.c0, .., row_a.cn) < (row_b.c0, .., row_b.cn)
//! ```
//!
//! under the requested direction and null placement of each column. This holds because (1)
//! every supported value is encoded so its bytes sort in the same order as the value, and (2)
//! fields are concatenated left to right, so lexicographic byte comparison performs tuple
//! comparison. `||` below means byte concatenation, `BE(x)` the fixed-width big-endian bytes of
//! `x`, and `!bytes` the bitwise complement of every byte.
//!
//! ## Field options
//!
//! Each input column carries a [`RowSortFieldOptions`] `{ descending, nulls_first }`.
//! `descending` reverses the order of non-null values; `nulls_first` is independent of
//! `descending`, so nulls can sort before or after non-nulls in either direction.
//!
//! ## Sentinels
//!
//! A leading sentinel byte classifies nullness (and, for variable-width values, empty vs
//! non-empty) before any value bytes are compared. The sentinel itself is never inverted for
//! `descending`, which keeps null placement independent of sort direction.
//!
//! | Family | Case | Asc, nulls first | Desc, nulls first | Asc, nulls last | Desc, nulls last |
//! | --- | --- | --- | --- | --- | --- |
//! | Fixed-width | Null | `0x00` | `0x00` | `0x02` | `0x02` |
//! | Fixed-width | Non-null | `0x01` | `0x01` | `0x01` | `0x01` |
//! | Variable-width | Null | `0x00` | `0x00` | `0xFF` | `0xFF` |
//! | Variable-width | Empty | `0x01` | `0xFE` | `0x01` | `0xFE` |
//! | Variable-width | Non-empty | `0x02` | `0xFD` | `0x02` | `0xFD` |
//!
//! Fixed-width sentinels are used by null, boolean, primitive, decimal, struct, and fixed-size
//! list values; variable-width sentinels by UTF-8 and binary values.
//!
//! ## Per-type encoding
//!
//! - **Null**: just the fixed-width sentinel, no body.
//! - **Boolean**: `sentinel || value_byte`, where `false = 0x01`, `true = 0x02` (inverted for
//!   descending). Null bodies are a single zero byte.
//! - **Unsigned integer** (`u8`–`u64`): `0x01 || BE(value)` (`!BE(value)` descending). Null
//!   bodies are `width(T)` zero bytes.
//! - **Signed integer** (`i8`–`i64`, and `i128` decimal storage): flip the sign bit of
//!   `BE(value)` so negatives sort before non-negatives, then apply the descending complement.
//! - **Floating point** (`f16`/`f32`/`f64`): treat the IEEE bits as unsigned; flip the top bit
//!   for non-negative values and all bits for negative, then big-endian. Yields total-ordering
//!   semantics (`-0.0 < +0.0`, NaNs ordered by bit pattern).
//! - **Decimal**: encoded as its scaled signed-integer storage value at the *precision-minimal*
//!   width (`1..=2 -> i8`, `3..=4 -> i16`, `5..=9 -> i32`, `10..=18 -> i64`, `19..=38 -> i128`),
//!   using the signed-integer encoding. `Decimal256` is unsupported. The width is a pure
//!   function of the precision, so storage physically wider than the precision requires is
//!   narrowed losslessly before encoding (precision bounds the magnitude of every valid value).
//! - **UTF-8 / Binary**: a variable-width sentinel, and for non-empty values a block-structured
//!   body. Each block is 32 data bytes plus a marker: non-final full blocks use marker `0xFF`,
//!   the final block is zero-padded to 32 bytes with a marker giving its real length (`1..=32`).
//!   Descending inverts the data bytes, padding, and markers. This preserves prefix order.
//! - **Struct / Fixed-size list**: an outer fixed-width sentinel followed by the children
//!   encoded recursively in order with the parent's options. A null parent emits a *canonical
//!   null body* (fixed-width children contribute their fixed null encoding; variable-width
//!   children collapse to one null sentinel byte) so two null parents are byte-equal regardless
//!   of underlying child data. A composite is fixed-width only when all of its children are.
//!
//! ## Output layout
//!
//! The result is a `ListView<u8>`: a single contiguous `elements` buffer holding every row's
//! bytes, with per-row `offsets` and `sizes`. Rows are not self-describing without `sizes`,
//! since a variable-width field can make one row longer than another. The sizing pass computes
//! `sizes` before writing, and the same array doubles as the per-row write cursor.
//!
//! Supported logical types are nulls, booleans, primitive integers and floats, decimals up to
//! 128 bits, UTF-8 and binary values, structs, and fixed-size lists. Extension, variant,
//! union, and variable-size list arrays are rejected because this crate does not define an
//! ordering for them.
//!
//! See `docs/specs/row-encoding.md` for the formal specification and a fully worked example
//! row.

mod codec;
mod encode;
mod encoder;
mod options;
mod size;

#[cfg(test)]
mod tests;

pub use encode::RowEncode;
pub use encoder::RowEncoder;
pub use options::RowEncodingOptions;
pub use options::RowSortFieldOptions;
pub use size::RowSize;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_error::VortexResult;
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

/// Convert N columnar arrays into a single row-oriented [`ListViewArray`] of `u8` whose bytes
/// are lexicographically comparable in the same order as a tuple comparison of the input
/// values according to `fields`. Convenience wrapper over [`RowEncoder::encode`].
pub fn convert_columns(
    cols: &[ArrayRef],
    fields: &[RowSortFieldOptions],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    RowEncoder::new(fields.iter().copied()).encode(cols, ctx)
}

/// Like [`convert_columns`] but takes a prebuilt [`RowEncodingOptions`].
pub fn convert_columns_with_options(
    cols: &[ArrayRef],
    options: &RowEncodingOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    RowEncoder::with_options(options.clone()).encode(cols, ctx)
}
