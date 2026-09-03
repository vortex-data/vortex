// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![cfg(target_endian = "little")]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::int_plus_one,
    clippy::manual_slice_size_calculation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::panic,
    clippy::unwrap_used
)]

//! OnPair: dictionary-based short-string compression for fast random access.
//!
//! Rust port of the algorithm described in
//! [arXiv:2508.02280](https://arxiv.org/abs/2508.02280). OnPair replaces
//! recurring substrings ("tokens") with fixed-width integer codes into a
//! dictionary; decoding is a gather-copy, so individual rows decode in
//! isolation at no extra cost.
//!
//! # Layout
//! A compressed [`Column`] is a [`CompactDictionary`] (token bytes + offsets), a
//! code stream (one [`Token`] per emitted token), and a row layer (offsets into
//! the code stream). Borrow it as a [`ColumnView`] — or build a view directly
//! from buffers deserialized from storage — and decode into a caller-owned buffer:
//! the whole column with [`ColumnView::decompress_into`], one row with
//! [`ColumnView::decompress_row_into`], or [`decode_into`] over a reusable
//! [`WideDictionary`]. The caller owns buffer sizing: size it from
//! [`ColumnView::decoded_len`] or [`ColumnView::row_decoded_len`] (plus
//! [`DECODE_PADDING`]).
//!
//! # Examples
//! ```
//! use onpair::{Column, DECODE_PADDING, DEFAULT_CONFIG};
//! use std::mem::MaybeUninit;
//!
//! // Compress an Arrow (bytes, offsets) value pair.
//! let bytes = b"catdogcat";
//! let offsets: [u32; 4] = [0, 3, 6, 9];
//! let col = Column::compress(bytes, &offsets, DEFAULT_CONFIG).unwrap();
//! let view = col.view();
//!
//! // Random-access a single row into a caller buffer — no full decode, no
//! // wide-table build. Size it from the row's decoded length plus DECODE_PADDING.
//! let mut row = vec![MaybeUninit::uninit(); view.row_decoded_len(1) + DECODE_PADDING];
//! // SAFETY: `col` is valid by construction and `row` is sized as required.
//! let n = unsafe { view.decompress_row_into(1, &mut row) };
//! assert_eq!(unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), n) }, b"dog");
//!
//! // Bulk-decode the whole column into a caller buffer, sized from the decoded length.
//! let mut buf = vec![MaybeUninit::uninit(); view.decoded_len() + DECODE_PADDING];
//! // SAFETY: `col` is freshly compressed (valid by construction) and `buf` is sized.
//! let n = unsafe { view.decompress_into(&mut buf) };
//! let out = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), n) };
//! assert_eq!(out, b"catdogcat");
//! ```
//!
//! The trained encoder is also available directly via [`Parser`], to reuse one
//! dictionary across several corpora.

mod column;
mod core;
mod decoding;
mod encoding;
pub mod search;

#[cfg(test)]
mod test_corpus;

pub use crate::column::{Column, ColumnView};
pub use crate::core::dictionary::{
    CompactDictionary, CompactDictionaryView, Dictionary, DictionaryStorage, DictionaryView,
    OwnedDictionaryStorage, WideDictionary, WideDictionaryView, code_bits_for_num_tokens,
};
pub use crate::core::offset::Offset;
pub use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
pub use crate::core::validate::InvalidColumn;
pub use crate::decoding::{
    DECODE_PADDING, OutputTooSmall, decode_into, decoded_len, try_decode_into,
};
pub use crate::encoding::config::{Config, DEFAULT_CONFIG, Error, MaxDictBits, Threshold};
pub use crate::encoding::parser::Parser;

/// Compress an Arrow `(bytes, offsets)` value pair end-to-end. Equivalent to
/// `Parser::train(..)?.parse(..)`, but validates the offsets once instead of in
/// both the train and parse steps. `offsets` has `n + 1` entries.
///
/// # Errors
/// [`Error::InvalidArg`] if `offsets` is empty or its last entry exceeds
/// `bytes.len()`.
pub fn compress<O: Offset>(bytes: &[u8], offsets: &[O], cfg: Config) -> Result<Column<O>, Error> {
    encoding::parser::validate_offsets(bytes, offsets)?;
    let parser = Parser::train_unchecked(bytes, offsets, cfg);
    Ok(parser.parse_unchecked(bytes, offsets))
}
