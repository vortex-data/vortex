// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::panic,
    clippy::unwrap_used
)]

//! Pure-Rust port of [`onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp).
//!
//! ## Scope
//!
//! - **Training + encoding** (mirrors `onpair_cpp`'s `OnPairColumn::compress`):
//!   BPE-style dictionary discovery + LSB-first bit-packed token encoding.
//!   Public entry point: [`Column::compress`].
//! - **Decoding**: point [`Column::decompress_row`] and bulk
//!   [`Column::decode_all`].
//! - **Predicates**: [`Column::equals_bitmap`], [`Column::starts_with_bitmap`],
//!   [`Column::contains_bitmap`], and multi-pattern
//!   [`Column::multi_pattern_bitmap`] (Aho-Corasick). All emit LSB-first
//!   packed bitmaps in the same layout `vortex-onpair-sys` does.
//! - **Boolean combinators**: [`bitmap_and`], [`bitmap_or`], [`bitmap_not`]
//!   compose any predicate results.
//! - **Raw access**: [`Column::parts`] borrows the dictionary, packed codes,
//!   and per-row boundaries — feeds directly into `vortex-onpair`'s
//!   SIMD-friendly decode and predicate kernels.
//!
//! ## Deliberately not ported
//!
//! The compressed-domain `TokenAutomaton` (`KmpAutomaton` / `PrefixAutomaton`
//! / `EqAutomaton` / `AhoCorasickAutomaton`) from `onpair_cpp/search/`.
//! The predicate APIs here decompress each row and run a byte-level match.
//! Same observable result; the C++ design is faster on very large columns
//! because it scans the bit-packed stream directly and composes
//! `A && !B` in one pass over the tokens.
//!
//! ## Layout produced by `Column::parts`
//!
//! ```text
//! dict_bytes:        concatenated token bytes, unpadded
//! dict_offsets:      &[u32], length dict_size + 1
//! codes_packed:      LSB-first bit-packed token stream (no sentinel word)
//! codes_boundaries:  &[u32], length num_rows + 1
//! bits:              9..=16
//! ```

pub mod bit_unpack;
pub mod bit_writer;
pub mod column;
pub mod combinators;
pub mod config;
pub mod decoder;
pub mod dict;
pub mod lpm;
pub mod parser;
pub mod search;
pub mod store;
pub mod trainer;
pub mod types;

#[cfg(test)]
mod test_corpus;

pub use bit_unpack::{read_bits_lsb, unpack_codes_to_u16};
pub use column::{Column, Parts};
pub use combinators::{
    bitmap_and, bitmap_and_in_place, bitmap_len, bitmap_not, bitmap_not_in_place, bitmap_or,
    bitmap_or_in_place, bitmap_popcount,
};
pub use config::{
    DEFAULT_DICT12_CONFIG, DynamicThreshold, Error, FixedThreshold, OnPairTrainingConfig,
    ThresholdSpec, TrainingConfig,
};
pub use decoder::{decode_all, decode_codes, decompress_row, row_codes};
pub use dict::Dictionary;
pub use lpm::LongestPrefixMatcher;
pub use parser::parse;
pub use search::{
    contains_bitmap, empty_bitmap, equals_bitmap, get_bit, multi_pattern_bitmap,
    starts_with_bitmap,
};
pub use store::Store;
pub use trainer::{TrainResult, train};
pub use types::{
    BitWidth, ByteSpan, MAX_TOKEN_SIZE, StreamSpan, Token, TokenRange, is_valid_bits,
    max_dict_size,
};
