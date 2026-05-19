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
//! ## Quick start
//!
//! ```ignore
//! use onpair_lib::{Column, KmpAutomaton, OnPairTrainingConfig, and, not};
//!
//! let col = Column::compress(&bytes, &offsets, OnPairTrainingConfig {
//!     bits: 12, threshold: 0.5, seed: 42,
//! })?;
//!
//! // Compressed-domain predicates (single pass over the token stream):
//! let mut user  = KmpAutomaton::new(b"user",  col.dictionary());
//! let mut admin = KmpAutomaton::new(b"admin", col.dictionary());
//! let row_ids = col.scan(and(&mut user, not(&mut admin)));
//! ```
//!
//! ## Module map
//!
//! - [`Column`] — the entry point: train + compress, decompress, scan
//! - [`TokenAutomaton`] + [`EqAutomaton`] / [`PrefixAutomaton`] /
//!   [`KmpAutomaton`] / [`AhoCorasickAutomaton`] — compressed-domain predicates
//! - [`and`], [`or`], [`not`] — combinators
//! - [`Parts`] — borrow the raw `(dict, codes, boundaries, bits)` for
//!   downstream consumers (drop-in for `vortex-onpair-sys::Parts`)

pub mod aho_corasick;
pub mod automaton;
pub mod bits;
pub mod column;
pub mod config;
pub mod dict;
pub mod kmp;
pub mod lpm;
pub mod parser;
pub mod store;
pub mod tokenize;
pub mod trainer;
pub mod types;

#[cfg(test)]
mod test_corpus;

pub use aho_corasick::{AhoCorasickAutomaton, AhoCorasickTrie};
pub use automaton::{And, EqAutomaton, Negated, Or, PrefixAutomaton, TokenAutomaton, and, not, or};
pub use bits::{read_bits_lsb, unpack_codes_to_u16};
pub use column::{Column, Parts};
pub use config::{
    DEFAULT_DICT12_CONFIG, DynamicThreshold, Error, FixedThreshold, OnPairTrainingConfig,
    ThresholdSpec, TrainingConfig,
};
pub use dict::Dictionary;
pub use kmp::KmpAutomaton;
pub use lpm::LongestPrefixMatcher;
pub use parser::parse;
pub use store::Store;
pub use tokenize::{tokenize, tokenize_with};
pub use trainer::{TrainResult, train};
pub use types::{
    BitWidth, ByteSpan, MAX_TOKEN_SIZE, StreamSpan, Token, TokenRange, is_valid_bits, max_dict_size,
};
