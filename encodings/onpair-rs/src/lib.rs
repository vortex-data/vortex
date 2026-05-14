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

//! Pure-Rust port of the training + encoding parts of
//! [`onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp).
//!
//! Scope: enough surface to drop in for the bits of `vortex-onpair-sys` that
//! `vortex-onpair` actually consumes — `Column::compress`, raw `parts()`
//! access, and the `unpack_codes_to_u16` helper. Decode, LIKE, and EQ
//! predicates already live in `vortex-onpair` as pure Rust and reuse the
//! same `(dict_bytes, dict_offsets, codes_packed, codes_boundaries, bits)`
//! layout produced here.

pub mod bit_unpack;
pub mod bit_writer;
pub mod column;
pub mod config;
pub mod dict;
pub mod lpm;
pub mod parser;
pub mod store;
pub mod trainer;
pub mod types;

pub use bit_unpack::{read_bits_lsb, unpack_codes_to_u16};
pub use column::{Column, Parts};
pub use config::{DEFAULT_DICT12_CONFIG, DynamicThreshold, Error, FixedThreshold,
                 OnPairTrainingConfig, ThresholdSpec, TrainingConfig};
pub use dict::Dictionary;
pub use lpm::LongestPrefixMatcher;
pub use parser::parse;
pub use store::Store;
pub use trainer::{TrainResult, train};
pub use types::{BitWidth, ByteSpan, MAX_TOKEN_SIZE, StreamSpan, Token, TokenRange,
                is_valid_bits, max_dict_size};
