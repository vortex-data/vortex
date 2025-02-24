#![feature(portable_simd)]

//! Implementation of Dictionary encoding.
//!
//! Expose a [DictArray] which is zero-copy equivalent to Arrow's
//! [DictionaryArray](https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html).
pub use array::*;

mod array;
pub mod builders;
mod compress;
mod compute;
mod serde;
mod stats;
#[cfg(feature = "test-harness")]
pub mod test;
mod variants;
