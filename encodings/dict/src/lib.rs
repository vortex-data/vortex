// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementation of Dictionary encoding.
//!
//! Expose a [DictArray] which is zero-copy equivalent to Arrow's
//! [DictionaryArray](https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html).
pub use array::*;

mod array;
#[cfg(feature = "arrow")]
mod arrow;
pub mod builders;
mod canonical;
mod compute;
mod display;
mod ops;
mod serde;
#[cfg(feature = "test-harness")]
pub mod test;
