// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementation of Dictionary encoding.
//!
//! Expose a [DictArray] which is zero-copy equivalent to Arrow's
//! [DictionaryArray](https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html).

mod array;
pub use array::*;

mod compute;
mod execute;

pub use execute::take_canonical;

pub mod vtable;
pub use vtable::*;

#[cfg(test)]
mod tests;
