// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementation of Dictionary encoding.
//!
//! Expose a [DictArray] which is zero-copy equivalent to Arrow's
//! [DictionaryArray](https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html).

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryDictArray;

mod array;
pub use array::*;

pub(crate) mod compute;
mod execute;

mod take;
pub use take::*;

pub mod vtable;
pub use vtable::*;

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    vtable::initialize(session);
}

#[cfg(test)]
mod tests;
