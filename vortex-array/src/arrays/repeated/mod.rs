// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An in-memory encoding that repeats a child array end to end.
//!
//! A [`RepeatedArray`] with child `c` and count `n` has `c.len() * n` rows, and row `i` is row
//! `i % c.len()` of `c`. For example, repeating `[1, 2, 3]` twice gives `[1, 2, 3, 1, 2, 3]`.
//!
//! Execution decodes the child once and appends it `n` times, so the child is never decoded per
//! copy.
//!
//! `Repeated` has no wire form, so the default [`ArraySession`] does not register it, and an array
//! that holds a `Repeated` cannot be serialized.
//!
//! [`ArraySession`]: crate::session::ArraySession

mod array;
mod rules;
mod vtable;

pub use array::RepeatedArraySlotsExt;
pub use array::RepeatedData;
pub use array::RepeatedSlots;
pub use array::RepeatedSlotsView;
pub use vtable::Repeated;
pub use vtable::RepeatedArray;

#[cfg(test)]
mod tests;
