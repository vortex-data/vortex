// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An in-memory encoding that repeats a child array end to end.
//!
//! Row `i` of a [`RepeatedArray`] with child `c` is row `i % c.len()` of `c`. Its length is any
//! number of rows, so the last repetition may be cut short: repeating `[1, 2, 3]` to 7 rows gives
//! `[1, 2, 3, 1, 2, 3, 1]`. Any slice of a `RepeatedArray` is therefore another `RepeatedArray`
//! over the child rotated to the slice's start, or a slice of the child.
//!
//! Execution decodes the child once and appends it per copy, so the child is never decoded per
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
pub use array::RepeatedSlots;
pub use array::RepeatedSlotsView;
pub use vtable::Repeated;
pub use vtable::RepeatedArray;

#[cfg(test)]
mod tests;
