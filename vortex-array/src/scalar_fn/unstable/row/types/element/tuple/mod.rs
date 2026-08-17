// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Combines [`InputElement`](super::InputElement)s into typed row argument lists.
//!
//! [`ElementTuple`] owns decoding, constant classification, and row access for supported arities.
//! [`IndexedElementTuple`] adds the validated indexed source used by vectorizable dense loops.

mod element_tuple;
pub use element_tuple::ElementTuple;

mod indexed;
pub use indexed::IndexedElementTuple;

#[cfg(test)]
mod tests;
