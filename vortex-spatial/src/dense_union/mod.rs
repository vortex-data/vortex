// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A dense physical encoding for spatial union arrays.
//!
//! [`DenseUnionArray`] stores one type ID and one child offset per logical row. Variant children
//! are compact: unlike the canonical sparse union, they do not contain placeholders for rows that
//! select a different variant. The array still has the logical
//! [`DType::Union`](vortex_array::dtype::DType::Union) dtype.
//!
//! Vortex does not require offsets for each child to increase. Selector-only operations can retain
//! the original compact children and reorder their offsets, so this encoding is not necessarily a
//! directly exportable Arrow dense-union layout without compaction and offset rebasing.

mod array;
mod canonical;
mod compute;
mod rules;
mod vtable;

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

pub(crate) fn initialize(session: &VortexSession) {
    session.arrays().register(DenseUnion);
}

#[cfg(test)]
mod tests;
