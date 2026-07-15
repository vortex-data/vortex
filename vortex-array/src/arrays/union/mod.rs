// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical sparse union arrays.
//!
//! A [`UnionArray`] stores one non-nullable `i8` type ID per row followed by one row-aligned child
//! for each variant. The type ID selects which child's value is active for a row; values in all
//! other children at that row are placeholders.
//!
//! Union nullability semantics are still being designed, so this encoding currently requires every
//! variant child to be non-nullable.

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

mod array;
pub use array::UnionArrayExt;
pub use array::UnionDataParts;
pub use vtable::UnionArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::Union;

pub(crate) const TYPE_IDS_DTYPE: DType = DType::Primitive(PType::I8, Nullability::NonNullable);

#[cfg(test)]
mod tests;
