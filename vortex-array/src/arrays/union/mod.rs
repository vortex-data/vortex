// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical sparse union arrays.
//!
//! A [`UnionArray`] stores one `u8` type ID per row followed by one row-aligned child for each
//! variant. The type ID selects which child's value is active for a row; values in all other
//! children at that row are placeholders. Outer union nulls are stored as nulls in the type IDs
//! child, independently of nulls in the selected variant child.
//!
//! Type ID values are not validated during construction. Accessing a non-null row whose type ID
//! is not declared by the union variants will panic.
//!
//! Mask rewrites only the type IDs. Slice and take rewrite every child to keep them row-aligned,
//! so each costs `O(variants)` child operations.

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

mod array;
pub use array::UnionArrayExt;
pub use array::UnionArraySlotsExt;
pub use array::UnionDataParts;
pub use array::UnionSlots;
pub use array::UnionSlotsView;
pub use vtable::UnionArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::Union;

pub(crate) fn union_type_ids_dtype(nullability: Nullability) -> DType {
    DType::Primitive(PType::U8, nullability)
}

#[cfg(test)]
mod tests;
