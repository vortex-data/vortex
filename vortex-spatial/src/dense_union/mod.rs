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
//! directly exportable Arrow dense-union layout; [`compact_for_arrow`] gathers the children and
//! rebases the offsets when one is needed.

mod array;
mod canonical;
mod compact;
mod compute;
mod rules;
mod vtable;

pub use array::*;
pub(crate) use compact::compact_for_arrow;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::UnionVariants;
use vortex_array::session::ArraySessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

pub(crate) fn initialize(session: &VortexSession) {
    session.arrays().register(DenseUnion);
}

/// Destructure a union dtype into its variants and nullability.
pub(crate) fn union_variants(dtype: &DType) -> VortexResult<(&UnionVariants, Nullability)> {
    let variants = dtype
        .as_union_variants_opt()
        .ok_or_else(|| vortex_err!("DenseUnion requires a union dtype, got {dtype}"))?;
    Ok((variants, dtype.nullability()))
}

/// Map every data-level type tag to its child index.
///
/// [`UnionVariants::tag_to_child_index`] linear-scans the variants, which is the right trade-off
/// for a one-off lookup but not for one resolved per row.
pub(crate) fn tag_lookup(variants: &UnionVariants) -> [Option<usize>; 256] {
    let mut lookup = [None; 256];
    for (child_index, tag) in variants.type_ids().iter().copied().enumerate() {
        lookup[usize::from(tag)] = Some(child_index);
    }
    lookup
}

#[cfg(test)]
mod tests;
