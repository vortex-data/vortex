// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate contains an implementation of the floating point compression algorithm from the
//! paper ["ALP: Adaptive Lossless floating-Point Compression"][paper] by Afroozeh et al.
//!
//! The compressor has two variants, classic ALP which is well-suited for data that does not use
//! the full precision, and "real doubles", values that do.
//!
//! Classic ALP will return small integers, and it is meant to be cascaded with other integer
//! compression techniques such as bit-packing and frame-of-reference encoding. Combined this allows
//! for significant compression on the order of what you can get for integer values.
//!
//! ALP-RD is generally terminal, and in the ideal case it can represent an f64 is just 49 bits,
//! though generally it is closer to 54 bits per value or ~12.5% compression.
//!
//! [paper]: https://ir.cwi.nl/pub/33334/33334.pdf

use std::iter;

pub use alp::*;
pub use alp_rd::*;
use vortex_array::execution::ExecutionCtx;
use vortex_array::patches::Patches;
use vortex_array::{Array, ArrayOperator};
use vortex_dtype::{
    IntegerPType, NativePType, PTypeDowncastExt, match_each_integer_ptype, match_each_native_ptype,
};
use vortex_error::{VortexResult, vortex_ensure};
use vortex_mask::Mask;
use vortex_vector::primitive::{PVector, PVectorMut, PrimitiveVectorMut};

mod alp;
mod alp_rd;

/// Apply the patches in-place to the given type node.
pub(crate) fn apply_patches_in_place(
    values: &mut PrimitiveVectorMut,
    patches: &Patches,
    ctx: &mut dyn ExecutionCtx,
) -> VortexResult<()> {
    let n_patches = patches.indices().len();
    let mask = Mask::new_true(n_patches);

    let patch_indices = patches
        .indices()
        .execute_batch(&mask, ctx)?
        .into_primitive();

    let patch_values = patches.values().execute_batch(&mask, ctx)?.into_primitive();

    vortex_ensure!(
        values.ptype() == patch_values.ptype(),
        "values ptype {} must match patch_values ptype {}",
        values.ptype(),
        patch_values.ptype()
    );

    match_each_native_ptype!(values.ptype(), |Value| {
        match_each_integer_ptype!(patch_indices.ptype(), |Index| {
            let values_pvec = values.downcast::<Value>();
            let patch_indices_pvec = patch_indices.downcast::<Index>();
            let patch_values_pvec = patch_values.downcast::<Value>();

            apply_in_place(values_pvec, &patch_indices_pvec, &patch_values_pvec);
        })
    });

    Ok(())
}

/// Applying patches in-place doesn't affect the output vector node instead.
pub(crate) fn apply_in_place<Value: NativePType, Index: IntegerPType>(
    values: &mut PVectorMut<Value>,
    patch_indices: &PVector<Index>,
    patch_values: &PVector<Value>,
) {
    // Overwrite values using the patch indices.
    let values_mut = values.as_mut();
    for (&index, &value) in iter::zip(patch_indices.as_ref(), patch_values.as_ref()) {
        values_mut[index.as_()] = value;
    }
}
