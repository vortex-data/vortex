// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;

/// Compresses the given patches by downscaling integers and checking for constant values.
pub fn compress_patches(patches: Patches, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
    // Downscale the patch indices.
    let indices = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .narrow(ctx)?
        .into_array();

    // Check if the values are constant.
    let values = patches.values();
    let values = if values
        .statistics()
        .compute_is_constant(ctx)
        .unwrap_or_default()
    {
        ConstantArray::new(values.execute_scalar(0, ctx)?, values.len()).into_array()
    } else {
        values.clone()
    };
    let chunk_offsets = patches
        .chunk_offsets()
        .as_ref()
        .map(|offsets| {
            let offsets_primitive = offsets
                .clone()
                .execute::<PrimitiveArray>(ctx)?
                .narrow(ctx)?
                .into_array();
            Ok::<ArrayRef, VortexError>(offsets_primitive)
        })
        .transpose()?;

    Patches::new(
        patches.array_len(),
        patches.offset(),
        indices,
        values,
        chunk_offsets,
        ctx,
    )
}

/// Build the selected sparse representation from the compressed patch children.
pub(super) fn sparse_from_children(
    indices: ArrayRef,
    values: ArrayRef,
    len: usize,
    fill: Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if use_experimental_patches() {
        let patches = Patches::new(len, 0, indices, values, None, ctx)?;
        let base = ConstantArray::new(fill, len).into_array();
        Ok(Patched::from_array_and_patches(base, &patches, ctx)?.into_array())
    } else {
        Ok(Sparse::try_new(indices, values, len, fill, ctx)?.into_array())
    }
}
