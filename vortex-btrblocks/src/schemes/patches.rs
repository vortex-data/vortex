// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::PatchedArrayExt;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::patched::PatchedSlots;
use vortex_compressor::scheme::SchemeId;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use crate::CascadingCompressor;
use crate::CompressorContext;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::patches::Patches;
use vortex_error::VortexError;
use vortex_error::VortexResult;

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

/// Re-encodes a [`Sparse`] array as a [`Patched`] array over its constant fill value, and
/// compresses the patches.
///
/// The patches are transposed into their chunk/lane layout before compression, so that the patch
/// values are reordered once here rather than lazily on every read. The lane offsets and patch
/// indices are compressed as the child `indices_child` of `scheme`, and the values as the child
/// `values_child` when it is given.
pub(crate) fn sparse_to_patched(
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    scheme: SchemeId,
    sparse: ArrayView<'_, Sparse>,
    indices_child: usize,
    values_child: Option<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let patches = sparse.patches();
    let values = patches.values().clone().execute::<Canonical>(ctx)?.into_array();
    let patches = Patches::new(
        patches.array_len(),
        patches.offset(),
        patches.indices().clone(),
        values,
        None,
        ctx,
    )?;
    let fill = ConstantArray::new(sparse.fill_scalar().clone(), sparse.len()).into_array();
    let patched = Patched::from_array_and_patches(fill, &patches, ctx)?;

    let compress = |child: &ArrayRef, idx: usize, ctx: &mut ExecutionCtx| {
        compressor.compress_child(child, compress_ctx, scheme, idx, ctx)
    };
    let lane_offsets = compress(patched.lane_offsets(), indices_child, ctx)?;
    let patch_indices = compress(patched.patch_indices(), indices_child, ctx)?;
    let patch_values = match values_child {
        Some(idx) => compress(patched.patch_values(), idx, ctx)?,
        None => patched.patch_values().clone(),
    };

    let slots = PatchedSlots {
        inner: patched.inner().clone(),
        lane_offsets,
        patch_indices,
        patch_values,
    }
    .into_slots();
    // SAFETY: compression preserves the logical values, lengths and dtypes of the children.
    Ok(unsafe {
        Patched::new_unchecked(
            patched.dtype().clone(),
            patched.len(),
            slots,
            patched.n_lanes(),
            patched.offset(),
        )
    }
    .into_array())
}
