// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::patches::Patches;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::SchemeId;

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

/// Narrows and compresses the patch chunk offsets of a sparse array as child `child_index` of
/// `scheme`.
///
/// Sparse arrays are often tiny, so the offsets, one per 1024 rows, can dominate their size
/// unless they are compressed like the indices.
pub(crate) fn compress_sparse_chunk_offsets(
    compressor: &CascadingCompressor,
    chunk_offsets: Option<&ArrayRef>,
    compress_ctx: &CompressorContext,
    scheme: SchemeId,
    child_index: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    chunk_offsets
        .map(|offsets| {
            let narrowed = offsets
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?
                .narrow(exec_ctx)?
                .into_array();
            compressor.compress_child(&narrowed, compress_ctx, scheme, child_index, exec_ctx)
        })
        .transpose()
}
