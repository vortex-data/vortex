// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::indices::lower_bound;
use crate::arrays::slice::SliceKernel;
use crate::match_each_unsigned_integer_ptype;
use crate::optimizer::ArrayOptimizer;
use crate::patches::PATCH_CHUNK_SIZE;

impl SliceKernel for Patched {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let inner = array.inner().slice(range.clone())?.optimize_ctx(ctx)?;
        let first = lower_bound(array, range.start, ctx)?;
        let last = lower_bound(array, range.end, ctx)?;
        if first == last {
            return Ok(Some(inner));
        }
        let offset = array.offset + range.start;
        let end = array.offset + range.end;
        let first_chunk = array.offset / PATCH_CHUNK_SIZE;
        let chunks =
            offset / PATCH_CHUNK_SIZE - first_chunk..end.div_ceil(PATCH_CHUNK_SIZE) - first_chunk;
        let offsets = array.patch_chunk_offsets();
        let offsets = if first == 0 {
            offsets.slice(chunks)?.optimize_ctx(ctx)?
        } else {
            let primitive = offsets.clone().execute::<PrimitiveArray>(ctx)?;
            match_each_unsigned_integer_ptype!(primitive.ptype(), |I| {
                let bias = I::try_from(first).unwrap_or(I::MAX);
                primitive.as_slice::<I>()[chunks]
                    .iter()
                    .map(|&v| v.saturating_sub(bias))
                    .collect::<Buffer<I>>()
                    .into_array()
            })
        };
        let indices = array
            .patch_indices()
            .slice(first..last)?
            .optimize_ctx(ctx)?;
        let values = array.patch_values().slice(first..last)?.optimize_ctx(ctx)?;
        Ok(Some(
            Patched::from_relative(inner, indices, values, offsets, offset, array.strategy())?
                .into_array(),
        ))
    }
}
