// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use lending_iterator::LendingIterator;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::primitive::IS_CONST_LANE_WIDTH;
use vortex_array::aggregate_fn::fns::is_constant::primitive::compute_is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// BitPacked-specific is_constant kernel with SIMD support.
#[derive(Debug)]
pub(crate) struct BitPackedIsConstantKernel;

impl DynAggregateKernel for BitPackedIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<BitPacked>() else {
            return Ok(None);
        };

        let result = match_each_integer_ptype!(array.dtype().as_ptype(), |P| {
            bitpacked_is_constant::<P, { IS_CONST_LANE_WIDTH / size_of::<P>() }>(array, ctx)?
        });

        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

fn bitpacked_is_constant<T: BitPackedUnpack, const WIDTH: usize>(
    array: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mut bit_unpack_iterator = array.unpacked_chunks::<T>()?;
    let patches = array
        .patches()
        .map(|p| -> VortexResult<_> {
            let values = p.values().clone().execute::<PrimitiveArray>(ctx)?;
            let indices = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
            let offset = p.offset();
            Ok((indices, values, offset))
        })
        .transpose()?;

    let mut header_constant_value = None;
    let mut current_idx = 0;
    if let Some(header) = bit_unpack_iterator.initial() {
        if let Some((indices, patches, offset)) = &patches {
            apply_patches(
                header,
                current_idx..header.len(),
                indices,
                patches.as_slice::<T>(),
                *offset,
            )
        }

        if !compute_is_constant::<_, WIDTH>(header) {
            return Ok(false);
        }
        header_constant_value = Some(header[0]);
        current_idx = header.len();
    }

    let mut first_chunk_value = None;
    let mut chunks_iter = bit_unpack_iterator.full_chunks();
    while let Some(chunk) = chunks_iter.next() {
        if let Some((indices, patches, offset)) = &patches {
            let chunk_len = chunk.len();
            apply_patches(
                chunk,
                current_idx..current_idx + chunk_len,
                indices,
                patches.as_slice::<T>(),
                *offset,
            )
        }

        if !compute_is_constant::<_, WIDTH>(chunk) {
            return Ok(false);
        }

        if let Some(chunk_value) = first_chunk_value {
            if chunk_value != chunk[0] {
                return Ok(false);
            }
        } else {
            if let Some(header_value) = header_constant_value
                && header_value != chunk[0]
            {
                return Ok(false);
            }
            first_chunk_value = Some(chunk[0]);
        }

        current_idx += chunk.len();
    }

    if let Some(trailer) = bit_unpack_iterator.trailer() {
        if let Some((indices, patches, offset)) = &patches {
            let chunk_len = trailer.len();
            apply_patches(
                trailer,
                current_idx..current_idx + chunk_len,
                indices,
                patches.as_slice::<T>(),
                *offset,
            )
        }

        if !compute_is_constant::<_, WIDTH>(trailer) {
            return Ok(false);
        }

        if let Some(previous_const_value) = header_constant_value.or(first_chunk_value)
            && previous_const_value != trailer[0]
        {
            return Ok(false);
        }
    }

    Ok(true)
}

fn apply_patches<T: BitPackedUnpack>(
    values: &mut [T],
    values_range: Range<usize>,
    patch_indices: &PrimitiveArray,
    patch_values: &[T],
    indices_offset: usize,
) {
    match_each_unsigned_integer_ptype!(patch_indices.ptype(), |I| {
        apply_patches_idx_typed(
            values,
            values_range,
            patch_indices.as_slice::<I>(),
            patch_values,
            indices_offset,
        )
    });
}

fn apply_patches_idx_typed<T: BitPackedUnpack, I: IntegerPType>(
    values: &mut [T],
    values_range: Range<usize>,
    patch_indices: &[I],
    patch_values: &[T],
    indices_offset: usize,
) {
    for (i, &v) in patch_indices
        .iter()
        .map(|i| i.as_() - indices_offset)
        .zip_eq(patch_values)
        .skip_while(|(i, _)| i < &values_range.start)
        .take_while(|(i, _)| i < &values_range.end)
    {
        values[i - values_range.start] = v
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_constant::is_constant;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::BitPackedData;

    #[test]
    fn is_constant_with_patches() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = BitPackedData::encode(&buffer![4; 1025].into_array(), 2, &mut ctx)?;
        assert!(is_constant(&array.into_array(), &mut ctx)?);
        Ok(())
    }
}
