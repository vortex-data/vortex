// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::validity::Validity;

impl PrimitiveArray {
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let patch_indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;

        let patch_validity = patch_values.validity();
        let patched_validity = self.validity().patch(
            self.len(),
            patches.offset(),
            &patch_indices.clone().into_array(),
            &patch_validity,
            ctx,
        )?;
        Ok(match_each_integer_ptype!(patch_indices.ptype(), |I| {
            match_each_native_ptype!(self.ptype(), |T| {
                self.patch_typed::<T, I>(
                    patch_indices,
                    patches.offset(),
                    patch_values,
                    patched_validity,
                )
            })
        }))
    }

    fn patch_typed<T, I>(
        self,
        patch_indices: PrimitiveArray,
        patch_indices_offset: usize,
        patch_values: PrimitiveArray,
        patched_validity: Validity,
    ) -> Self
    where
        T: NativePType,
        I: IntegerPType,
    {
        let mut own_values = self.into_buffer_mut::<T>();

        let patch_indices = patch_indices.as_slice::<I>();
        let patch_values = patch_values.as_slice::<T>();
        for (idx, value) in itertools::zip_eq(patch_indices, patch_values) {
            own_values[idx.as_() - patch_indices_offset] = *value;
        }
        Self::new(own_values, patched_validity)
    }
}

/// Computes the index range for a chunk, accounting for slice offset.
///
/// # Arguments
///
/// * `chunk_idx` - Index of the chunk
/// * `offset` - Offset from slice
/// * `array_len` - Length of the sliced array
#[inline]
pub fn chunk_range(chunk_idx: usize, offset: usize, array_len: usize) -> Range<usize> {
    let offset_in_chunk = offset % PATCH_CHUNK_SIZE;
    let local_start = (chunk_idx * PATCH_CHUNK_SIZE).saturating_sub(offset_in_chunk);
    let local_end = ((chunk_idx + 1) * PATCH_CHUNK_SIZE)
        .saturating_sub(offset_in_chunk)
        .min(array_len);
    local_start..local_end
}

/// Patches a chunk of decoded values.
///
/// # Arguments
///
/// * `decoded_values` - Mutable slice of decoded values to be patched
/// * `patches_indices` - Indices indicating which positions to patch
/// * `patches_values` - Values to apply at the patched indices
/// * `patches_offset` - Absolute position where the slice starts
/// * `chunk_offsets_slice` - Slice containing offsets for each chunk
/// * `chunk_idx` - Index of the chunk to patch
/// * `offset_within_chunk` - Number of patches to skip at the start of the first chunk
pub fn patch_chunk<T, I, C>(
    decoded_values: &mut [T],
    patches_indices: &[I],
    patches_values: &[T],
    patches_offset: usize,
    chunk_offsets_slice: &[C],
    chunk_idx: usize,
    offset_within_chunk: usize,
) where
    T: NativePType,
    I: UnsignedPType,
    C: UnsignedPType,
{
    // Compute base_offset from the first chunk offset.
    let base_offset: usize = chunk_offsets_slice[0].as_();

    // Use the same logic as patches slice implementation for calculating patch ranges.
    let patches_start_idx =
        (chunk_offsets_slice[chunk_idx].as_() - base_offset).saturating_sub(offset_within_chunk);
    let patches_end_idx = if chunk_idx + 1 < chunk_offsets_slice.len() {
        chunk_offsets_slice[chunk_idx + 1].as_() - base_offset - offset_within_chunk
    } else {
        patches_indices.len()
    };

    let chunk_start = chunk_range(chunk_idx, patches_offset, /* ignore */ usize::MAX).start;

    for patches_idx in patches_start_idx..patches_end_idx {
        let chunk_relative_index =
            (patches_indices[patches_idx].as_() - patches_offset) - chunk_start;
        decoded_values[chunk_relative_index] = patches_values[patches_idx];
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::ToCanonical;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::new(buffer![2u32; 10], Validity::AllValid);
        let sliced = input.slice(2..8).unwrap();
        assert_arrays_eq!(
            sliced.to_primitive(),
            PrimitiveArray::new(buffer![2u32; 6], Validity::AllValid)
        );
    }
}
