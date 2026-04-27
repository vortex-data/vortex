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

        let patch_validity = patch_values.validity()?;
        let patched_validity = self.validity()?.patch(
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
    // Clamp: chunk_offsets are sliced at chunk granularity but patches at element
    // granularity, so the next chunk offset may exceed the actual patches length.
    let patches_end_idx = if chunk_idx + 1 < chunk_offsets_slice.len() {
        (chunk_offsets_slice[chunk_idx + 1].as_() - base_offset)
            .saturating_sub(offset_within_chunk)
            .min(patches_indices.len())
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
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    /// Regression: patch_chunk must not OOB when chunk_offsets (chunk granularity)
    /// reference more patches than patches_indices (element granularity) contains.
    #[test]
    fn patch_chunk_no_oob_on_mid_chunk_slice() {
        let mut decoded_values = vec![0.0f64; PATCH_CHUNK_SIZE];
        // 10 patches, but chunk_offsets claim 15 exist past offset adjustment.
        let patches_indices: Vec<u64> = (0..10)
            .map(|i| (PATCH_CHUNK_SIZE as u64) + i * 10)
            .collect();
        let patches_values: Vec<f64> = (0..10).map(|i| (i + 1) as f64 * 100.0).collect();
        // chunk_offsets [5, 12, 20]: for chunk_idx=1 with offset_within_chunk=3,
        // unclamped end = (20-5)-3 = 12, which exceeds patches len of 10.
        let chunk_offsets: Vec<u32> = vec![5, 12, 20];

        patch_chunk(
            &mut decoded_values,
            &patches_indices,
            &patches_values,
            0,
            &chunk_offsets,
            1,
            3,
        );

        // Spot-check: patch index 4 (first in range) should be applied.
        assert_ne!(
            decoded_values[usize::try_from(patches_indices[4]).unwrap() - PATCH_CHUNK_SIZE],
            0.0
        );
    }

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::new(buffer![2u32; 10], Validity::AllValid);
        let sliced = input.slice(2..8).unwrap();
        #[expect(deprecated)]
        let sliced_primitive = sliced.to_primitive();
        assert_arrays_eq!(
            sliced_primitive,
            PrimitiveArray::new(buffer![2u32; 6], Validity::AllValid)
        );
    }
}
