// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{
    IntegerPType, NativePType, UnsignedPType, match_each_integer_ptype, match_each_native_ptype,
};

use crate::ToCanonical;
use crate::arrays::PrimitiveArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl PrimitiveArray {
    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: &Patches) -> Self {
        let patch_indices = patches.indices().to_primitive();
        let patch_values = patches.values().to_primitive();

        let patched_validity = self.validity().clone().patch(
            self.len(),
            patches.offset(),
            patch_indices.as_ref(),
            patch_values.validity(),
        );
        match_each_integer_ptype!(patch_indices.ptype(), |I| {
            match_each_native_ptype!(self.ptype(), |T| {
                self.patch_typed::<T, I>(
                    patch_indices,
                    patches.offset(),
                    patch_values,
                    patched_validity,
                )
            })
        })
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

/// Patches a chunk of decoded values.
///
/// # Arguments
///
/// * `decoded_values` - Mutable slice of decoded values to be patched
/// * `patches_indices` - Indices indicating which positions to patch
/// * `patches_values` - Values to apply at the patched indices
/// * `patches_offset` - Offset to subtract from patch indices
/// * `chunk_offsets_slice` - Slice containing offsets for each chunk
/// * `chunk_idx` - Index of the chunk to patch
#[inline]
pub fn patch_chunk<T, I, C>(
    decoded_values: &mut [T],
    patches_indices: &[I],
    patches_values: &[T],
    patches_offset: usize,
    chunk_offsets_slice: &[C],
    chunk_idx: usize,
) where
    T: NativePType,
    I: UnsignedPType,
    C: UnsignedPType,
{
    let patches_start_idx = chunk_offsets_slice[chunk_idx].as_();
    let patches_end_idx = if chunk_idx + 1 < chunk_offsets_slice.len() {
        chunk_offsets_slice[chunk_idx + 1].as_()
    } else {
        patches_indices.len()
    };

    let chunk_start = chunk_idx * 1024;
    for patches_idx in patches_start_idx..patches_end_idx {
        let patched_value = patches_values[patches_idx];
        let absolute_index: usize = patches_indices[patches_idx].as_() - patches_offset;
        let chunk_relative_index = absolute_index - chunk_start;
        decoded_values[chunk_relative_index] = patched_value;
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::ToCanonical;
    use crate::validity::Validity;

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::new(buffer![2u32; 10], Validity::AllValid);
        let sliced = input.slice(2..8);
        assert_eq!(sliced.to_primitive().as_slice::<u32>(), &[2u32; 6]);
    }
}
