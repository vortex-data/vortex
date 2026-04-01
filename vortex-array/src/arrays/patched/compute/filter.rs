// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FilterArray;
use crate::arrays::Patched;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Patched {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        // Find the contiguous chunk range that the mask covers. We use this to slice the inner
        // components, then wrap the rest up with another FilterArray.
        //
        // This is helpful when we have a very selective filter that is clustered to a small
        // range.
        let (chunk_start, chunk_stop) = match mask.slices() {
            AllOr::All | AllOr::None => {
                // This is handled as the precondition to this method, see the FilterReduce
                // documentation.
                unreachable!("mask must be a MaskValues here")
            }
            AllOr::Some(slices) => {
                let (first, _) = slices[0];
                let (_, last) = slices[slices.len() - 1];

                // Convert mask indices to absolute positions by adding offset
                (
                    (array.offset + first) / 1024,
                    (array.offset + last).div_ceil(1024),
                )
            }
        };

        let n_chunks = (array.offset + array.len).div_ceil(1024);

        // If all chunks already covered, there is nothing to do.
        if chunk_start == 0 && chunk_stop == n_chunks {
            return Ok(None);
        }

        let sliced = array.slice_chunks(chunk_start..chunk_stop)?;

        // Slice the mask according to if the chunk is sliced.
        // Convert chunk bounds back to mask indices by subtracting offset.
        let mask_start = (chunk_start * 1024).saturating_sub(array.offset);
        let mask_end = (chunk_stop * 1024)
            .saturating_sub(array.offset)
            .min(array.len());
        let remainder = mask.slice(mask_start..mask_end);

        Ok(Some(
            FilterArray::new(sliced.into_array(), remainder).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;

    #[test]
    fn test_filter_noop() -> VortexResult<()> {
        // Filter that doesn't prune any chunks (all data fits in one chunk).
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let array = buffer![u16::MIN; 5].into_array();
        let patched_indices = buffer![3u8, 4].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(5, 0, patched_indices, patched_values, None)?;

        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        let mask = Mask::from_iter([true, false, false, false, true]);
        let filtered = array
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        // Values at indices 0 and 4: MIN and MAX.
        let expected = PrimitiveArray::from_iter([u16::MIN, u16::MAX]);

        assert_arrays_eq!(expected, filtered);

        Ok(())
    }

    #[test]
    fn test_filter_with_offset() -> VortexResult<()> {
        // Test filtering where offset > 0.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![5u16, 1030].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(4096, 0, patched_indices, patched_values, None)?;

        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        let sliced = array.slice(5..4096)?.optimize()?;

        // Filter that touches only the first 2 chunks.
        let mask = Mask::from_indices(4091, vec![0, 1, 2, 1025]);
        let filtered = sliced
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MIN, u16::MIN, u16::MAX]);
        assert_arrays_eq!(expected, filtered);

        Ok(())
    }

    #[test]
    fn test_filter_basic() -> VortexResult<()> {
        // Basic test: filter with mask that crosses boundaries.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![1024u16, 1025].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(4096, 0, patched_indices, patched_values, None)?;

        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Filter that only touches the middle 2 chunks.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);
        let filtered = array
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);

        assert_arrays_eq!(expected, filtered);

        Ok(())
    }

    #[test]
    fn test_filter_complex() -> VortexResult<()> {
        // Filter with mask that crosses boundaries, with patches offset.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![1024u16, 1025].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(4096, 1, patched_indices, patched_values, None)?;

        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Filter that only touches the middle 2 chunks.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);
        let filtered = array
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MIN, u16::MIN]);

        assert_arrays_eq!(expected, filtered);

        Ok(())
    }

    #[test]
    fn test_filter_sliced() -> VortexResult<()> {
        // Test filter on a sliced PatchedArray to exercise codepath where offset > 0.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        // Create a larger array (6 chunks) so we can slice and still have room
        // for the filter to prune chunks.
        let array = buffer![u16::MIN; 6144].into_array();
        // Patches at indices 2048 and 2049 (start of chunk 2).
        let patched_indices = buffer![2048u16, 2049].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(6144, 0, patched_indices, patched_values, None)?;

        let patched = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?;

        // Slice at chunk boundary to create offset > 0. After slicing [1024..5120],
        // we have 4096 elements and patches are at relative indices 1024 and 1025.
        let sliced = patched.into_array().slice(1024..5120)?;
        assert_eq!(sliced.len(), 4096);

        // Filter that only touches the middle 2 chunks (chunks 1 and 2).
        // Indices 1024 and 1025 fall in chunk 1, and 3000 falls in chunk 2.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);

        let filtered = sliced
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);

        assert_arrays_eq!(expected, filtered);

        Ok(())
    }

    #[test]
    fn test_filter_with_offset_nonuniform() -> VortexResult<()> {
        // Test filtering with offset > 0 using non-uniform base values.
        // This catches slice_chunks bugs where inner coordinates are miscalculated.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        // Use non-uniform values so that incorrect slicing is detectable.
        let base_values: Vec<u16> = (0u16..4096).collect();
        let array = PrimitiveArray::from_iter(base_values).into_array();

        // Patch at index 5 (value becomes 9999) and index 1030 (value becomes 8888).
        let patched_indices = buffer![5u16, 1030].into_array();
        let patched_values = buffer![9999u16, 8888].into_array();

        let patches = Patches::new(4096, 0, patched_indices, patched_values, None)?;
        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Slice to create offset > 0.
        // After slice(5..4096), logical position 0 = original position 5 (patched to 9999).
        let sliced = array.slice(5..4096)?.optimize()?;
        assert_eq!(sliced.len(), 4091);

        // Filter that touches the first 2 chunks.
        // Logical indices: 0 (was 5, patched), 1 (was 6, value 6), 1025 (was 1030, patched).
        let mask = Mask::from_indices(4091, vec![0, 1, 1025]);
        let filtered = sliced
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        // Expected: 9999 (patched at logical 0), 6 (original at logical 1), 8888 (patched at logical 1025).
        let expected = PrimitiveArray::from_iter([9999u16, 6, 8888]);
        assert_arrays_eq!(expected, filtered);

        Ok(())
    }

    #[test]
    fn test_filter_with_offset_last_chunk() -> VortexResult<()> {
        // Test filtering with offset > 0 where the mask touches the last chunk.
        // This ensures we don't accidentally slice past the end of the array or mask.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        // Create a 6-chunk array (6144 elements).
        let array = buffer![u16::MIN; 6144].into_array();
        // Patches near the end of the array at indices 5000 and 6000.
        let patched_indices = buffer![5000u16, 6000].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(6144, 0, patched_indices, patched_values, None)?;

        let patched = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?;

        // Slice at chunk boundary to create offset > 0.
        // Slice [1024..6144] gives us 5120 elements (5 chunks).
        // Original patches at 5000 and 6000 become relative indices 3976 and 4976.
        let sliced = patched.into_array().slice(1024..6144)?.optimize()?;
        assert_eq!(sliced.len(), 5120);

        // Filter that touches only the last 2 chunks (chunks 3 and 4).
        // Chunk 3: indices 3072-4095, Chunk 4: indices 4096-5119.
        // Patch at 3976 is in chunk 3, patch at 4976 is in chunk 4.
        let mask = Mask::from_indices(5120, vec![3976, 4976, 5119]);

        let filtered = sliced
            .filter(mask)?
            .optimize()?
            .execute::<PrimitiveArray>(&mut ctx)?;

        // Expected: patch at 3976 (was 5000), patch at 4976 (was 6000), and MIN at 5119.
        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);

        assert_arrays_eq!(expected, filtered);

        Ok(())
    }
}
