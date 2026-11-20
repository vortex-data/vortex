// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::filter::FilterKernel;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::validity::Validity;

impl FilterKernel for Patched {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let inner = array.inner().filter(mask.clone())?;

        let Some(mask_values) = mask.values() else {
            return Ok(None);
        };
        let bits = mask_values.bit_buffer();

        // The rank of each kept position is the number of kept positions before it.
        let mut positions = Vec::new();
        let layout = array.execute_layout(ctx)?;
        let layout = layout.layout()?;
        layout.for_each(|pos, patch| {
            if bits.value(pos) {
                positions.push((pos, patch));
            }
        });
        if positions.is_empty() {
            return Ok(Some(inner));
        }
        positions.sort_unstable();

        let mut new_positions = BufferMut::<u32>::with_capacity(positions.len());
        let mut sources = BufferMut::<u32>::with_capacity(positions.len());
        let mut rank = 0usize;
        let mut ranked_up_to = 0usize;
        for (pos, patch) in positions {
            rank += bits.slice(ranked_up_to..pos).true_count();
            ranked_up_to = pos;
            #[expect(clippy::cast_possible_truncation, reason = "patch counts fit in u32")]
            {
                new_positions.push(rank as u32);
                sources.push(patch as u32);
            }
        }

        let values = array
            .patch_values()
            .take(PrimitiveArray::new(sources.freeze(), Validity::NonNullable).into_array())?;
        let new_positions =
            PrimitiveArray::new(new_positions.freeze(), Validity::NonNullable).into_array();
        Ok(Some(
            Patched::try_new(inner, new_positions, values, ctx)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_filter_noop() -> VortexResult<()> {
        // Filter that doesn't prune any chunks (all data fits in one chunk).
        let mut ctx = SESSION.create_execution_ctx();

        let array = buffer![u16::MIN; 5].into_array();
        let patched_indices = buffer![3u8, 4].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(5, 0, patched_indices, patched_values, None, &mut ctx)?;

        let array = Patched::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        let mask = Mask::from_iter([true, false, false, false, true]);
        let filtered = array.filter(mask)?.optimize_ctx(&SESSION)?;

        // Values at indices 0 and 4: MIN and MAX.
        let expected = PrimitiveArray::from_iter([u16::MIN, u16::MAX]);

        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_filter_with_offset() -> VortexResult<()> {
        // Test filtering where offset > 0.
        let mut ctx = SESSION.create_execution_ctx();

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![5u16, 1030].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(4096, 0, patched_indices, patched_values, None, &mut ctx)?;

        let array = Patched::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        let sliced = array.slice(5..4096)?.optimize_ctx(&SESSION)?;

        // Filter that touches only the first 2 chunks.
        let mask = Mask::from_indices(4091, vec![0, 1, 2, 1025]);
        let filtered = sliced.filter(mask)?.optimize_ctx(&SESSION)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MIN, u16::MIN, u16::MAX]);
        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_filter_basic() -> VortexResult<()> {
        // Basic test: filter with mask that crosses boundaries.
        let mut ctx = SESSION.create_execution_ctx();

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![1024u16, 1025].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(4096, 0, patched_indices, patched_values, None, &mut ctx)?;

        let array = Patched::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Filter that only touches the middle 2 chunks.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);
        let filtered = array.filter(mask)?.optimize_ctx(&SESSION)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);

        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_filter_complex() -> VortexResult<()> {
        // Filter with mask that crosses boundaries, with patches offset.
        let mut ctx = SESSION.create_execution_ctx();

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![1024u16, 1025].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(4096, 1, patched_indices, patched_values, None, &mut ctx)?;

        let array = Patched::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Filter that only touches the middle 2 chunks.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);
        let filtered = array.filter(mask)?.optimize_ctx(&SESSION)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MIN, u16::MIN]);

        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_filter_sliced() -> VortexResult<()> {
        // Test filter on a sliced PatchedArray to exercise codepath where offset > 0.
        let mut ctx = SESSION.create_execution_ctx();

        // Create a larger array (6 chunks) so we can slice and still have room
        // for the filter to prune chunks.
        let array = buffer![u16::MIN; 6144].into_array();
        // Patches at indices 2048 and 2049 (start of chunk 2).
        let patched_indices = buffer![2048u16, 2049].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(6144, 0, patched_indices, patched_values, None, &mut ctx)?;

        let patched = Patched::from_array_and_patches(array, &patches, &mut ctx)?;

        // Slice at chunk boundary to create offset > 0. After slicing [1024..5120],
        // we have 4096 elements and patches are at relative indices 1024 and 1025.
        let sliced = patched.into_array().slice(1024..5120)?;
        assert_eq!(sliced.len(), 4096);

        // Filter that only touches the middle 2 chunks (chunks 1 and 2).
        // Indices 1024 and 1025 fall in chunk 1, and 3000 falls in chunk 2.
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);

        let filtered = sliced.filter(mask)?.optimize_ctx(&SESSION)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);

        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_filter_with_offset_nonuniform() -> VortexResult<()> {
        // Test filtering with offset > 0 using non-uniform base values.
        // This catches slice_chunks bugs where inner coordinates are miscalculated.
        let mut ctx = SESSION.create_execution_ctx();

        // Use non-uniform values so that incorrect slicing is detectable.
        let base_values: Vec<u16> = (0u16..4096).collect();
        let array = PrimitiveArray::from_iter(base_values).into_array();

        // Patch at index 5 (value becomes 9999) and index 1030 (value becomes 8888).
        let patched_indices = buffer![5u16, 1030].into_array();
        let patched_values = buffer![9999u16, 8888].into_array();

        let patches = Patches::new(4096, 0, patched_indices, patched_values, None, &mut ctx)?;
        let array = Patched::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Slice to create offset > 0.
        // After slice(5..4096), logical position 0 = original position 5 (patched to 9999).
        let sliced = array.slice(5..4096)?.optimize_ctx(&SESSION)?;
        assert_eq!(sliced.len(), 4091);

        // Filter that touches the first 2 chunks.
        // Logical indices: 0 (was 5, patched), 1 (was 6, value 6), 1025 (was 1030, patched).
        let mask = Mask::from_indices(4091, vec![0, 1, 1025]);
        let filtered = sliced.filter(mask)?.optimize_ctx(&SESSION)?;

        // Expected: 9999 (patched at logical 0), 6 (original at logical 1), 8888 (patched at logical 1025).
        let expected = PrimitiveArray::from_iter([9999u16, 6, 8888]);
        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_filter_with_offset_last_chunk() -> VortexResult<()> {
        // Test filtering with offset > 0 where the mask touches the last chunk.
        // This ensures we don't accidentally slice past the end of the array or mask.
        let mut ctx = SESSION.create_execution_ctx();

        // Create a 6-chunk array (6144 elements).
        let array = buffer![u16::MIN; 6144].into_array();
        // Patches near the end of the array at indices 5000 and 6000.
        let patched_indices = buffer![5000u16, 6000].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(6144, 0, patched_indices, patched_values, None, &mut ctx)?;

        let patched = Patched::from_array_and_patches(array, &patches, &mut ctx)?;

        // Slice at chunk boundary to create offset > 0.
        // Slice [1024..6144] gives us 5120 elements (5 chunks).
        // Original patches at 5000 and 6000 become relative indices 3976 and 4976.
        let sliced = patched
            .into_array()
            .slice(1024..6144)?
            .optimize_ctx(&SESSION)?;
        assert_eq!(sliced.len(), 5120);

        // Filter that touches only the last 2 chunks (chunks 3 and 4).
        // Chunk 3: indices 3072-4095, Chunk 4: indices 4096-5119.
        // Patch at 3976 is in chunk 3, patch at 4976 is in chunk 4.
        let mask = Mask::from_indices(5120, vec![3976, 4976, 5119]);

        let filtered = sliced.filter(mask)?.optimize_ctx(&SESSION)?;

        // Expected: patch at 3976 (was 5000), patch at 4976 (was 6000), and MIN at 5119.
        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]);

        assert_arrays_eq!(filtered, expected, &mut ctx);

        Ok(())
    }
}
