// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::Patched;
use crate::arrays::filter::FilterReduce;

impl FilterReduce for Patched {
    fn filter(array: &Self::Array, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
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

                (first / 1024, last.div_ceil(1024))
            }
        };

        // If all chunks already covered, there is nothing to do.
        if chunk_start == 0 && chunk_stop == array.n_chunks {
            return Ok(None);
        }

        let sliced = array.slice_chunks(chunk_start..chunk_stop)?;

        let slice_start = chunk_start * 1024;
        let slice_end = (chunk_stop * 1024).min(array.len());
        let remainder = mask.slice(slice_start..slice_end);

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

    use crate::DynArray;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::FilterArray;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;

    #[test]
    fn test_filter_noop() -> VortexResult<()> {
        let array = buffer![u16::MIN; 5].into_array();
        let patched_indices = buffer![3u8, 4].into_array();
        let patched_values = buffer![u16::MAX; 2].into_array();

        let patches = Patches::new(5, 0, patched_indices, patched_values, None)?;

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        let filtered = FilterArray::new(
            array.clone(),
            Mask::from_iter([true, false, false, false, true]),
        )
        .into_array();

        let reduced = array.vtable().reduce_parent(&array, &filtered, 0)?;

        // Filter does not get pushed through to child because it does not prune any chunks.
        assert!(reduced.is_none());

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

        // Filter that only touches the middle 2 chunks
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);

        let filtered = FilterArray::new(array.clone(), mask).into_array();
        let reduced = array.vtable().reduce_parent(&array, &filtered, 0)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MAX, u16::MIN]).into_array();

        assert_arrays_eq!(expected, reduced.unwrap());

        Ok(())
    }

    #[test]
    fn test_filter_complex() -> VortexResult<()> {
        // Basic test: filter with mask that crosses boundaries.
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let array = buffer![u16::MIN; 4096].into_array();
        let patched_indices = buffer![1024u16, 1025].into_array();
        let patched_values = buffer![u16::MAX, u16::MAX].into_array();

        let patches = Patches::new(4096, 1, patched_indices, patched_values, None)?;

        let array = PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array();

        // Filter that only touches the middle 2 chunks
        let mask = Mask::from_indices(4096, vec![1024, 1025, 3000]);

        let filtered = FilterArray::new(array.clone(), mask).into_array();
        let reduced = array.vtable().reduce_parent(&array, &filtered, 0)?;

        let expected = PrimitiveArray::from_iter([u16::MAX, u16::MIN, u16::MIN]).into_array();

        assert_arrays_eq!(expected, reduced.unwrap());

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
        let sliced = patched.slice(1024..5120)?.into_array();
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
}
