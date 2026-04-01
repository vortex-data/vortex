// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Patched;
use crate::arrays::PatchedArray;
use crate::arrays::slice::SliceReduce;
use crate::stats::ArrayStats;

impl SliceReduce for Patched {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // We **always** slice the patches at 1024-element chunk boundaries. We keep the offset + len
        // around so that when we execute we know how much to chop off.
        let new_offset = (range.start + array.offset) % 1024;
        let new_len = range.end - range.start;

        let chunk_start = (range.start + array.offset) / 1024;
        let chunk_stop = (range.end + array.offset).div_ceil(1024);
        let sliced_lane_offsets = array
            .lane_offsets()
            .slice((chunk_start * array.n_lanes)..(chunk_stop * array.n_lanes) + 1)?;

        // Unlike the patches, we slice the inner to the exact range. This is handled
        // at execution time by making sure to skip patch indices that are < offset
        // or >= len.
        let inner = array.base_array().slice(range.start..range.end)?;

        Ok(Some(
            PatchedArray {
                slots: vec![
                    Some(inner),
                    Some(sliced_lane_offsets),
                    Some(array.patch_indices().clone()),
                    Some(array.patch_values().clone()),
                ],
                n_lanes: array.n_lanes,
                offset: new_offset,
                len: new_len,
                stats_set: ArrayStats::default(),
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::NativePType;
    use crate::patches::Patches;

    #[test]
    fn test_reduce() -> VortexResult<()> {
        let values = buffer![0u16; 512].into_array();
        let patch_indices = buffer![1u32, 8, 30].into_array();
        let patch_values = buffer![u16::MAX; 3].into_array();
        let patches = Patches::new(512, 0, patch_indices, patch_values, None)?;

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let patched_array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)?;

        let sliced = patched_array.into_array().slice(1..10)?;

        insta::assert_snapshot!(
            sliced.display_tree_encodings_only(),
            @r#"
            root: vortex.patched(u16, len=9)
              inner: vortex.primitive(u16, len=9)
              lane_offsets: vortex.primitive(u32, len=33)
              patch_indices: vortex.primitive(u16, len=3)
              patch_values: vortex.primitive(u16, len=3)
            "#);

        let executed = sliced.execute::<Canonical>(&mut ctx)?.into_primitive();

        assert_eq!(
            &[u16::MAX, 0, 0, 0, 0, 0, 0, u16::MAX, 0],
            executed.as_slice::<u16>()
        );

        Ok(())
    }

    #[rstest]
    #[case::trivial(buffer![1u64; 2], buffer![1u32], buffer![u64::MAX], 1..2)]
    #[case::one_chunk(buffer![0u64; 1024], buffer![1u32, 8, 30], buffer![u64::MAX; 3], 1..10)]
    #[case::multichunk(buffer![1u64; 10_000], buffer![0u32, 1, 2, 3, 4, 16, 17, 18, 19, 1024, 2048, 2049], buffer![u64::MAX; 12], 1024..5000)]
    fn test_cases<T: NativePType>(
        #[case] inner: Buffer<T>,
        #[case] patch_indices: Buffer<u32>,
        #[case] patch_values: Buffer<T>,
        #[case] range: Range<usize>,
    ) {
        // Create patched array.
        let patches = Patches::new(
            inner.len(),
            0,
            patch_indices.into_array(),
            patch_values.into_array(),
            None,
        )
        .unwrap();

        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let patched_array =
            PatchedArray::from_array_and_patches(inner.into_array(), &patches, &mut ctx)
                .unwrap()
                .into_array();

        // Verify that applying slice first yields same result as applying slice at end.
        let slice_first = patched_array
            .slice(range.clone())
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();

        let slice_last = patched_array
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive()
            .slice(range)
            .unwrap();

        assert_arrays_eq!(slice_first, slice_last);
    }

    #[test]
    fn test_stacked_slices() {
        let values = PrimitiveArray::from_iter(0u64..10_000).into_array();

        let patched_indices = buffer![1u32, 2, 1024, 2048, 3072, 3088].into_array();
        let patched_values = buffer![0u64, 1, 2, 3, 4, 5].into_array();

        let patches = Patches::new(10_000, 0, patched_indices, patched_values, None).unwrap();
        let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

        let patched_array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let sliced = patched_array
            .slice(1024..5000)
            .unwrap()
            .slice(1..2065)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_array();

        let mut expected = BufferMut::from_iter(1025u64..=3088);
        expected[1023] = 3;
        expected[2047] = 4;
        expected[2063] = 5;

        let expected = expected.into_array();

        assert_arrays_eq!(expected, sliced);
    }
}
