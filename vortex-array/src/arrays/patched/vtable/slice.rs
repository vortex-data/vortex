// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Patched {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        array.slice_range(range).map(Some)
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
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patched;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patched::PatchedArrayExt;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::dtype::NativePType;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;

    #[test]
    fn test_reduce() -> VortexResult<()> {
        let values = buffer![0u16; 512].into_array();
        let patch_indices = buffer![1u32, 8, 30].into_array();
        let patch_values = buffer![u16::MAX; 3].into_array();
        let patches = Patches::new(512, 0, patch_indices, patch_values, None)?;

        let mut ctx = crate::array_session().create_execution_ctx();

        let patched_array = Patched::from_array_and_patches(values, &patches, &mut ctx)?;

        let sliced = patched_array.into_array().slice(1..10)?;

        insta::assert_snapshot!(
            sliced.display_tree_encodings_only(),
            @r#"
            root: vortex.patched(u16, len=9)
              inner: vortex.primitive(u16, len=9)
              patch_indices: vortex.primitive(u16, len=3)
              patch_values: vortex.primitive(u16, len=3)
              chunk_offsets: vortex.primitive(u32, len=2)
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
    #[case::unaligned(buffer![1u64; 10_000], buffer![0u32, 1030, 1031, 2200, 2999, 9999], buffer![u64::MAX; 6], 1031..3000)]
    fn test_cases<T: NativePType>(
        #[case] inner: Buffer<T>,
        #[case] patch_indices: Buffer<u32>,
        #[case] patch_values: Buffer<T>,
        #[case] range: Range<usize>,
    ) -> VortexResult<()> {
        let patches = Patches::new(
            inner.len(),
            0,
            patch_indices.into_array(),
            patch_values.into_array(),
            None,
        )?;

        let mut ctx = crate::array_session().create_execution_ctx();

        let patched_array =
            Patched::from_array_and_patches(inner.into_array(), &patches, &mut ctx)?.into_array();

        // Verify that applying slice first yields same result as applying slice at end.
        let slice_first = patched_array
            .slice(range.clone())?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        let slice_last = patched_array
            .execute::<Canonical>(&mut ctx)?
            .into_primitive()
            .slice(range)?;

        assert_arrays_eq!(slice_first, slice_last, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_stacked_slices() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u64..10_000).into_array();

        let patched_indices = buffer![1u32, 2, 1024, 2048, 3072, 3088].into_array();
        let patched_values = buffer![0u64, 1, 2, 3, 4, 5].into_array();

        let patches = Patches::new(10_000, 0, patched_indices, patched_values, None)?;
        let mut ctx = crate::array_session().create_execution_ctx();

        let patched_array =
            Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        let sliced = patched_array
            .slice(1024..5000)?
            .slice(1..2065)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        let mut expected = BufferMut::from_iter(1025u64..=3088);
        expected[1023] = 3;
        expected[2047] = 4;
        expected[2063] = 5;

        let expected = expected.into_array();

        assert_arrays_eq!(expected, sliced, &mut ctx);
        Ok(())
    }

    #[test]
    fn slice_keeps_the_grid_and_shares_patch_children() -> VortexResult<()> {
        let values = buffer![0u32; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![5u32, 1030, 1031, 2200, 2999, 4095].into_array(),
            buffer![1u32, 2, 3, 4, 5, 6].into_array(),
            None,
        )?;
        let mut ctx = crate::array_session().create_execution_ctx();
        let patched = Patched::from_array_and_patches(values, &patches, &mut ctx)?;

        let sliced = patched.into_array().slice(1031..3000)?.optimize()?;
        let sliced = sliced.as_::<Patched>();

        // Row 1031 sits at local index 7 of chunk 1, so the slice starts mid-chunk.
        assert_eq!(sliced.offset(), 7);
        assert_eq!(sliced.n_chunks(), 2);
        assert_eq!(sliced.chunk_offsets().len(), 3);
        // The patch children are shared, so their dead prefix and suffix remain.
        assert_eq!(sliced.patch_indices().len(), 6);
        assert_eq!(
            sliced.chunk_offsets().as_::<Primitive>().as_slice::<u32>(),
            &[1, 3, 5]
        );

        let executed = sliced.array().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let mut expected = BufferMut::<u32>::zeroed(3000 - 1031);
        expected[0] = 3;
        expected[2200 - 1031] = 4;
        expected[2999 - 1031] = 5;
        assert_arrays_eq!(expected.into_array(), executed, &mut ctx);
        Ok(())
    }

    #[test]
    fn slice_without_patches_drops_the_layer() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u32..4096).into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![5u32, 3000].into_array(),
            buffer![100u32, 200].into_array(),
            None,
        )?;
        let mut ctx = crate::array_session().create_execution_ctx();
        let patched = Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array();

        // Chunk 1 has no patches at all.
        let sliced = patched.slice(1024..2048)?.optimize()?;
        assert!(sliced.is::<Primitive>());

        // Chunk 0 has a patch, but not inside this range.
        let sliced = patched.slice(6..1000)?.optimize()?;
        assert!(sliced.is::<Primitive>());
        assert_arrays_eq!(
            PrimitiveArray::from_iter(6u32..1000),
            sliced.execute::<PrimitiveArray>(&mut ctx)?,
            &mut ctx
        );
        Ok(())
    }
}
