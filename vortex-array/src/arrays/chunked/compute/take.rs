// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::TakeExecute;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;

// TODO(joe): this is pretty unoptimized but better than before. We want canonical using a builder
// we also want to return a chunked array ideally.
fn take_chunked_via_sort(
    array: ArrayView<'_, Chunked>,
    indices: &PrimitiveArray,
    indices_mask: Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices_values = indices.as_slice::<u64>();
    let n = indices_values.len();
    let mut pairs: Vec<(u64, usize)> = indices_values
        .iter()
        .enumerate()
        .filter(|&(position, _)| indices_mask.value(position))
        .map(|(position, &index)| (index, position))
        .collect();
    pairs.sort_unstable();

    if let Some(&(index, _)) = pairs.last() {
        let index = usize::try_from(index)?;
        if index >= array.len() {
            vortex_bail!(OutOfBounds: index, 0, array.len());
        }
    }

    let chunk_offsets = array.chunk_offset_values();
    let nchunks = array.nchunks();
    let mut chunks = Vec::with_capacity(nchunks);
    let mut final_take = BufferMut::<u64>::with_capacity(n);
    final_take.push_n(0u64, n);
    let mut cursor = 0usize;
    let mut dedup_idx = 0u64;

    for chunk_idx in 0..nchunks {
        let chunk_start = chunk_offsets[chunk_idx];
        let chunk_end = chunk_offsets[chunk_idx + 1];
        let chunk_len = chunk_end - chunk_start;
        let chunk_end_u64 = u64::try_from(chunk_end)?;
        let range_end = cursor + pairs[cursor..].partition_point(|&(v, _)| v < chunk_end_u64);
        let chunk_pairs = &pairs[cursor..range_end];

        if !chunk_pairs.is_empty() {
            let mut local_indices = Vec::new();
            for (i, &(value, original_position)) in chunk_pairs.iter().enumerate() {
                if cursor + i > 0 && value != pairs[cursor + i - 1].0 {
                    dedup_idx += 1;
                }
                let local_index = usize::try_from(value)? - chunk_start;
                if local_indices.last() != Some(&local_index) {
                    local_indices.push(local_index);
                }
                final_take[original_position] = dedup_idx;
            }

            chunks.push(
                array
                    .chunk(chunk_idx)
                    .filter(Mask::from_indices(chunk_len, local_indices))?,
            );
        }

        cursor = range_end;
    }

    // SAFETY: every chunk came from a filter on a chunk with the same base dtype.
    let flat = unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()) }
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array();
    let take_validity = Validity::from_mask(indices_mask, indices.dtype().nullability());
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

fn valid_indices_are_monotonic(indices: &[u64], indices_mask: &Mask) -> bool {
    let mut previous = None;
    for (position, &index) in indices.iter().enumerate() {
        if !indices_mask.value(position) {
            continue;
        }
        if previous.is_some_and(|previous| index < previous) {
            return false;
        }
        previous = Some(index);
    }
    true
}

fn take_chunked(
    array: ArrayView<'_, Chunked>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices = indices
        .cast(DType::Primitive(PType::U64, indices.dtype().nullability()))?
        .execute::<PrimitiveArray>(ctx)?;

    let indices_mask = indices
        .as_ref()
        .validity()?
        .execute_mask(indices.as_ref().len(), ctx)?;
    let indices_values = indices.as_slice::<u64>();
    let n = indices_values.len();
    let chunk_offsets = array.chunk_offset_values();
    let nchunks = array.nchunks();

    // For a small unsorted fixed-size-list take over a small number of chunks, sorting has lower
    // fixed overhead and lets the nested elements filter in source order. Larger, monotonic, and
    // non-nested takes use bucketing.
    if matches!(array.dtype(), DType::FixedSizeList(..))
        && n <= 64
        && nchunks <= 16
        && !valid_indices_are_monotonic(indices_values, &indices_mask)
    {
        return take_chunked_via_sort(array, &indices, indices_mask, ctx);
    }

    // Route each valid index into its source chunk. Within each bucket, preserve the request order
    // so the taken chunks can be assembled and then restored to the original cross-chunk order.
    let mut buckets = vec![Vec::<(u64, usize)>::new(); nchunks];
    let mut monotonic = true;
    let mut last_index = None;
    let mut sorted_chunk_idx = 0;

    for (original_position, &index) in indices_values.iter().enumerate() {
        if !indices_mask.value(original_position) {
            continue;
        }

        let index = usize::try_from(index)?;
        if index >= array.len() {
            vortex_bail!(OutOfBounds: index, 0, array.len());
        }

        let still_sorted = last_index.is_none_or(|last_index| index >= last_index);
        let chunk_idx = if monotonic && still_sorted {
            while chunk_offsets[sorted_chunk_idx + 1] <= index {
                sorted_chunk_idx += 1;
            }
            sorted_chunk_idx
        } else {
            monotonic = false;
            chunk_offsets.partition_point(|&offset| offset <= index) - 1
        };
        last_index = Some(index);

        let local_index = u64::try_from(index - chunk_offsets[chunk_idx])?;
        buckets[chunk_idx].push((local_index, original_position));
    }

    let mut chunks = Vec::with_capacity(nchunks);
    let mut final_take = (!monotonic || indices.dtype().is_nullable()).then(|| {
        let mut final_take = BufferMut::<u64>::with_capacity(n);
        final_take.push_n(0u64, n);
        final_take
    });
    let mut grouped_position = 0u64;

    for (chunk_idx, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }

        let mut local_indices = BufferMut::<u64>::with_capacity(bucket.len());
        for (local_index, original_position) in bucket {
            local_indices.push(local_index);
            if let Some(final_take) = &mut final_take {
                final_take[original_position] = grouped_position;
            }
            grouped_position += 1;
        }

        let local_indices =
            PrimitiveArray::new(local_indices.freeze(), Validity::NonNullable).into_array();
        chunks.push(array.chunk(chunk_idx).take(local_indices)?);
    }

    // SAFETY: every chunk came from a take on a chunk with the same base dtype,
    // unioned with the index nullability.
    let flat = unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()) }
        .into_array()
        // TODO(joe): can we relax this.
        .execute::<Canonical>(ctx)?
        .into_array();

    // Non-nullable monotonic indices are already in the same order as the assembled chunks, so no
    // final reorder is needed.
    let Some(final_take) = final_take else {
        return Ok(flat);
    };

    // Restore original order. Carry the original index validity so null indices produce null
    // outputs.
    let take_validity = Validity::from_mask(indices_mask, indices.dtype().nullability());
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

impl TakeExecute for Chunked {
    fn take(
        array: ArrayView<'_, Chunked>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_chunked(array, indices, ctx).map(Some)
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::bitbuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = buffer![0u64, 0, 6, 4].into_array();

        let result = arr.take(indices).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 1, 1, 2]), &mut ctx);
    }

    #[test]
    fn test_take_nullable_values() {
        let mut ctx = array_session().create_execution_ctx();
        let a = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = PrimitiveArray::new(buffer![0u64, 0, 6, 4], Validity::NonNullable);

        let result = arr.take(indices.into_array()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([1i32, 1, 1, 2].map(Some)),
            &mut ctx
        );
    }

    #[test]
    fn test_take_nullable_indices() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = PrimitiveArray::new(
            buffer![0u64, 0, 6, 4],
            Validity::Array(bitbuffer![1 0 0 1].into_array()),
        );

        let result = arr.take(indices.into_array()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(1i32), None, None, Some(2)]),
            &mut ctx
        );
    }

    #[test]
    fn test_take_nullable_struct() {
        let mut ctx = array_session().create_execution_ctx();
        let struct_array =
            StructArray::try_new(FieldNames::default(), vec![], 100, Validity::NonNullable)
                .unwrap();

        let arr = ChunkedArray::from_iter(vec![
            struct_array.clone().into_array(),
            struct_array.into_array(),
        ]);

        let result = arr
            .take(PrimitiveArray::from_option_iter(vec![Some(0), None, Some(101)]).into_array())
            .unwrap();

        let expect = StructArray::try_new(
            FieldNames::default(),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array()),
        )
        .unwrap();
        assert_arrays_eq!(result, expect, &mut ctx);
    }

    #[test]
    fn test_empty_take() {
        let mut ctx = array_session().create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);

        let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
        let result = arr.take(indices.into_array()).unwrap();

        assert!(result.is_empty());
        assert_eq!(result.dtype(), arr.dtype());
        assert_arrays_eq!(
            result,
            PrimitiveArray::empty::<i32>(Nullability::NonNullable),
            &mut ctx
        );
    }

    #[test]
    fn test_take_shuffled_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![0i32, 1, 2].into_array();
        let c1 = buffer![3i32, 4, 5].into_array();
        let c2 = buffer![6i32, 7, 8].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1, c2],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Fully shuffled indices that cross every chunk boundary.
        let indices = buffer![8u64, 0, 5, 3, 2, 7, 1, 6, 4].into_array();
        let result = arr.take(indices)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([8i32, 0, 5, 3, 2, 7, 1, 6, 4]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_shuffled_duplicates_with_empty_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let empty = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
        let arr = ChunkedArray::try_new(
            vec![
                empty.clone(),
                buffer![0i32, 1].into_array(),
                empty.clone(),
                buffer![2i32, 3].into_array(),
                empty,
            ],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        let result = arr.take(buffer![3u64, 0, 2, 0, 3, 1, 2].into_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([3i32, 0, 2, 0, 3, 1, 2]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_small_shuffled_fixed_size_lists() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let first = FixedSizeListArray::new(
            buffer![0i32, 1, 2, 3].into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let second = FixedSizeListArray::new(
            buffer![4i32, 5, 6, 7].into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let dtype = first.dtype().clone();
        let array = ChunkedArray::try_new(vec![first, second], dtype)?;

        let result = array.take(buffer![3u64, 0, 2, 1, 3].into_array())?;
        let expected = FixedSizeListArray::new(
            buffer![6i32, 7, 0, 1, 4, 5, 2, 3, 6, 7].into_array(),
            2,
            Validity::NonNullable,
            5,
        );

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_take_shuffled_large() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let nchunks: i32 = 100;
        let chunk_len: i32 = 1_000;
        let total = nchunks * chunk_len;

        let chunks: Vec<_> = (0..nchunks)
            .map(|c| {
                let start = c * chunk_len;
                PrimitiveArray::from_iter(start..start + chunk_len).into_array()
            })
            .collect();
        let dtype = chunks[0].dtype().clone();
        let arr = ChunkedArray::try_new(chunks, dtype)?;

        // Fisher-Yates shuffle with a fixed seed for determinism.
        let mut indices: Vec<u64> = (0..u64::try_from(total)?).collect();
        let mut seed: u64 = 0xdeadbeef;
        for i in (1..indices.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let indices_arr = PrimitiveArray::new(
            vortex_buffer::Buffer::from(indices.clone()),
            Validity::NonNullable,
        );
        let result = arr.take(indices_arr.into_array())?;

        // Verify every element.
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;
        let result_vals = result.as_slice::<i32>();
        for (pos, &idx) in indices.iter().enumerate() {
            assert_eq!(
                result_vals[pos],
                i32::try_from(idx)?,
                "mismatch at position {pos}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_take_null_indices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let c0 = buffer![10i32, 20, 30].into_array();
        let c1 = buffer![40i32, 50, 60].into_array();
        let arr = ChunkedArray::try_new(
            vec![c0, c1],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )?;

        // Indices with nulls scattered across chunk boundaries.
        let indices =
            PrimitiveArray::from_option_iter([Some(5u64), None, Some(0), Some(3), None, Some(2)]);
        let result = arr.take(indices.into_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([
                Some(60i32),
                None,
                Some(10),
                Some(40),
                None,
                Some(30)
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_chunked_conformance() {
        let a = buffer![1i32, 2, 3].into_array();
        let b = buffer![4i32, 5].into_array();
        let arr = ChunkedArray::try_new(
            vec![a, b],
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
                .dtype()
                .clone(),
        )
        .unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );

        // Test with nullable chunked array
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let b = PrimitiveArray::from_option_iter([Some(4i32), Some(5)]);
        let dtype = a.dtype().clone();
        let arr = ChunkedArray::try_new(vec![a.into_array(), b.into_array()], dtype).unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );

        // Test with multiple identical chunks
        let chunk = buffer![10i32, 20, 30, 40, 50].into_array();
        let arr = ChunkedArray::try_new(
            vec![chunk.clone(), chunk.clone(), chunk.clone()],
            chunk.dtype().clone(),
        )
        .unwrap();
        test_take_conformance(
            &arr.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
