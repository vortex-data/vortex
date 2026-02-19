// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::ChunkedVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeExecute;
use crate::arrays::chunked::ChunkedArray;
use crate::builtins::ArrayBuiltins;
use crate::canonical::ToCanonical;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;

// TODO(joe): this is pretty unoptimized but better than before. We want canonical using a builder
// we also want to return a chunked array ideally.
fn take_chunked(
    array: &ChunkedArray,
    indices: &dyn Array,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices = indices
        .to_array()
        .cast(DType::Primitive(PType::U64, indices.dtype().nullability()))?
        .to_primitive();

    let indices_mask = indices.validity_mask()?;
    let indices_values = indices.as_slice::<u64>();
    let n = indices_values.len();

    // 1. Sort (value, orig_pos) pairs so indices for the same chunk are contiguous.
    //    Skip null indices — their final_take slots stay 0 and are masked null by validity.
    let mut pairs: Vec<(u64, usize)> = indices_values
        .iter()
        .enumerate()
        .filter(|&(i, _)| indices_mask.value(i))
        .map(|(i, &v)| (v, i))
        .collect();
    pairs.sort_unstable();

    // 2. Fused pass: walk sorted pairs against chunk boundaries.
    //    - Dedup inline → build per-chunk filter masks
    //    - Scatter final_take[orig_pos] = dedup_idx for every pair
    let chunk_offsets = array.chunk_offsets();
    let nchunks = array.nchunks();
    let mut chunks = Vec::with_capacity(nchunks);
    let mut final_take = BufferMut::<u64>::with_capacity(n);
    final_take.push_n(0u64, n);

    let mut cursor = 0usize;
    let mut dedup_idx = 0u64;

    for chunk_idx in 0..nchunks {
        let chunk_start = chunk_offsets[chunk_idx];
        let chunk_end = chunk_offsets[chunk_idx + 1];
        let chunk_len = usize::try_from(chunk_end - chunk_start)?;

        let range_end = cursor + pairs[cursor..].partition_point(|&(v, _)| v < chunk_end);
        let chunk_pairs = &pairs[cursor..range_end];

        if !chunk_pairs.is_empty() {
            let mut local_indices: Vec<usize> = Vec::new();
            for (i, &(val, orig_pos)) in chunk_pairs.iter().enumerate() {
                if cursor + i > 0 && val != pairs[cursor + i - 1].0 {
                    dedup_idx += 1;
                }
                let local = usize::try_from(val - chunk_start)?;
                if local_indices.last() != Some(&local) {
                    local_indices.push(local);
                }
                final_take[orig_pos] = dedup_idx;
            }

            let filter_mask = Mask::from_indices(chunk_len, local_indices);
            chunks.push(array.chunk(chunk_idx).filter(filter_mask)?);
        }

        cursor = range_end;
    }

    let nullability = indices.dtype().nullability();

    let result_dtype = array.dtype().clone().union_nullability(nullability);
    // SAFETY: every chunk came from a filter on a chunk with the same base dtype,
    // unioned with the index nullability.
    let flat = unsafe { ChunkedArray::new_unchecked(chunks, result_dtype) }
        .into_array()
        // TODO(joe): can we relax this.
        .execute::<Canonical>(ctx)?
        .into_array();

    // 4. Single take to restore original order and expand duplicates.
    //    Carry the original index validity so null indices produce null outputs.
    let take_validity = Validity::from_mask(indices_mask, nullability);
    flat.take(PrimitiveArray::new(final_take.freeze(), take_validity).into_array())
}

impl TakeExecute for ChunkedVTable {
    fn take(
        array: &ChunkedArray,
        indices: &dyn Array,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_chunked(array, indices, ctx).map(Some)
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::array::Array;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::chunked::ChunkedArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = buffer![0u64, 0, 6, 4].into_array();

        let result = arr.take(indices.to_array()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 1, 1, 2]));
    }

    #[test]
    fn test_take_nullability() {
        let struct_array =
            StructArray::try_new(FieldNames::default(), vec![], 100, Validity::NonNullable)
                .unwrap();

        let arr = ChunkedArray::from_iter(vec![struct_array.to_array(), struct_array.to_array()]);

        let result = arr
            .take(PrimitiveArray::from_option_iter(vec![Some(0), None, Some(101)]).to_array())
            .unwrap();

        let expect = StructArray::try_new(
            FieldNames::default(),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter(vec![true, false, true]).to_array()),
        )
        .unwrap();
        assert_arrays_eq!(result, expect);
    }

    #[test]
    fn test_empty_take() {
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);

        let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
        let result = arr.take(indices.to_array()).unwrap();

        assert!(result.is_empty());
        assert_eq!(result.dtype(), arr.dtype());
        assert_arrays_eq!(
            result,
            PrimitiveArray::empty::<i32>(Nullability::NonNullable)
        );
    }

    #[test]
    fn test_take_shuffled_indices() -> VortexResult<()> {
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
        let result = arr.take(indices.to_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([8i32, 0, 5, 3, 2, 7, 1, 6, 4])
        );
        Ok(())
    }

    #[test]
    fn test_take_shuffled_large() -> VortexResult<()> {
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
        let result = arr.take(indices_arr.to_array())?;

        // Verify every element.
        let result = result.to_primitive();
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
        let result = arr.take(indices.to_array())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([
                Some(60i32),
                None,
                Some(10),
                Some(40),
                None,
                Some(30)
            ])
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
        test_take_conformance(arr.as_ref());

        // Test with nullable chunked array
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let b = PrimitiveArray::from_option_iter([Some(4i32), Some(5)]);
        let dtype = a.dtype().clone();
        let arr = ChunkedArray::try_new(vec![a.into_array(), b.into_array()], dtype).unwrap();
        test_take_conformance(arr.as_ref());

        // Test with multiple identical chunks
        let chunk = buffer![10i32, 20, 30, 40, 50].into_array();
        let arr = ChunkedArray::try_new(
            vec![chunk.clone(), chunk.clone(), chunk.clone()],
            chunk.dtype().clone(),
        )
        .unwrap();
        test_take_conformance(arr.as_ref());
    }
}
