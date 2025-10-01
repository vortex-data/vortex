// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::arrays::chunked::ChunkedArray;
use crate::arrays::{ChunkedVTable, PrimitiveArray};
use crate::builders::{ArrayBuilder, builder_with_capacity};
use crate::compute::{TakeKernel, TakeKernelAdapter, cast, take};
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

/// The multiplier for determining the maximum number of resulting chunks before we fall back
/// to using an ArrayBuilder. The threshold is calculated as initial_chunks * CHUNK_GROWTH_MULTIPLIER.
/// When the indices are unsorted, the number of chunks can grow to be as large as the number
/// of indices (worst case), which can lead to poor performance.
const CHUNK_GROWTH_MULTIPLIER: usize = 2;

impl TakeKernel for ChunkedVTable {
    fn take(&self, array: &ChunkedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = cast(
            indices,
            &DType::Primitive(PType::U64, indices.dtype().nullability()),
        )?
        .to_primitive();

        // TODO(joe): Should we split this implementation based on indices nullability?
        let nullability = indices.dtype().nullability();
        let indices_mask = indices.validity_mask();
        let indices = indices.as_slice::<u64>();

        let mut chunks: Vec<ArrayRef> = Vec::new();
        let mut indices_in_chunk = BufferMut::<u64>::empty();
        let mut start = 0;
        let mut stop = 0;
        let mut builder: Option<Box<dyn ArrayBuilder>> = None;
        let max_chunks_threshold = array.nchunks() * CHUNK_GROWTH_MULTIPLIER;

        // We assume indices are non-empty as it's handled in the top-level `take` function
        let mut prev_chunk_idx = array.find_chunk_idx(indices[0].try_into()?).0;
        for idx in indices {
            let idx = usize::try_from(*idx)?;
            let (chunk_idx, idx_in_chunk) = array.find_chunk_idx(idx);

            if chunk_idx != prev_chunk_idx {
                // Check if we're exceeding the threshold
                if chunks.len() >= max_chunks_threshold && builder.is_none() {
                    // Initialize builder and canonicalize all existing chunks
                    let mut new_builder = builder_with_capacity(
                        &array.dtype().clone().union_nullability(nullability),
                        indices.len(),
                    );
                    for chunk in chunks.drain(..) {
                        new_builder.extend_from_array(chunk.as_ref());
                    }
                    builder = Some(new_builder);
                }

                // Process the completed chunk
                let indices_in_chunk_array = PrimitiveArray::new(
                    indices_in_chunk.clone().freeze(),
                    Validity::from_mask(indices_mask.slice(start..stop), nullability),
                );
                let taken = take(array.chunk(prev_chunk_idx), indices_in_chunk_array.as_ref())?;

                if let Some(ref mut b) = builder {
                    b.extend_from_array(taken.as_ref());
                } else {
                    chunks.push(taken);
                }

                indices_in_chunk.clear();
                start = stop;
            }

            indices_in_chunk.push(idx_in_chunk as u64);
            stop += 1;
            prev_chunk_idx = chunk_idx;
        }

        if !indices_in_chunk.is_empty() {
            let indices_in_chunk_array = PrimitiveArray::new(
                indices_in_chunk.freeze(),
                Validity::from_mask(indices_mask.slice(start..stop), nullability),
            );
            let taken = take(array.chunk(prev_chunk_idx), indices_in_chunk_array.as_ref())?;

            if let Some(ref mut b) = builder {
                b.extend_from_array(taken.as_ref());
            } else {
                chunks.push(taken);
            }
        }

        if let Some(mut b) = builder {
            log::trace!(
                "Started with {}, exceeded threshold of {} chunks, merging chunks into a single array",
                array.nchunks(),
                max_chunks_threshold,
            );
            Ok(b.finish())
        } else {
            // SAFETY: take on chunks that all have same DType retains same DType
            unsafe {
                Ok(ChunkedArray::new_unchecked(
                    chunks,
                    array.dtype().clone().union_nullability(nullability),
                )
                .into_array())
            }
        }
    }
}

register_kernel!(TakeKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{FieldNames, Nullability};

    use crate::IntoArray;
    use crate::array::Array;
    use crate::arrays::chunked::ChunkedArray;
    use crate::arrays::{BoolArray, PrimitiveArray, StructArray};
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;

    #[test]
    fn test_take() {
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);
        let indices = buffer![0u64, 0, 6, 4].into_array();

        let result = take(arr.as_ref(), indices.as_ref()).unwrap().to_primitive();
        assert_eq!(result.as_slice::<i32>(), &[1, 1, 1, 2]);
    }

    #[test]
    fn test_take_nullability() {
        let struct_array =
            StructArray::try_new(FieldNames::default(), vec![], 100, Validity::NonNullable)
                .unwrap();

        let arr = ChunkedArray::from_iter(vec![struct_array.to_array(), struct_array.to_array()]);

        let result = take(
            arr.as_ref(),
            PrimitiveArray::from_option_iter(vec![Some(0), None, Some(101)]).as_ref(),
        )
        .unwrap();

        let expect = StructArray::try_new(
            FieldNames::default(),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter(vec![true, false, true]).to_array()),
        )
        .unwrap();
        assert_eq!(result.dtype(), expect.dtype());
        assert_eq!(result.scalar_at(0), expect.scalar_at(0));
        assert_eq!(result.scalar_at(1), expect.scalar_at(1));
        assert_eq!(result.scalar_at(2), expect.scalar_at(2));
    }

    #[test]
    fn test_empty_take() {
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);

        let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
        let result = take(arr.as_ref(), indices.as_ref()).unwrap().to_primitive();

        assert!(result.is_empty());
        assert_eq!(result.dtype(), arr.dtype());
        assert!(result.as_slice::<i32>().is_empty());
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

    #[test]
    fn test_take_unsorted_exceeds_threshold() {
        // Create a chunked array with many chunks
        let chunk = buffer![1i32, 2, 3].into_array();
        let chunks: Vec<_> = (0..100).map(|_| chunk.clone()).collect();
        let arr = ChunkedArray::try_new(chunks, chunk.dtype().clone()).unwrap();

        // Create unsorted indices that will create more than MAX_CHUNKS_THRESHOLD chunks
        // Each alternating index will be from a different chunk
        let mut indices = Vec::new();
        for i in 0..100 {
            indices.push((i * 3) as u64); // First element of each chunk
        }
        // Now add indices in reverse order to create an unsorted pattern
        for i in (0..100).rev() {
            indices.push(((i * 3) + 1) as u64); // Second element of each chunk
        }

        let indices_array = PrimitiveArray::new(indices, Validity::NonNullable);
        let result = take(arr.as_ref(), indices_array.as_ref()).unwrap();

        // Verify the result is correct
        assert_eq!(result.len(), 200);
        let result_primitive = result.to_primitive();
        let result_slice = result_primitive.as_slice::<i32>();

        // First 100 elements should be 1s (first element of each chunk)
        for i in 0..100 {
            assert_eq!(result_slice[i], 1);
        }
        // Next 100 elements should be 2s (second element of each chunk)
        for i in 100..200 {
            assert_eq!(result_slice[i], 2);
        }
    }

    #[test]
    fn test_take_unsorted_within_threshold() {
        // Create a smaller chunked array that won't exceed the threshold
        let chunk = buffer![10i32, 20, 30].into_array();
        let chunks: Vec<_> = (0..10)
            .map(|idx| buffer![10 * idx, 1 + 10 * idx, 2 + 10 * idx].into_array())
            .collect();
        let arr = ChunkedArray::try_new(chunks, chunk.dtype().clone()).unwrap();

        // Create unsorted indices
        let indices = buffer![0u64, 15, 5, 20, 10, 25].into_array();
        let result = take(arr.as_ref(), indices.as_ref()).unwrap();

        // Verify the result is correct
        assert_eq!(result.len(), 6);
        let result_primitive = result.to_primitive();
        assert_eq!(result_primitive.as_slice::<i32>(), &[0, 50, 12, 62, 31, 81]);
    }
}
