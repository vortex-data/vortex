// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::arrays::chunked::ChunkedArray;
use crate::arrays::{ChunkedVTable, PrimitiveArray};
use crate::compute::{TakeKernel, TakeKernelAdapter, cast, take};
use crate::validity::Validity;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for ChunkedVTable {
    fn take(&self, array: &ChunkedArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = cast(
            indices,
            &DType::Primitive(PType::U64, indices.dtype().nullability()),
        )?
        .to_primitive()?;

        // TODO(joe): Should we split this implementation based on indices nullability?
        let nullability = indices.dtype().nullability();
        let indices_mask = indices.validity_mask()?;
        let indices = indices.as_slice::<u64>();

        let mut chunks = Vec::new();
        let mut indices_in_chunk = BufferMut::<u64>::empty();
        let mut start = 0;
        let mut stop = 0;
        // We assume indices are non-empty as it's handled in the top-level `take` function
        let mut prev_chunk_idx = array.find_chunk_idx(indices[0].try_into()?).0;
        for idx in indices {
            let idx = usize::try_from(*idx)?;
            let (chunk_idx, idx_in_chunk) = array.find_chunk_idx(idx);

            if chunk_idx != prev_chunk_idx {
                // Start a new chunk
                let indices_in_chunk_array = PrimitiveArray::new(
                    indices_in_chunk.clone().freeze(),
                    Validity::from_mask(indices_mask.slice(start, stop - start), nullability),
                );
                chunks.push(take(
                    array.chunk(prev_chunk_idx)?,
                    indices_in_chunk_array.as_ref(),
                )?);
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
                Validity::from_mask(indices_mask.slice(start, stop - start), nullability),
            );
            chunks.push(take(
                array.chunk(prev_chunk_idx)?,
                indices_in_chunk_array.as_ref(),
            )?);
        }

        Ok(ChunkedArray::new_unchecked(
            chunks,
            array.dtype().clone().union_nullability(nullability),
        )
        .into_array())
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

        let result = take(arr.as_ref(), indices.as_ref())
            .unwrap()
            .to_primitive()
            .unwrap();
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
        assert_eq!(result.scalar_at(0).unwrap(), expect.scalar_at(0).unwrap());
        assert_eq!(result.scalar_at(1).unwrap(), expect.scalar_at(1).unwrap());
        assert_eq!(result.scalar_at(2).unwrap(), expect.scalar_at(2).unwrap());
    }

    #[test]
    fn test_empty_take() {
        let a = buffer![1i32, 2, 3].into_array();
        let arr = ChunkedArray::try_new(vec![a.clone(), a.clone(), a.clone()], a.dtype().clone())
            .unwrap();
        assert_eq!(arr.nchunks(), 3);
        assert_eq!(arr.len(), 9);

        let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
        let result = take(arr.as_ref(), indices.as_ref())
            .unwrap()
            .to_primitive()
            .unwrap();

        assert!(result.is_empty());
        assert_eq!(result.dtype(), arr.dtype());
        assert!(result.as_slice::<i32>().is_empty());
    }
}
