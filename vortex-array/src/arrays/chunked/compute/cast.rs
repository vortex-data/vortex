// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::compute::CastKernel;
use crate::compute::CastKernelAdapter;
use crate::compute::cast;
use crate::register_kernel;

impl CastKernel for ChunkedVTable {
    fn cast(&self, array: &ChunkedArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let mut cast_chunks = Vec::new();
        for chunk in array.chunks() {
            cast_chunks.push(cast(chunk, dtype)?);
        }

        // SAFETY: casting all chunks retains all chunks have same DType
        unsafe {
            Ok(Some(
                ChunkedArray::new_unchecked(cast_chunks, dtype.clone()).into_array(),
            ))
        }
    }
}

register_kernel!(CastKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArray;
    use crate::assert_arrays_eq;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;

    #[test]
    fn test_cast_chunked() {
        let arr0 = buffer![0u32, 1].into_array();
        let arr1 = buffer![2u32, 3].into_array();

        let chunked = ChunkedArray::try_new(
            vec![arr0, arr1],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        // Two levels of chunking, just to be fancy.
        let root = ChunkedArray::try_new(
            vec![chunked],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        let result = cast(
            &root,
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u64, 1, 2, 3]));
    }

    #[rstest]
    #[case(ChunkedArray::try_new(
        vec![buffer![0u32, 1, 2].into_array(), buffer![3u32, 4].into_array()],
        DType::Primitive(PType::U32, Nullability::NonNullable)
    ).unwrap().into_array())]
    #[case(ChunkedArray::try_new(
        vec![
            buffer![-10i32, -5, 0].into_array(),
            buffer![5i32, 10].into_array()
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable)
    ).unwrap().into_array())]
    #[case(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1.5f32), None, Some(2.5)]).into_array(),
            PrimitiveArray::from_option_iter([Some(3.5f32), Some(4.5)]).into_array()
        ],
        DType::Primitive(PType::F32, Nullability::Nullable)
    ).unwrap().into_array())]
    #[case(ChunkedArray::try_new(
        vec![buffer![42u8].into_array()],
        DType::Primitive(PType::U8, Nullability::NonNullable)
    ).unwrap().into_array())]
    fn test_cast_chunked_conformance(#[case] array: crate::ArrayRef) {
        test_cast_conformance(array.as_ref());
    }
}
