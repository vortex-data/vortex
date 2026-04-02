// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Chunked {
    fn cast(array: ArrayView<'_, Chunked>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let mut cast_chunks = Vec::new();
        for chunk in array.iter_chunks() {
            cast_chunks.push(chunk.cast(dtype.clone())?);
        }

        // SAFETY: casting all chunks retains all chunks have same DType
        unsafe {
            Ok(Some(
                ChunkedArray::new_unchecked(cast_chunks, dtype.clone()).into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

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

        let result = root
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))
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
        test_cast_conformance(&array);
    }
}
