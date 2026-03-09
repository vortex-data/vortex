// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod aggregate;
mod cast;
mod fill_null;
mod filter;
mod is_constant;
mod is_sorted;
pub(crate) mod kernel;
mod mask;
mod min_max;
pub(crate) mod rules;
mod slice;
mod sum;
mod take;
mod zip;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[rstest]
    // Basic chunked arrays
    #[case::chunked_primitive(ChunkedArray::try_new(
        vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![4i32, 5].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_nullable(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            PrimitiveArray::from_option_iter([Some(4i32), Some(5)]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::Nullable),
    ).unwrap())]
    // Many chunks
    #[case::many_small_chunks(ChunkedArray::try_new(
        (0..10).map(|i| buffer![i as i64, i as i64 + 10, i as i64 + 20].into_array()).collect(),
        DType::Primitive(PType::I64, Nullability::NonNullable),
    ).unwrap())]
    // Edge cases
    #[case::single_chunk(ChunkedArray::try_new(
        vec![buffer![1i32, 2, 3, 4, 5].into_array()],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case::empty_chunks_mixed(ChunkedArray::try_new(
        vec![
            buffer![1u64, 2].into_array(),
            PrimitiveArray::empty::<u64>(Nullability::NonNullable).into_array(),
            buffer![3u64, 4].into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    ).unwrap())]
    // Large chunks
    #[case::large_chunks(ChunkedArray::try_new(
        vec![
            buffer![0..1000i32].into_array(),
            buffer![1000..2000i32].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    // Mixed validity across chunks
    #[case::mixed_validity(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1.0f32), Some(2.0), Some(3.0)]).into_array(),
            PrimitiveArray::from_option_iter([Some(4.0f32), None, Some(6.0)]).into_array(),
            PrimitiveArray::from_option_iter([Some(7.0f32), Some(8.0)]).into_array(),
        ],
        DType::Primitive(PType::F32, Nullability::Nullable),
    ).unwrap())]

    fn test_chunked_consistency(#[case] array: ChunkedArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::chunked_i32_basic(ChunkedArray::try_new(
        vec![
            buffer![10i32, 20, 30].into_array(),
            buffer![40i32, 50, 60].into_array(),
            buffer![70i32, 80, 90].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_u32_basic(ChunkedArray::try_new(
        vec![
            buffer![100u32, 200, 300].into_array(),
            buffer![400u32, 500].into_array(),
        ],
        DType::Primitive(PType::U32, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_i64_basic(ChunkedArray::try_new(
        vec![
            buffer![1000i64, 2000].into_array(),
            buffer![3000i64, 4000, 5000].into_array(),
            buffer![6000i64].into_array(),
        ],
        DType::Primitive(PType::I64, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_f32_basic(ChunkedArray::try_new(
        vec![
            buffer![1.5f32, 2.5, 3.5].into_array(),
            buffer![4.5f32, 5.5].into_array(),
        ],
        DType::Primitive(PType::F32, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_f64_basic(ChunkedArray::try_new(
        vec![
            buffer![10.1f64, 20.2].into_array(),
            buffer![30.3f64, 40.4, 50.5].into_array(),
        ],
        DType::Primitive(PType::F64, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_single_chunk(ChunkedArray::try_new(
        vec![
            buffer![1i32, 2, 3, 4, 5].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_many_small_chunks(ChunkedArray::try_new(
        (0..10).map(|i| buffer![i * 10, i * 10 + 1].into_array()).collect(),
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_nullable(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
            PrimitiveArray::from_option_iter([Some(400i32), Some(500)]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::Nullable),
    ).unwrap())]
    #[case::chunked_mixed_chunk_sizes(ChunkedArray::try_new(
        vec![
            buffer![1i64].into_array(),
            buffer![2i64, 3, 4, 5].into_array(),
            buffer![6i64, 7].into_array(),
            buffer![8i64, 9, 10, 11, 12].into_array(),
        ],
        DType::Primitive(PType::I64, Nullability::NonNullable),
    ).unwrap())]
    #[case::chunked_large(ChunkedArray::try_new(
        vec![
            buffer![0..500].into_array().into_array(),
            buffer![500..1000].into_array().into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ).unwrap())]
    fn test_chunked_binary_numeric(#[case] array: ChunkedArray) {
        test_binary_numeric_array(array.into_array())
    }
}
