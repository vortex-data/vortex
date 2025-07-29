// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod elementwise;
mod fill_null;
mod filter;
mod invert;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod sum;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use crate::arrays::{ChunkedArray, PrimitiveArray};
    use crate::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use crate::IntoArray;

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
            PrimitiveArray::from_iter(0..1000i32).into_array(),
            PrimitiveArray::from_iter(1000..2000i32).into_array(),
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
        test_array_consistency(array.as_ref());
    }
}
