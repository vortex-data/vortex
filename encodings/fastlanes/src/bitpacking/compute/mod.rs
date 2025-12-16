// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod between;
mod cast;
mod filter;
mod is_constant;
mod take;

// TODO(connor): This is duplicated in `encodings/fastlanes/src/bitpacking/kernels/mod.rs`.
fn chunked_indices<F: FnMut(usize, &[usize])>(
    mut indices: impl Iterator<Item = usize>,
    offset: usize,
    mut chunk_fn: F,
) {
    let mut indices_within_chunk: Vec<usize> = Vec::with_capacity(1024);

    let Some(first_idx) = indices.next() else {
        return;
    };

    let mut current_chunk_idx = (first_idx + offset) / 1024;
    indices_within_chunk.push((first_idx + offset) % 1024);
    for idx in indices {
        let new_chunk_idx = (idx + offset) / 1024;

        if new_chunk_idx != current_chunk_idx {
            chunk_fn(current_chunk_idx, &indices_within_chunk);
            indices_within_chunk.clear();
        }

        current_chunk_idx = new_chunk_idx;
        indices_within_chunk.push((idx + offset) % 1024);
    }

    if !indices_within_chunk.is_empty() {
        chunk_fn(current_chunk_idx, &indices_within_chunk);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;

    use crate::BitPackedArray;
    use crate::bitpack_compress::bitpack_encode;
    use crate::bitpacking::compute::chunked_indices;

    #[test]
    fn chunk_indices_repeated() {
        let mut called = false;
        chunked_indices([0; 1025].into_iter(), 0, |chunk_idx, idxs| {
            assert_eq!(chunk_idx, 0);
            assert_eq!(idxs, [0; 1025]);
            called = true;
        });
        assert!(called);
    }

    #[rstest]
    // Basic integer arrays that can be bitpacked
    #[case::u8_small(bitpack_encode(&PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]), 3, None).unwrap())]
    #[case::u16_array(bitpack_encode(&PrimitiveArray::from_iter([10u16, 20, 30, 40, 50]), 6, None).unwrap())]
    #[case::u32_array(bitpack_encode(&PrimitiveArray::from_iter([100u32, 200, 300, 400, 500]), 9, None).unwrap())]
    // Arrays with nulls
    #[case::nullable_u8(bitpack_encode(&PrimitiveArray::from_option_iter([Some(1u8), None, Some(3), Some(4), None]), 3, None).unwrap())]
    #[case::nullable_u32(bitpack_encode(&PrimitiveArray::from_option_iter([Some(100u32), None, Some(300), Some(400), None]), 9, None).unwrap())]
    // Edge cases
    #[case::single_element(bitpack_encode(&PrimitiveArray::from_iter([42u32]), 6, None).unwrap())]
    #[case::all_zeros(bitpack_encode(&PrimitiveArray::from_iter([0u16; 100]), 1, None).unwrap())]
    // Large arrays (multiple chunks - fastlanes uses 1024-element chunks)
    #[case::large_u16(bitpack_encode(&PrimitiveArray::from_iter((0..2048).map(|i| (i % 256) as u16)), 8, None).unwrap())]
    #[case::large_u32(bitpack_encode(&PrimitiveArray::from_iter((0..3000).map(|i| (i % 1024) as u32)), 10, None).unwrap())]
    #[case::large_u8_many_chunks(bitpack_encode(&PrimitiveArray::from_iter((0..5120).map(|i| (i % 128) as u8)), 7, None).unwrap())] // 5 chunks
    #[case::large_nullable(bitpack_encode(&PrimitiveArray::from_option_iter((0..2500).map(|i| if i % 10 == 0 { None } else { Some((i % 512) as u16) })), 9, None).unwrap())]
    // Arrays with specific bit patterns
    #[case::max_value_for_bits(bitpack_encode(&PrimitiveArray::from_iter([7u8, 7, 7, 7, 7]), 3, None).unwrap())] // max value for 3 bits
    #[case::alternating_bits(bitpack_encode(&PrimitiveArray::from_iter([0u16, 255, 0, 255, 0, 255]), 8, None).unwrap())]

    fn test_bitpacked_consistency(#[case] array: BitPackedArray) {
        test_array_consistency(array.as_ref());
    }

    #[rstest]
    #[case::u8_basic(bitpack_encode(&PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]), 3, None).unwrap())]
    #[case::u16_basic(bitpack_encode(&PrimitiveArray::from_iter([10u16, 20, 30, 40, 50]), 6, None).unwrap())]
    #[case::u32_basic(bitpack_encode(&PrimitiveArray::from_iter([100u32, 200, 300, 400, 500]), 9, None).unwrap())]
    #[case::u64_basic(bitpack_encode(&PrimitiveArray::from_iter([1000u64, 2000, 3000, 4000, 5000]), 13, None).unwrap())]
    #[case::i32_basic(bitpack_encode(&PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]), 7, None).unwrap())]
    #[case::large_u32(bitpack_encode(&PrimitiveArray::from_iter((0..100).map(|i| i as u32)), 7, None).unwrap())]
    fn test_bitpacked_binary_numeric(#[case] array: BitPackedArray) {
        test_binary_numeric_array(array.into_array());
    }
}
