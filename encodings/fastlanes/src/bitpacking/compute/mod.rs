// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod filter;
pub(crate) mod is_constant;
mod slice;
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::NativePType;

    use crate::BitPackedArray;
    use crate::bitpack_compress::BitPackEncoder;
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
    #[case::u8_small([Some(1u8), Some(2), Some(3), Some(4), Some(5)], 3)]
    #[case::u16_array([Some(10u16), Some(20), Some(30), Some(40), Some(50)], 6)]
    #[case::u32_array([Some(100u32), Some(200), Some(300), Some(400), Some(500)], 9)]
    // Arrays with nulls
    #[case::nullable_u8([Some(1u8), None, Some(3), Some(4), None], 3)]
    #[case::nullable_u32([Some(100u32), None, Some(300), Some(400), None], 9)]
    // Edge cases
    #[case::single_element([Some(42u32)], 6)]
    #[case::all_zeros([Some(0u16); 100], 1)]
    // Large arrays (multiple chunks - fastlanes uses 1024-element chunks)
    #[case::large_u16((0..2048).map(|i| Some((i % 256) as u16)), 8)]
    #[case::large_u32((0..3000).map(|i| Some((i % 1024) as u32)), 10)]
    #[case::large_u8_many_chunks((0..5120).map(|i| Some((i % 128) as u8)), 7)]
    #[case::large_nullable((0..2500).map(|i| if i % 10 == 0 { None } else { Some((i % 512) as u16) }), 9)]
    // Arrays with specific bit patterns
    #[case::max_value_for_bits([Some(7u8), Some(7), Some(7), Some(7), Some(7)], 3)] // max value for 3 bits
    #[case::alternating_bits([Some(0u16), Some(255), Some(0), Some(255), Some(0), Some(255)], 8)]
    fn test_bitpacked_consistency<T: NativePType>(
        #[case] values: impl IntoIterator<Item = Option<T>>,
        #[case] bit_width: u8,
    ) {
        let parray = PrimitiveArray::from_option_iter(values);
        let packed = BitPackEncoder::new(&parray)
            .with_bit_width(bit_width)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();
        test_array_consistency(&packed);
    }

    #[rstest]
    #[case::u8_basic([1u8, 2, 3, 4, 5], 3)]
    #[case::u16_basic([10u16, 20, 30, 40, 50], 6)]
    #[case::u32_basic([100u32, 200, 300, 400, 500], 9)]
    #[case::u64_basic([1000u64, 2000, 3000, 4000, 5000], 13)]
    #[case::i32_basic([10i32, 20, 30, 40, 50], 7)]
    #[case::large_u32((0..100).map(|i| i as u32), 7)]
    fn test_bitpacked_binary_numeric<T: NativePType>(
        #[case] values: impl IntoIterator<Item = T>,
        #[case] bit_width: u8,
    ) {
        let parray = PrimitiveArray::from_iter(values);
        let packed = BitPackEncoder::new(&parray)
            .with_bit_width(bit_width)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();
        test_binary_numeric_array(packed);
    }
}
