// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementations of filters for buffers.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512;

mod by_bitview;
mod by_mask;

// TODO(connor): This is super inefficient.
/// Filter elements from a source slice into a destination slice based on the given mask.
///
/// The mask is represented as a slice of bytes (LSB is the first element).
///
/// Returns the true count of the mask (number of elements written to destination).
///
/// This function uses a scalar implementation that reads from source and writes selected
/// elements to the destination buffer.
///
/// # Panics
///
/// Panics if:
/// - `mask.len() != src.len().div_ceil(8)`
/// - `dest.len() < src.len()`
#[inline]
pub fn filter_into_scalar<T: Copy>(src: &[T], dest: &mut [T], mask: &[u8]) -> usize {
    assert_eq!(
        mask.len(),
        src.len().div_ceil(8),
        "Mask length must be src.len().div_ceil(8)"
    );
    assert!(
        dest.len() >= src.len(),
        "Destination buffer must be at least as large as source"
    );

    let mut write_pos = 0;
    let src_len = src.len();

    for read_pos in 0..src_len {
        let byte_idx = read_pos / 8;
        let bit_idx = read_pos % 8;

        if (mask[byte_idx] >> bit_idx) & 1 == 1 {
            dest[write_pos] = src[read_pos];
            write_pos += 1;
        }
    }

    write_pos
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::avx512::filter_into;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use super::avx512::filter_into_avx512;
    use super::*;

    fn create_mask(bits: &[bool]) -> Vec<u8> {
        let mut mask = vec![0u8; bits.len().div_ceil(8)];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                mask[i / 8] |= 1 << (i % 8);
            }
        }
        mask
    }

    fn test_implementation<F>(filter_fn: F)
    where
        F: Fn(&[i32], &mut [i32], &[u8]) -> usize,
    {
        // Test 1: Small array - all elements pass
        let src = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut dest = vec![0; 8];
        let mask = vec![0xFF]; // All 1s
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 8);
        assert_eq!(&dest[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);

        // Test 2: Small array - no elements pass
        let src = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut dest = vec![99; 8];
        let mask = vec![0x00]; // All 0s
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 0);
        // Destination should remain unchanged past the write position
        assert_eq!(dest[0], 99);

        // Test 3: Small array - every other element
        let src = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut dest = vec![0; 8];
        let mask = vec![0x55]; // 01010101
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 4);
        assert_eq!(&dest[..4], &[0, 2, 4, 6]);

        // Test 4: 16 elements - all pass
        let src: Vec<i32> = (0..16).collect();
        let mut dest = vec![0; 16];
        let mask = vec![0xFF, 0xFF];
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 16);
        assert_eq!(
            &dest[..16],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );

        // Test 5: 16 elements - alternating pattern
        let src: Vec<i32> = (0..16).collect();
        let mut dest = vec![0; 16];
        let mask = vec![0xAA, 0xAA]; // 10101010 10101010
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 8);
        assert_eq!(&dest[..8], &[1, 3, 5, 7, 9, 11, 13, 15]);

        // Test 6: Larger array (32 elements)
        let src: Vec<i32> = (0..32).collect();
        let mut dest = vec![0; 32];
        let mask = vec![0xFF, 0x00, 0xFF, 0x00]; // First and third bytes
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 16);
        assert_eq!(
            &dest[..16],
            &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
        );

        // Test 7: Non-aligned size (23 elements)
        let src: Vec<i32> = (0..23).collect();
        let mut dest = vec![0; 23];
        let mask = create_mask(&[
            true, false, true, false, true, false, true, false, // byte 0
            false, true, false, true, false, true, false, true, // byte 1
            true, true, false, false, true, true, false, // byte 2 (partial)
        ]);
        let count = filter_fn(&src, &mut dest, &mask);
        assert_eq!(count, 12);
        assert_eq!(&dest[..12], &[0, 2, 4, 6, 9, 11, 13, 15, 16, 17, 20, 21]);
    }

    #[test]
    fn test_scalar() {
        test_implementation(filter_into_scalar::<i32>);
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_avx512() {
        test_implementation(|src, dest, mask| unsafe {
            filter_into_avx512::<i32>(src, dest, mask)
        });
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_runtime_dispatch() {
        test_implementation(filter_into::<i32>);
    }

    #[expect(clippy::cast_possible_truncation)]
    #[test]
    fn test_all_implementations_match() {
        // Test that all available implementations produce the same results

        // Test various sizes and patterns
        let test_cases: Vec<(usize, Vec<u8>)> = vec![
            (8, vec![0xAA]),                    // 8 elements, alternating
            (16, vec![0xFF, 0xFF]),             // 16 elements, all pass
            (16, vec![0x00, 0x00]),             // 16 elements, none pass
            (32, vec![0x55, 0x55, 0x55, 0x55]), // 32 elements, alternating
            (24, vec![0xFF, 0x00, 0xFF]),       // 24 elements, mixed
            (100, vec![0xFF; 13]),              // 100 elements (needs 13 bytes)
        ];

        for (size, mask) in test_cases {
            let src: Vec<i32> = (0..size as i32).collect();
            let mut dest_scalar = vec![0i32; size];
            let count_scalar = filter_into_scalar::<i32>(&src, &mut dest_scalar, &mask);

            // Test AVX-512 on x86/x86_64
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let mut dest_avx512 = vec![0i32; size];
                let count_avx512 =
                    unsafe { filter_into_avx512::<i32>(&src, &mut dest_avx512, &mask) };
                assert_eq!(
                    count_scalar, count_avx512,
                    "Count mismatch for size {}",
                    size
                );
                assert_eq!(
                    &dest_scalar[..count_scalar],
                    &dest_avx512[..count_avx512],
                    "Data mismatch for size {}",
                    size
                );
            }

            let _ = count_scalar;
        }
    }

    #[expect(clippy::cast_possible_truncation)]
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_large_arrays() {
        // Test with very large arrays to ensure chunking works correctly
        let sizes: Vec<usize> = vec![1024, 1000, 2048, 4096, 10000];

        for size in sizes {
            let src: Vec<i32> = (0..size as i32).collect();
            let mut dest = vec![0i32; size];
            // Create alternating mask
            let mut mask = vec![0u8; size.div_ceil(8)];
            mask.fill(0x55); // 01010101

            let count = filter_into::<i32>(&src, &mut dest, &mask);
            assert_eq!(count, size / 2);

            // Verify first few and last few elements
            (0..10.min(count)).for_each(|i| {
                assert_eq!(dest[i], (i * 2) as i32);
            });
        }
    }
}
