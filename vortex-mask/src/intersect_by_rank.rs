// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;

use crate::AllOr;
use crate::Mask;

/// Extract up to 64 bits starting at bit position `start` from pre-computed chunks.
#[inline]
fn extract_bits_from_chunks(chunks: &[u64], remainder: u64, start: usize) -> u64 {
    let chunk_idx = start / 64;
    let bit_offset = start % 64;
    let num_full_chunks = chunks.len();

    let first_chunk = if chunk_idx < num_full_chunks {
        chunks[chunk_idx]
    } else {
        remainder
    };

    if bit_offset == 0 {
        first_chunk
    } else {
        let bits_from_first = first_chunk >> bit_offset;
        let second_chunk = if chunk_idx + 1 < num_full_chunks {
            chunks[chunk_idx + 1]
        } else if chunk_idx + 1 == num_full_chunks {
            remainder
        } else {
            0
        };
        bits_from_first | (second_chunk << (64 - bit_offset))
    }
}

/// Portable implementation of PDEP (parallel bit deposit).
///
/// Deposits the low bits of `source` at the positions indicated by 1-bits in `mask`.
#[inline]
fn pdep_portable(mut source: u64, mut mask: u64) -> u64 {
    let mut result = 0u64;
    while mask != 0 {
        let lowest_bit = mask & mask.wrapping_neg();
        if source & 1 != 0 {
            result |= lowest_bit;
        }
        source >>= 1;
        mask &= mask - 1;
    }
    result
}

impl Mask {
    /// Simple baseline implementation using indices lookup.
    ///
    /// This is a straightforward O(mask.true_count) algorithm that iterates over
    /// the mask's true indices and sets corresponding bits in the output.
    /// Used for benchmarking comparison against the optimized PDEP implementation.
    #[doc(hidden)]
    pub fn intersect_by_rank_simple(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.indices(), mask.indices()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_indices), AllOr::Some(mask_indices)) => {
                let len = self.len();
                if mask_indices.is_empty() {
                    return Self::new_false(len);
                }

                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::zeroed(num_chunks);
                let chunks = buffer.as_mut_slice();

                for &mask_idx in mask_indices {
                    // SAFETY: mask_idx < mask.len() == self.true_count() == self_indices.len()
                    let result_idx = unsafe { *self_indices.get_unchecked(mask_idx) };
                    chunks[result_idx / 64] |= 1u64 << (result_idx % 64);
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }

    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// This uses a chunk-based algorithm optimized for correlated data patterns.
    /// For chunks that are all 1s (runs of consecutive trues), it directly copies
    /// 64 bits from the rank mask. For mixed chunks, it uses PDEP-style bit scattering.
    ///
    /// # Examples
    ///
    /// Keep the third and fifth set values from mask `m1`:
    /// ```
    /// use vortex_mask::Mask;
    ///
    /// let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
    /// let m2 = Mask::from_iter([false, false, true, false, true]);
    /// assert_eq!(
    ///     m1.intersect_by_rank(&m2),
    ///     Mask::from_iter([false, false, false, false, true, false, false, true])
    /// );
    /// ```
    pub fn intersect_by_rank(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let num_chunks = len.div_ceil(64);
                let mut buffer: BufferMut<u64> = BufferMut::with_capacity(num_chunks);
                let mut rank = 0usize;

                let self_chunks = self_buffer.chunks();
                let mask_chunks = mask_buffer.chunks();
                let mask_chunk_vec: Vec<u64> = mask_chunks.iter().collect();
                let mask_remainder = mask_chunks.remainder_bits();

                // Process full 64-bit chunks
                for self_chunk in self_chunks.iter() {
                    let popcount = self_chunk.count_ones() as usize;

                    let result_chunk = if self_chunk == 0 {
                        0u64
                    } else if self_chunk == u64::MAX {
                        // Fast path: copy 64 bits directly from mask
                        extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank)
                    } else {
                        // Scatter rank bits according to self pattern
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep_portable(rank_bits, self_chunk)
                    };

                    rank += popcount;
                    // SAFETY: we allocated enough capacity
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                // Handle remainder bits
                let remainder = len % 64;
                if remainder != 0 {
                    let self_chunk = self_chunks.remainder_bits();
                    let popcount = self_chunk.count_ones() as usize;

                    let result_chunk = if self_chunk == 0 || popcount == 0 {
                        0u64
                    } else {
                        let rank_bits =
                            extract_bits_from_chunks(&mask_chunk_vec, mask_remainder, rank);
                        pdep_portable(rank_bits, self_chunk)
                    };

                    // SAFETY: we allocated enough capacity
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                buffer.truncate(len.div_ceil(8));
                Self::from_buffer(BitBuffer::new(buffer.freeze().into_byte_buffer(), len))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;

    use crate::Mask;

    #[test]
    fn mask_bitand_all_as_bit_and() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, true, true, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![false, true, false, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![1, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_all_true() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![false, false, true, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![true, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![2, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_true() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, false, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![0, 4])
        );
    }

    #[test]
    fn mask_bitand_false() {
        let this = Mask::from_buffer(BitBuffer::from_iter(vec![true, false, false, true, true]));
        let mask = Mask::from_buffer(BitBuffer::from_iter(vec![false, false, false]));
        assert_eq!(this.intersect_by_rank(&mask), Mask::from_indices(5, vec![]));
    }

    #[test]
    fn mask_intersect_by_rank_all_false() {
        let this = Mask::AllFalse(10);
        let mask = Mask::AllFalse(0);
        assert_eq!(this.intersect_by_rank(&mask), Mask::AllFalse(10));
    }

    #[rstest]
    #[case::all_true_with_all_true(Mask::new_true(5), Mask::new_true(5), vec![0, 1, 2, 3, 4])]
    #[case::all_true_with_all_false(Mask::new_true(5), Mask::new_false(5), vec![])]
    #[case::all_false_with_any(Mask::new_false(10), Mask::new_true(0), vec![])]
    #[case::indices_with_all_true(
        Mask::from_indices(10, vec![2, 5, 7, 9]),
        Mask::new_true(4),
        vec![2, 5, 7, 9]
    )]
    #[case::indices_with_all_false(
        Mask::from_indices(10, vec![2, 5, 7, 9]),
        Mask::new_false(4),
        vec![]
    )]
    fn test_intersect_by_rank_special_cases(
        #[case] base_mask: Mask,
        #[case] rank_mask: Mask,
        #[case] expected_indices: Vec<usize>,
    ) {
        let result = base_mask.intersect_by_rank(&rank_mask);
        match result.indices() {
            crate::AllOr::All => assert_eq!(expected_indices.len(), result.len()),
            crate::AllOr::None => assert!(expected_indices.is_empty()),
            crate::AllOr::Some(indices) => assert_eq!(indices, &expected_indices[..]),
        }
    }

    #[test]
    fn test_intersect_by_rank_example() {
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_intersect_by_rank_wrong_length() {
        let m1 = Mask::from_indices(10, vec![2, 5, 7]);
        let m2 = Mask::new_true(5);
        m1.intersect_by_rank(&m2);
    }

    #[rstest]
    #[case::single_element(vec![3], vec![true], vec![3])]
    #[case::single_element_masked(vec![3], vec![false], vec![])]
    #[case::alternating(vec![0, 2, 4, 6, 8], vec![true, false, true, false, true], vec![0, 4, 8])]
    #[case::consecutive(vec![5, 6, 7, 8, 9], vec![false, true, true, true, false], vec![6, 7, 8])]
    fn test_intersect_by_rank_patterns(
        #[case] base_indices: Vec<usize>,
        #[case] rank_pattern: Vec<bool>,
        #[case] expected_indices: Vec<usize>,
    ) {
        let base = Mask::from_indices(10, base_indices);
        let rank = Mask::from_iter(rank_pattern);
        let result = base.intersect_by_rank(&rank);

        match result.indices() {
            crate::AllOr::Some(indices) => assert_eq!(indices, &expected_indices[..]),
            crate::AllOr::None => assert!(expected_indices.is_empty()),
            _ => panic!("Unexpected result"),
        }
    }

    #[rstest]
    #[case::sparse_base_sparse_rank(0.1, 0.1)]
    #[case::sparse_base_dense_rank(0.1, 0.9)]
    #[case::dense_base_sparse_rank(0.5, 0.1)]
    #[case::dense_base_dense_rank(0.5, 0.9)]
    #[case::very_sparse(0.01, 0.5)]
    #[case::very_dense_rank(0.1, 0.99)]
    fn test_intersect_by_rank_densities(#[case] base_density: f64, #[case] rank_density: f64) {
        let base_len = 1000;
        let step = (1.0 / base_density).ceil() as usize;
        let base_indices: Vec<usize> = (0..base_len).step_by(step).collect();
        let base = Mask::from_indices(base_len, base_indices.clone());

        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (rank_density * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })));

        let result = base.intersect_by_rank(&rank);

        // Verify result correctness by checking each expected index
        let result_indices: Vec<usize> = match result.indices() {
            crate::AllOr::Some(indices) => indices.to_vec(),
            crate::AllOr::None => vec![],
            crate::AllOr::All => (0..result.len()).collect(),
        };

        let expected: Vec<usize> = base_indices
            .iter()
            .enumerate()
            .filter(|(rank_idx, _)| match rank.bit_buffer() {
                crate::AllOr::Some(buf) => unsafe { buf.value_unchecked(*rank_idx) },
                crate::AllOr::All => true,
                crate::AllOr::None => false,
            })
            .map(|(_, &idx)| idx)
            .collect();

        assert_eq!(result_indices, expected);
    }

    #[test]
    fn test_large_mask() {
        let base_len = 200;
        let base = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| i % 3 == 0)));
        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| i % 2 == 0)));

        let result = base.intersect_by_rank(&rank);
        assert!(result.true_count() > 0);
    }

    #[test]
    fn test_all_ones_chunk() {
        let base_len = 128;
        let base = Mask::new_true(base_len);
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..base_len).map(|i| i % 2 == 0)));

        let result = base.intersect_by_rank(&rank);
        assert_eq!(result.true_count(), 64);
    }

    #[test]
    fn test_consecutive_runs() {
        let base_len = 128;
        let base = Mask::from_buffer(BitBuffer::from_iter(
            (0..base_len).map(|i| (i < 10) || (30..60).contains(&i) || (100 <= i)),
        ));
        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| i % 3 != 0)));

        let result = base.intersect_by_rank(&rank);
        assert!(result.true_count() > 0);
    }

    #[test]
    fn test_pdep_portable() {
        use super::pdep_portable;

        assert_eq!(pdep_portable(0b11, 0b01010100), 0b00010100);
        assert_eq!(pdep_portable(u64::MAX, 0b10101010), 0b10101010);
        assert_eq!(pdep_portable(0, 0b11111111), 0);
        assert_eq!(pdep_portable(1, 0b00001000), 0b00001000);
        assert_eq!(pdep_portable(0, 0b00001000), 0);
    }

    #[test]
    fn test_extract_bits_from_chunks() {
        use super::extract_bits_from_chunks;

        let chunks = &[0xAAAAAAAAAAAAAAAAu64, 0x5555555555555555u64];
        let remainder = 0u64;

        assert_eq!(extract_bits_from_chunks(chunks, remainder, 0), chunks[0]);
        assert_eq!(extract_bits_from_chunks(chunks, remainder, 64), chunks[1]);

        let result = extract_bits_from_chunks(chunks, remainder, 32);
        let expected = (chunks[0] >> 32) | (chunks[1] << 32);
        assert_eq!(result, expected);
    }
}
