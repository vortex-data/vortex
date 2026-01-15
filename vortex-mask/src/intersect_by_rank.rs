// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;

use crate::AllOr;
use crate::Mask;

impl Mask {
    /// Take the intersection of the `mask` with the set of true values in `self`.
    ///
    /// We are more interested in low selectivity `self` (as indices) with a boolean buffer mask,
    /// so we don't optimize for other cases, yet.
    ///
    /// Note: we might be able to accelerate this function on x86 with BMI, see:
    /// <https://www.microsoft.com/en-us/research/uploads/prod/2023/06/parquet-select-sigmod23.pdf>
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

        match (self.indices(), mask.indices()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_indices), AllOr::Some(mask_indices)) => {
                Self::from_indices(
                    self.len(),
                    mask_indices
                        .iter()
                        .map(|idx|
                            // This is verified as safe because we know that the indices are less than the
                            // mask.len() and we known mask.len() <= self.len(),
                            // implied by `self.true_count() == mask.len()`.
                            unsafe{*self_indices.get_unchecked(*idx)})
                        .collect(),
                )
            }
        }
    }

    /// Alternative implementation using BitBuffers directly without materializing indices.
    ///
    /// This approach iterates through `self`'s bit buffer and checks each position's
    /// rank against the mask's bit buffer. It avoids creating intermediate index vectors.
    ///
    /// Trade-offs vs index-based approach:
    /// - **Pro**: No index vector allocation, better for high-density masks
    /// - **Con**: Always O(self.len()) iterations regardless of density
    pub fn intersect_by_rank_bitbuffer(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let mut result = BitBufferMut::new_unset(self.len());
                let mut rank = 0usize;

                for (i, self_bit) in self_buffer.iter().enumerate() {
                    if self_bit {
                        // SAFETY: rank < mask.len() because we increment rank only when
                        // self_bit is true, and self.true_count() == mask.len()
                        if unsafe { mask_buffer.value_unchecked(rank) } {
                            result.set(i);
                        }
                        rank += 1;
                    }
                }

                Self::from_buffer(result.freeze())
            }
        }
    }

    /// BitBuffer implementation processing u64 chunks at a time.
    ///
    /// This builds output u64s directly instead of setting individual bits,
    /// which is faster for dense masks.
    pub fn intersect_by_rank_u64(&self, mask: &Mask) -> Mask {
        assert_eq!(self.true_count(), mask.len());

        match (self.bit_buffer(), mask.bit_buffer()) {
            (AllOr::All, _) => mask.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Self::new_false(self.len()),

            (AllOr::Some(self_buffer), AllOr::Some(mask_buffer)) => {
                let len = self.len();
                let mut buffer: BufferMut<u64> = BufferMut::with_capacity(len.div_ceil(64));
                let mut rank = 0usize;

                let self_chunks = self_buffer.chunks();

                // Process full 64-bit chunks
                for self_chunk in self_chunks.iter() {
                    let mut result_chunk = 0u64;

                    // Process each bit in the chunk
                    for bit_idx in 0..64 {
                        let self_bit = (self_chunk >> bit_idx) & 1 == 1;
                        if self_bit {
                            // SAFETY: rank < mask.len() because self.true_count() == mask.len()
                            if unsafe { mask_buffer.value_unchecked(rank) } {
                                result_chunk |= 1u64 << bit_idx;
                            }
                            rank += 1;
                        }
                    }

                    // SAFETY: we allocated enough capacity
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                // Handle remainder bits
                let remainder = len % 64;
                if remainder != 0 {
                    let self_chunk = self_chunks.remainder_bits();
                    let mut result_chunk = 0u64;

                    for bit_idx in 0..remainder {
                        let self_bit = (self_chunk >> bit_idx) & 1 == 1;
                        if self_bit {
                            // SAFETY: rank < mask.len()
                            if unsafe { mask_buffer.value_unchecked(rank) } {
                                result_chunk |= 1u64 << bit_idx;
                            }
                            rank += 1;
                        }
                    }

                    // SAFETY: we allocated enough capacity
                    unsafe { buffer.push_unchecked(result_chunk) };
                }

                // Truncate to correct byte length
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
    use vortex_error::VortexResult;

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
    #[case::all_true_with_all_true(
        Mask::new_true(5),
        Mask::new_true(5),
        vec![0, 1, 2, 3, 4]
    )]
    #[case::all_true_with_all_false(
        Mask::new_true(5),
        Mask::new_false(5),
        vec![]
    )]
    #[case::all_false_with_any(
        Mask::new_false(10),
        Mask::new_true(0),
        vec![]
    )]
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
        // Example from the documentation
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_intersect_by_rank_wrong_length() {
        let m1 = Mask::from_indices(10, vec![2, 5, 7]); // 3 true values
        let m2 = Mask::new_true(5); // 5 true values - doesn't match
        m1.intersect_by_rank(&m2);
    }

    #[rstest]
    #[case::single_element(
        vec![3],
        vec![true],
        vec![3]
    )]
    #[case::single_element_masked(
        vec![3],
        vec![false],
        vec![]
    )]
    #[case::alternating(
        vec![0, 2, 4, 6, 8],
        vec![true, false, true, false, true],
        vec![0, 4, 8]
    )]
    #[case::consecutive(
        vec![5, 6, 7, 8, 9],
        vec![false, true, true, true, false],
        vec![6, 7, 8]
    )]
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

    // Tests for BitBuffer-based implementation

    #[rstest]
    #[case::sparse_base_sparse_rank(0.1, 0.1)]
    #[case::sparse_base_dense_rank(0.1, 0.9)]
    #[case::dense_base_sparse_rank(0.5, 0.1)]
    #[case::dense_base_dense_rank(0.5, 0.9)]
    #[case::very_sparse(0.01, 0.5)]
    #[case::very_dense_rank(0.1, 0.99)]
    fn test_bitbuffer_impl_matches_indices_impl(
        #[case] base_density: f64,
        #[case] rank_density: f64,
    ) -> VortexResult<()> {
        let base_len = 1000;
        let step = (1.0 / base_density).ceil() as usize;
        let base_indices: Vec<usize> = (0..base_len).step_by(step).collect();
        let base = Mask::from_indices(base_len, base_indices);

        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (rank_density * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })));

        let result_indices = base.intersect_by_rank(&rank);
        let result_bitbuffer = base.intersect_by_rank_bitbuffer(&rank);

        assert_eq!(result_indices, result_bitbuffer);
        Ok(())
    }

    #[test]
    fn test_bitbuffer_impl_example() {
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank_bitbuffer(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bitbuffer_impl_fast_paths() {
        // AllTrue base
        let base = Mask::new_true(5);
        let rank = Mask::from_iter([true, false, true, false, true]);
        assert_eq!(
            base.intersect_by_rank_bitbuffer(&rank),
            base.intersect_by_rank(&rank)
        );

        // AllTrue rank
        let base = Mask::from_iter([true, false, true, false, true]);
        let rank = Mask::new_true(3);
        assert_eq!(
            base.intersect_by_rank_bitbuffer(&rank),
            base.intersect_by_rank(&rank)
        );

        // AllFalse base
        let base = Mask::new_false(5);
        let rank = Mask::new_true(0);
        assert_eq!(
            base.intersect_by_rank_bitbuffer(&rank),
            base.intersect_by_rank(&rank)
        );

        // AllFalse rank
        let base = Mask::from_iter([true, false, true, false, true]);
        let rank = Mask::new_false(3);
        assert_eq!(
            base.intersect_by_rank_bitbuffer(&rank),
            base.intersect_by_rank(&rank)
        );
    }

    // Tests for u64-based implementation

    #[rstest]
    #[case::sparse_base_sparse_rank(0.1, 0.1)]
    #[case::sparse_base_dense_rank(0.1, 0.9)]
    #[case::dense_base_sparse_rank(0.5, 0.1)]
    #[case::dense_base_dense_rank(0.5, 0.9)]
    #[case::very_sparse(0.01, 0.5)]
    #[case::very_dense_rank(0.1, 0.99)]
    fn test_u64_impl_matches_indices_impl(
        #[case] base_density: f64,
        #[case] rank_density: f64,
    ) -> VortexResult<()> {
        let base_len = 1000;
        let step = (1.0 / base_density).ceil() as usize;
        let base_indices: Vec<usize> = (0..base_len).step_by(step).collect();
        let base = Mask::from_indices(base_len, base_indices);

        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter((0..rank_len).map(|i| {
            let threshold = (rank_density * 1000.0) as usize;
            (i * 7 + 13) % 1000 < threshold
        })));

        let result_indices = base.intersect_by_rank(&rank);
        let result_u64 = base.intersect_by_rank_u64(&rank);

        assert_eq!(result_indices, result_u64);
        Ok(())
    }

    #[test]
    fn test_u64_impl_example() {
        let m1 = Mask::from_iter([true, false, false, true, true, true, false, true]);
        let m2 = Mask::from_iter([false, false, true, false, true]);
        let result = m1.intersect_by_rank_u64(&m2);
        let expected = Mask::from_iter([false, false, false, false, true, false, false, true]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_u64_impl_large() {
        // Test with more than 64 bits to ensure chunk handling is correct
        let base_len = 200;
        let base = Mask::from_buffer(BitBuffer::from_iter(
            (0..base_len).map(|i| i % 3 == 0), // every 3rd bit
        ));
        let rank_len = base.true_count();
        let rank = Mask::from_buffer(BitBuffer::from_iter(
            (0..rank_len).map(|i| i % 2 == 0), // every other
        ));

        let result_indices = base.intersect_by_rank(&rank);
        let result_u64 = base.intersect_by_rank_u64(&rank);

        assert_eq!(result_indices, result_u64);
    }
}
