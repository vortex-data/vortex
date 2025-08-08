// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{AllOr, Mask};

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
            (AllOr::None, _) => Self::new_false(0),
            (_, AllOr::None) => Self::new_false(self.len()),
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
}

#[cfg(test)]
#[allow(clippy::panic)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use rstest::rstest;

    use crate::Mask;

    #[test]
    fn mask_bitand_all_as_bit_and() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![true, true, true, true, true]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            false, true, false, true, true,
        ]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![1, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_all_true() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            false, false, true, true, true,
        ]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![true, true, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![2, 3, 4])
        );
    }

    #[test]
    fn mask_bitand_true() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            true, false, false, true, true,
        ]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![true, false, true]));
        assert_eq!(
            this.intersect_by_rank(&mask),
            Mask::from_indices(5, vec![0, 4])
        );
    }

    #[test]
    fn mask_bitand_false() {
        let this = Mask::from_buffer(BooleanBuffer::from_iter(vec![
            true, false, false, true, true,
        ]));
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(vec![false, false, false]));
        assert_eq!(this.intersect_by_rank(&mask), Mask::from_indices(5, vec![]));
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
        let _ = m1.intersect_by_rank(&m2);
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
}
