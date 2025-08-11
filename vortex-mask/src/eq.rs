// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Mask;

impl PartialEq for Mask {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.true_count() != other.true_count() {
            return false;
        }

        // TODO(ngates): we could compare by indices if density is low enough
        self.boolean_buffer() == other.boolean_buffer()
    }
}

impl Eq for Mask {}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;

    use crate::Mask;

    #[test]
    fn filter_mask_eq() {
        assert_eq!(
            Mask::new_true(5),
            Mask::from_buffer(BooleanBuffer::new_set(5))
        );
        assert_eq!(
            Mask::new_false(5),
            Mask::from_buffer(BooleanBuffer::new_unset(5))
        );
        assert_eq!(
            Mask::from_indices(5, vec![0, 2, 3]),
            Mask::from_slices(5, vec![(0, 1), (2, 4)])
        );
        assert_eq!(
            Mask::from_indices(5, vec![0, 2, 3]),
            Mask::from_buffer(BooleanBuffer::from_iter([true, false, true, true, false]))
        );
    }

    #[test]
    fn test_mask_eq_different_lengths() {
        let mask1 = Mask::new_true(5);
        let mask2 = Mask::new_true(3);
        assert_ne!(mask1, mask2);
    }

    #[test]
    fn test_mask_eq_different_true_counts() {
        let mask1 = Mask::from_buffer(BooleanBuffer::from_iter([true, true, false]));
        let mask2 = Mask::from_buffer(BooleanBuffer::from_iter([true, false, false]));
        assert_ne!(mask1, mask2);
    }

    #[test]
    fn test_mask_eq_same_count_different_positions() {
        let mask1 = Mask::from_buffer(BooleanBuffer::from_iter([true, false, false]));
        let mask2 = Mask::from_buffer(BooleanBuffer::from_iter([false, true, false]));
        assert_ne!(mask1, mask2);
    }

    #[test]
    fn test_mask_eq_all_variants() {
        // Test AllTrue == AllTrue
        let all_true1 = Mask::new_true(5);
        let all_true2 = Mask::new_true(5);
        assert_eq!(all_true1, all_true2);

        // Test AllFalse == AllFalse
        let all_false1 = Mask::new_false(5);
        let all_false2 = Mask::new_false(5);
        assert_eq!(all_false1, all_false2);

        // Test AllTrue != AllFalse
        assert_ne!(all_true1, all_false1);

        // Test Values == Values
        let values1 = Mask::from_buffer(BooleanBuffer::from_iter([true, false, true]));
        let values2 = Mask::from_buffer(BooleanBuffer::from_iter([true, false, true]));
        assert_eq!(values1, values2);

        // Test AllTrue != Values (even if all values are true)
        let all_true_values = Mask::from_buffer(BooleanBuffer::new_set(5));
        assert_eq!(all_true1, all_true_values); // They should be equal

        // Test AllFalse != Values (even if all values are false)
        let all_false_values = Mask::from_buffer(BooleanBuffer::new_unset(5));
        assert_eq!(all_false1, all_false_values); // They should be equal
    }

    #[test]
    fn test_mask_eq_reflexive() {
        // Test that a mask equals itself
        let mask = Mask::from_buffer(BooleanBuffer::from_iter([true, false, true, false, true]));
        assert_eq!(mask, mask);
    }

    #[test]
    fn test_mask_eq_symmetric() {
        // Test that if a == b then b == a
        let mask1 = Mask::from_indices(5, vec![0, 2, 4]);
        let mask2 = Mask::from_slices(5, vec![(0, 1), (2, 3), (4, 5)]);
        assert_eq!(mask1, mask2);
        assert_eq!(mask2, mask1);
    }

    #[test]
    fn test_mask_eq_transitive() {
        // Test that if a == b and b == c then a == c
        let mask1 = Mask::from_indices(5, vec![1, 3]);
        let mask2 = Mask::from_slices(5, vec![(1, 2), (3, 4)]);
        let mask3 = Mask::from_buffer(BooleanBuffer::from_iter([false, true, false, true, false]));

        assert_eq!(mask1, mask2);
        assert_eq!(mask2, mask3);
        assert_eq!(mask1, mask3);
    }

    #[test]
    fn test_mask_eq_empty() {
        // All empty masks become AllFalse regardless of input type
        let empty1 = Mask::new_true(0);
        let empty2 = Mask::new_false(0);
        let empty3 = Mask::from_buffer(BooleanBuffer::new_set(0));
        let empty4 = Mask::from_buffer(BooleanBuffer::new_unset(0));

        // All should be AllFalse(0) when created from buffer
        assert!(matches!(empty3, Mask::AllFalse(0)));
        assert!(matches!(empty4, Mask::AllFalse(0)));

        // new_true(0) is AllTrue(0), new_false(0) is AllFalse(0)
        assert!(matches!(empty1, Mask::AllTrue(0)));
        assert!(matches!(empty2, Mask::AllFalse(0)));
    }

    #[test]
    fn test_mask_eq_different_representations() {
        // Test that masks with the same logical values but different internal representations are equal
        let indices = vec![0, 1, 2, 5, 6, 9];
        let slices = vec![(0, 3), (5, 7), (9, 10)];
        let buffer = BooleanBuffer::from_iter([
            true, true, true, false, false, true, true, false, false, true,
        ]);

        let mask1 = Mask::from_indices(10, indices);
        let mask2 = Mask::from_slices(10, slices);
        let mask3 = Mask::from_buffer(buffer);

        assert_eq!(mask1, mask2);
        assert_eq!(mask2, mask3);
        assert_eq!(mask1, mask3);
    }
}
