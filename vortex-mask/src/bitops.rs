// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_error::vortex_panic;

use crate::AllOr;
use crate::Mask;

impl BitAnd for &Mask {
    type Output = Mask;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) => rhs.clone(),
            (_, AllOr::All) => self.clone(),
            (AllOr::None, _) => Mask::new_false(self.len()),
            (_, AllOr::None) => Mask::new_false(self.len()),
            (AllOr::Some(lhs), AllOr::Some(rhs)) => Mask::from_buffer(lhs & rhs),
        }
    }
}

impl BitAnd<&Mask> for Mask {
    type Output = Mask;

    /// Owned-left AND: can reuse the left buffer in-place when possible.
    fn bitand(self, rhs: &Mask) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) => rhs.clone(),
            (AllOr::None, _) | (_, AllOr::None) => Mask::new_false(self.len()),
            (_, AllOr::All) => self,
            (AllOr::Some(_), AllOr::Some(rhs_buf)) => {
                Mask::from_buffer(self.into_bit_buffer() & rhs_buf)
            }
        }
    }
}

impl BitOr for &Mask {
    type Output = Mask;

    fn bitor(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) => Mask::new_true(self.len()),
            (_, AllOr::All) => Mask::new_true(self.len()),
            (AllOr::None, _) => rhs.clone(),
            (_, AllOr::None) => self.clone(),
            (AllOr::Some(lhs), AllOr::Some(rhs)) => Mask::from_buffer(lhs | rhs),
        }
    }
}

impl BitOr<&Mask> for Mask {
    type Output = Mask;

    /// Owned-left OR: can reuse the left buffer in-place when possible.
    fn bitor(self, rhs: &Mask) -> Self::Output {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }

        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::All, _) | (_, AllOr::All) => Mask::new_true(self.len()),
            (AllOr::None, _) => rhs.clone(),
            (_, AllOr::None) => self,
            (AllOr::Some(_), AllOr::Some(rhs_buf)) => {
                Mask::from_buffer(self.into_bit_buffer() | rhs_buf)
            }
        }
    }
}

impl Mask {
    /// Computes `self & !rhs` (AND NOT), equivalent to set difference.
    pub fn bitand_not(self, rhs: &Mask) -> Mask {
        if self.len() != rhs.len() {
            vortex_panic!("Masks must have the same length");
        }
        match (self.bit_buffer(), rhs.bit_buffer()) {
            (AllOr::None, _) | (_, AllOr::All) => Mask::new_false(self.len()),
            (_, AllOr::None) => self,
            (AllOr::All, _) => !rhs,
            (AllOr::Some(_), AllOr::Some(rhs_buf)) => {
                Mask::from_buffer(self.into_bit_buffer().into_bitand_not(rhs_buf))
            }
        }
    }
}

impl Not for Mask {
    type Output = Mask;

    fn not(self) -> Self::Output {
        !(&self)
    }
}

impl Not for &Mask {
    type Output = Mask;

    fn not(self) -> Self::Output {
        match self.bit_buffer() {
            AllOr::All => Mask::new_false(self.len()),
            AllOr::None => Mask::new_true(self.len()),
            AllOr::Some(buffer) => Mask::from_buffer(!buffer),
        }
    }
}

#[cfg(test)]
#[expect(clippy::many_single_char_names)]
mod tests {
    use vortex_buffer::BitBuffer;

    use super::*;

    #[test]
    fn test_bitand_all_combinations() {
        let len = 5;

        // Test AllTrue & AllTrue
        let all_true = Mask::new_true(len);
        let result = &all_true & &all_true;
        assert!(result.all_true());
        assert_eq!(result.true_count(), len);

        // Test AllTrue & AllFalse
        let all_false = Mask::new_false(len);
        let result = &all_true & &all_false;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);

        // Test AllFalse & AllTrue
        let result = &all_false & &all_true;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);

        // Test AllFalse & AllFalse
        let result = &all_false & &all_false;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);
    }

    #[test]
    fn test_bitand_with_values() {
        let mask1 = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let mask2 = Mask::from_buffer(BitBuffer::from_iter([true, true, false, false, true]));

        let result = &mask1 & &mask2;
        assert_eq!(result.len(), 5);
        assert_eq!(result.true_count(), 2);
        assert!(result.value(0)); // true & true = true
        assert!(!result.value(1)); // false & true = false
        assert!(!result.value(2)); // true & false = false
        assert!(!result.value(3)); // false & false = false
        assert!(result.value(4)); // true & true = true
    }

    #[test]
    fn test_bitand_all_true_with_values() {
        let all_true = Mask::new_true(5);
        let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));

        // AllTrue & Values should return Values
        let result = &all_true & &values;
        assert_eq!(result.true_count(), 3);
        assert_eq!(result.len(), 5);
        assert!(result.value(0));
        assert!(!result.value(1));
        assert!(result.value(2));
        assert!(!result.value(3));
        assert!(result.value(4));
    }

    #[test]
    fn test_bitand_all_false_with_values() {
        let all_false = Mask::new_false(5);
        let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));

        // AllFalse & Values should return AllFalse
        let result = &all_false & &values;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);
    }

    #[test]
    fn test_bitand_values_with_all_true() {
        let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let all_true = Mask::new_true(5);

        // Values & AllTrue should return Values
        let result = &values & &all_true;
        assert_eq!(result.true_count(), 3);
        assert_eq!(result.len(), 5);
        assert!(result.value(0));
        assert!(!result.value(1));
        assert!(result.value(2));
        assert!(!result.value(3));
        assert!(result.value(4));
    }

    #[test]
    fn test_bitand_values_with_all_false() {
        let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let all_false = Mask::new_false(5);

        // Values & AllFalse should return AllFalse
        let result = &values & &all_false;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);
    }

    #[test]
    fn test_bitand_empty_masks() {
        let empty1 = Mask::new_true(0);
        let empty2 = Mask::new_false(0);

        let result = &empty1 & &empty2;
        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
    }

    #[test]
    #[should_panic(expected = "Masks must have the same length")]
    fn test_bitand_different_lengths() {
        let mask1 = Mask::new_true(5);
        let mask2 = Mask::new_true(3);
        let _unused = &mask1 & &mask2;
    }

    #[test]
    fn test_not_all_true() {
        let all_true = Mask::new_true(5);
        let result = !&all_true;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_not_all_false() {
        let all_false = Mask::new_false(5);
        let result = !&all_false;
        assert!(result.all_true());
        assert_eq!(result.true_count(), 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_not_values() {
        let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let result = !&values;

        assert_eq!(result.len(), 5);
        assert_eq!(result.true_count(), 2);
        assert!(!result.value(0)); // !true = false
        assert!(result.value(1)); // !false = true
        assert!(!result.value(2)); // !true = false
        assert!(result.value(3)); // !false = true
        assert!(!result.value(4)); // !true = false
    }

    #[test]
    fn test_not_empty() {
        let empty_true = Mask::new_true(0);
        let result = !&empty_true;
        assert_eq!(result.len(), 0);
        assert!(result.is_empty());

        let empty_false = Mask::new_false(0);
        let result = !&empty_false;
        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_double_not() {
        let original = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let double_not = !&(!&original);

        // Double negation should return the original
        assert_eq!(double_not.true_count(), original.true_count());
        for i in 0..5 {
            assert_eq!(double_not.value(i), original.value(i));
        }
    }

    #[test]
    fn test_demorgan_law() {
        // Test De Morgan's law: !(A & B) = !A | !B
        let a = Mask::from_buffer(BitBuffer::from_iter([true, true, false, false]));
        let b = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));

        let and_result = &a & &b;
        let not_and = !&and_result;

        let not_a = !&a;
        let not_b = !&b;
        let or_result = &not_a | &not_b;

        assert_eq!(not_and.len(), 4);
        assert!(!not_and.value(0)); // !(true & true) = false
        assert!(not_and.value(1)); // !(true & false) = true
        assert!(not_and.value(2)); // !(false & true) = true
        assert!(not_and.value(3)); // !(false & false) = true

        assert_eq!(or_result.len(), 4);
        assert_eq!(or_result, not_and)
    }

    #[test]
    fn test_bitand_associativity() {
        // Test (A & B) & C = A & (B & C)
        let a = Mask::from_buffer(BitBuffer::from_iter([true, true, false, true]));
        let b = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true]));
        let c = Mask::from_buffer(BitBuffer::from_iter([false, true, true, true]));

        let left_assoc = &(&a & &b) & &c;
        let right_assoc = &a & &(&b & &c);

        assert_eq!(left_assoc.true_count(), right_assoc.true_count());
        for i in 0..4 {
            assert_eq!(left_assoc.value(i), right_assoc.value(i));
        }
    }

    #[test]
    fn test_bitand_commutativity() {
        // Test A & B = B & A
        let a = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));
        let b = Mask::from_buffer(BitBuffer::from_iter([false, true, false, true]));

        let a_and_b = &a & &b;
        let b_and_a = &b & &a;

        assert_eq!(a_and_b.true_count(), b_and_a.true_count());
        for i in 0..4 {
            assert_eq!(a_and_b.value(i), b_and_a.value(i));
        }
    }

    #[test]
    fn test_bitand_identity() {
        // Test A & AllTrue = A
        let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));
        let all_true = Mask::new_true(4);

        let result = &mask & &all_true;
        assert_eq!(result.true_count(), mask.true_count());
        for i in 0..4 {
            assert_eq!(result.value(i), mask.value(i));
        }
    }

    #[test]
    fn test_bitand_annihilator() {
        // Test A & AllFalse = AllFalse
        let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));
        let all_false = Mask::new_false(4);

        let result = &mask & &all_false;
        assert!(result.all_false());
        assert_eq!(result.true_count(), 0);
    }

    #[test]
    fn test_bitand_idempotence() {
        // Test A & A = A
        let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let result = &mask & &mask;

        assert_eq!(result.true_count(), mask.true_count());
        for i in 0..5 {
            assert_eq!(result.value(i), mask.value(i));
        }
    }

    #[test]
    fn test_complex_expression() {
        // Test a more complex expression: (!(!A) | B) & !C
        let a = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));
        let b = Mask::from_buffer(BitBuffer::from_iter([true, true, false, false]));
        let c = Mask::from_buffer(BitBuffer::from_iter([false, true, false, true]));

        let not_not_a = !(&(!&a));
        let not_not_a_or_b = &not_not_a | &b;
        let not_c = !&c;
        let result = &not_not_a_or_b & &not_c;

        // Verify the result manually
        assert!(result.value(0)); // (!(!true) | true) & !false = (true | true) & true = true
        assert!(!result.value(1)); // (!(!false) | true) & !true = (false | true) & false = false
        assert!(result.value(2)); // (!(!true) | false) & !false = (true | false) & true = true
        assert!(!result.value(3)); // (!(!false) | false) & !true = (false | false) & false = false
    }

    #[test]
    fn test_bitand_not() {
        let a = Mask::from_buffer(BitBuffer::from_iter([true, true, false, false]));
        let b = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));
        let result = a.clone().bitand_not(&b);
        assert!(!result.value(0)); // true & !true  = false
        assert!(result.value(1)); // true & !false = true
        assert!(!result.value(2)); // false & !true  = false
        assert!(!result.value(3)); // false & !false = false

        // bitand_not(All) = None
        assert!(a.clone().bitand_not(&Mask::new_true(4)).all_false());

        // bitand_not(None) = self
        let none = Mask::new_false(4);
        assert_eq!(a.clone().bitand_not(&none).true_count(), a.true_count());

        // None.bitand_not(_) = None
        assert!(none.bitand_not(&a).all_false());

        // All.bitand_not(x) = !x
        let not_b = !&b;
        let all_bitand_not_b = Mask::new_true(4).bitand_not(&b);
        for i in 0..4 {
            assert_eq!(all_bitand_not_b.value(i), not_b.value(i));
        }
    }

    #[test]
    fn test_bitor() {
        // Test basic OR operations
        let mask1 = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
        let mask2 = Mask::from_buffer(BitBuffer::from_iter([true, true, false, false, true]));

        let result = &mask1 | &mask2;
        assert_eq!(result.len(), 5);
        assert_eq!(result.true_count(), 4);
        assert!(result.value(0)); // true | true = true
        assert!(result.value(1)); // false | true = true
        assert!(result.value(2)); // true | false = true
        assert!(!result.value(3)); // false | false = false
        assert!(result.value(4)); // true | true = true

        // Test with AllTrue
        let all_true = Mask::new_true(5);
        let result = &mask1 | &all_true;
        assert!(result.all_true());

        // Test with AllFalse
        let all_false = Mask::new_false(5);
        let result = &mask1 | &all_false;
        assert_eq!(result.true_count(), mask1.true_count());
    }
}
