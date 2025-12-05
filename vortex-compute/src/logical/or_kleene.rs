// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;

use crate::logical::KleeneOr;
use crate::logical::LogicalOp;
use crate::logical::LogicalOrKleene;

impl LogicalOp<KleeneOr> for &BoolScalar {
    type Output = BoolScalar;

    fn op(self, rhs: &BoolScalar) -> BoolScalar {
        let result = match (self.value(), rhs.value()) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        };
        BoolScalar::new(result)
    }
}

impl LogicalOrKleene<&BoolScalar> for BoolScalar {
    type Output = BoolScalar;

    fn or_kleene(self, rhs: &BoolScalar) -> BoolScalar {
        <&Self as LogicalOp<KleeneOr>>::op(&self, rhs)
    }
}

impl LogicalOp<KleeneOr> for &BoolVector {
    type Output = BoolVector;

    fn op(self, rhs: &BoolVector) -> BoolVector {
        match (self.validity(), rhs.validity()) {
            (Mask::AllTrue(_), Mask::AllTrue(_)) => {
                BoolVector::new(self.bits().bitor(rhs.bits()), Mask::new_true(self.len()))
            }
            (Mask::AllTrue(_), Mask::AllFalse(_)) => {
                // self valid, rhs all null
                // Result: true where self is true (valid), null where self is false
                let result_bits = BitBuffer::new_set(self.len());
                let validity = self.bits().clone(); // valid where self is true
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllFalse(_), Mask::AllTrue(_)) => {
                // self all null, rhs valid
                // Result: true where rhs is true (valid), null where rhs is false
                let result_bits = BitBuffer::new_set(self.len());
                let validity = rhs.bits().clone(); // valid where rhs is true
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllFalse(_), Mask::AllFalse(_)) => {
                // All values are null
                BoolVector::new(
                    BitBuffer::new_unset(self.len()),
                    Mask::new_false(self.len()),
                )
            }
            (Mask::Values(lv), Mask::AllTrue(_)) => {
                // self partial validity, rhs all valid
                // Result valid where self valid OR self is null but rhs is true
                let result_bits = self.bits().bitor(rhs.bits());
                let validity = lv.bit_buffer().bitor(rhs.bits());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllTrue(_), Mask::Values(rv)) => {
                // self all valid, rhs partial validity
                // Result valid where rhs valid OR rhs is null but self is true
                let result_bits = self.bits().bitor(rhs.bits());
                let validity = rv.bit_buffer().bitor(self.bits());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::Values(lv), Mask::AllFalse(_)) => {
                // self partial validity, rhs all null
                // Result: true where self is true (valid), null otherwise
                let result_bits = BitBuffer::new_set(self.len());
                let validity = lv.bit_buffer().bitand(self.bits());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllFalse(_), Mask::Values(rv)) => {
                // self all null, rhs partial validity
                // Result: true where rhs is true (valid), null otherwise
                let result_bits = BitBuffer::new_set(self.len());
                let validity = rv.bit_buffer().bitand(rhs.bits());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::Values(lv), Mask::Values(rv)) => {
                // Both have partial validity
                // Result is valid where:
                // 1. Both are valid, OR
                // 2. One is null but the other is true (and valid)
                let result_bits = self.bits().bitor(rhs.bits());

                let both_valid = lv.bit_buffer().bitand(rv.bit_buffer());
                let self_null_rhs_true = lv
                    .bit_buffer()
                    .not()
                    .bitand(rv.bit_buffer())
                    .bitand(rhs.bits());
                let rhs_null_self_true = rv
                    .bit_buffer()
                    .not()
                    .bitand(lv.bit_buffer())
                    .bitand(self.bits());

                let validity = both_valid
                    .bitor(&self_null_rhs_true)
                    .bitor(&rhs_null_self_true);
                BoolVector::new(result_bits, Mask::from(validity))
            }
        }
    }
}

impl LogicalOrKleene<&BoolVector> for BoolVector {
    type Output = BoolVector;

    fn or_kleene(self, rhs: &BoolVector) -> BoolVector {
        <&Self as LogicalOp<KleeneOr>>::op(&self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolScalar;
    use vortex_vector::bool::BoolVector;

    use super::*;

    #[test]
    fn test_or_kleene_all_valid() {
        // When both sides are all valid, behaves like regular OR
        let left = BoolVector::new(bitbuffer![1 0 0], Mask::new_true(3));
        let right = BoolVector::new(bitbuffer![0 1 0], Mask::new_true(3));

        let result = left.or_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![1 1 0]);
        assert_eq!(result.validity(), &Mask::new_true(3));
    }

    #[test]
    fn test_or_kleene_all_null() {
        // When both are null, result is all null
        let left = BoolVector::new(bitbuffer![0 0], Mask::new_false(2));
        let right = BoolVector::new(bitbuffer![0 0], Mask::new_false(2));

        let result = left.or_kleene(&right);
        assert_eq!(result.validity(), &Mask::new_false(2));
    }

    #[test]
    fn test_or_kleene_true_and_null() {
        // true OR null = true (Kleene logic)
        let left = BoolVector::new(bitbuffer![1], Mask::new_true(1));
        let right = BoolVector::new(bitbuffer![0], Mask::new_false(1));

        let result = left.or_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![1]);
        // Result should be valid because true OR anything is true
        assert_eq!(result.validity(), &Mask::new_true(1));
    }

    #[test]
    fn test_scalar_or_kleene_true_true() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(true));
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_true_false() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(false));
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_false_false() {
        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(Some(false));
        assert_eq!(left.or_kleene(&right).value(), Some(false));
    }

    #[test]
    fn test_scalar_or_kleene_true_null() {
        // true OR null = true (Kleene logic)
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(None);
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_null_true() {
        // null OR true = true (Kleene logic)
        let left = BoolScalar::new(None);
        let right = BoolScalar::new(Some(true));
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_false_null() {
        // false OR null = null
        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(None);
        assert_eq!(left.or_kleene(&right).value(), None);
    }

    #[test]
    fn test_scalar_or_kleene_null_null() {
        let left = BoolScalar::new(None);
        let right = BoolScalar::new(None);
        assert_eq!(left.or_kleene(&right).value(), None);
    }
}
