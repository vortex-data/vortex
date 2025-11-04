// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, BitOr, Not};

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;

use crate::logical::LogicalAndKleene;

impl LogicalAndKleene for &BoolVector {
    type Output = BoolVector;

    fn and_kleene(self, rhs: Self) -> Self::Output {
        match (self.validity(), rhs.validity()) {
            (Mask::AllTrue(_), Mask::AllTrue(_)) => {
                BoolVector::new(self.bits().bitand(rhs.bits()), Mask::new_true(self.len()))
            }
            (Mask::AllTrue(_), Mask::AllFalse(_)) => {
                // self valid, rhs all null
                // Result: false where self is false (valid), null where self is true
                let result_bits = BitBuffer::new_unset(self.len());
                let validity = self.bits().not(); // valid where self is false
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllFalse(_), Mask::AllTrue(_)) => {
                // self all null, rhs valid
                // Result: false where rhs is false (valid), null where rhs is true
                let result_bits = BitBuffer::new_unset(self.len());
                let validity = rhs.bits().not(); // valid where rhs is false
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
                // Result valid where self valid OR self is null but rhs is false
                let result_bits = self.bits().bitand(rhs.bits());
                let validity = lv.bit_buffer().bitor(&rhs.bits().not());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllTrue(_), Mask::Values(rv)) => {
                // self all valid, rhs partial validity
                // Result valid where rhs valid OR rhs is null but self is false
                let result_bits = self.bits().bitand(rhs.bits());
                let validity = rv.bit_buffer().bitor(&self.bits().not());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::Values(lv), Mask::AllFalse(_)) => {
                // self partial validity, rhs all null
                // Result: false where self is false (valid), null otherwise
                let result_bits = BitBuffer::new_unset(self.len());
                let validity = lv.bit_buffer().bitand(&self.bits().not());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::AllFalse(_), Mask::Values(rv)) => {
                // self all null, rhs partial validity
                // Result: false where rhs is false (valid), null otherwise
                let result_bits = BitBuffer::new_unset(self.len());
                let validity = rv.bit_buffer().bitand(&rhs.bits().not());
                BoolVector::new(result_bits, Mask::from(validity))
            }
            (Mask::Values(lv), Mask::Values(rv)) => {
                // Both have partial validity
                // Result is valid where:
                // 1. Both are valid, OR
                // 2. One is null but the other is false (and valid)
                let result_bits = self.bits().bitand(rhs.bits());

                let both_valid = lv.bit_buffer().bitand(rv.bit_buffer());
                let self_null_rhs_false = lv
                    .bit_buffer()
                    .not()
                    .bitand(rv.bit_buffer())
                    .bitand(&rhs.bits().not());
                let rhs_null_self_false = rv
                    .bit_buffer()
                    .not()
                    .bitand(lv.bit_buffer())
                    .bitand(&self.bits().not());

                let validity = both_valid
                    .bitor(&self_null_rhs_false)
                    .bitor(&rhs_null_self_false);
                BoolVector::new(result_bits, Mask::from(validity))
            }
        }
    }
}

impl LogicalAndKleene<&BoolVector> for BoolVector {
    type Output = BoolVector;

    fn and_kleene(self, rhs: &BoolVector) -> Self::Output {
        (&self).and_kleene(rhs)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolVector;

    use super::*;

    #[test]
    fn test_and_kleene_all_valid() {
        // When both sides are all valid, behaves like regular AND
        let left = BoolVector::new(bitbuffer![1 0 1], Mask::new_true(3));
        let right = BoolVector::new(bitbuffer![1 1 0], Mask::new_true(3));

        let result = left.and_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![1 0 0]);
        assert_eq!(result.validity(), &Mask::new_true(3));
    }

    #[test]
    fn test_and_kleene_all_null() {
        // When both are null, result is all null
        let left = BoolVector::new(bitbuffer![1 1], Mask::new_false(2));
        let right = BoolVector::new(bitbuffer![1 1], Mask::new_false(2));

        let result = left.and_kleene(&right);
        assert_eq!(result.validity(), &Mask::new_false(2));
    }

    #[test]
    fn test_and_kleene_false_and_null() {
        // false AND null = false (Kleene logic)
        let left = BoolVector::new(bitbuffer![0], Mask::new_true(1));
        let right = BoolVector::new(bitbuffer![1], Mask::new_false(1));

        let result = left.and_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![0]);
        // Result should be valid because false AND anything is false
        assert_eq!(result.validity(), &Mask::new_true(1));
    }
}
