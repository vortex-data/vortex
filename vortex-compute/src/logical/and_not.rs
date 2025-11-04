// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;

use crate::logical::LogicalAndNot;

// TODO(ngates): should we try to into_mut and reuse the existing buffer? Let's benchmark.
impl LogicalAndNot for &BoolVector {
    type Output = BoolVector;

    fn and_not(self, other: &BoolVector) -> BoolVector {
        BoolVector::new(
            self.bits().bitand_not(other.bits()),
            self.validity().bitand(other.validity()),
        )
    }
}

impl LogicalAndNot<&BoolVector> for BoolVector {
    type Output = BoolVector;

    fn and_not(self, other: &BoolVector) -> BoolVector {
        (&self).and_not(other)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolVector;

    use super::*;

    #[test]
    fn test_and_not_basic() {
        // left AND (NOT right)
        let left = BoolVector::new(bitbuffer![1 1 0 0], Mask::new_true(4));
        let right = BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4));

        let result = left.and_not(&right);
        // 1 & !1 = 0, 1 & !0 = 1, 0 & !1 = 0, 0 & !0 = 0
        assert_eq!(result.bits(), &bitbuffer![0 1 0 0]);
    }

    #[test]
    fn test_and_not_all_true() {
        let left = BoolVector::new(bitbuffer![1 1], Mask::new_true(2));
        let right = BoolVector::new(bitbuffer![1 1], Mask::new_true(2));

        let result = left.and_not(&right);
        assert_eq!(result.bits(), &bitbuffer![0 0]);
    }
}
