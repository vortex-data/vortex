// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, BitOr};

use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;

use crate::logical::LogicalOr;

// TODO(ngates): should we try to into_mut and reuse the existing buffer? Let's benchmark.
impl LogicalOr for &BoolVector {
    type Output = BoolVector;

    fn or(self, other: &BoolVector) -> BoolVector {
        BoolVector::new(
            self.bits().bitor(other.bits()),
            self.validity().bitand(other.validity()),
        )
    }
}

impl LogicalOr<&BoolVector> for BoolVector {
    type Output = BoolVector;

    fn or(self, other: &BoolVector) -> BoolVector {
        (&self).or(other)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolVector;

    use super::*;

    #[test]
    fn test_or_basic() {
        let left = BoolVector::new(bitbuffer![1 1 0 0], Mask::new_true(4));
        let right = BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4));

        let result = left.or(&right);
        assert_eq!(result.bits(), &bitbuffer![1 1 1 0]);
    }

    #[test]
    fn test_or_with_nulls() {
        let left = BoolVector::new(bitbuffer![0 1], Mask::from(bitbuffer![0 1]));
        let right = BoolVector::new(bitbuffer![0 0], Mask::new_true(2));

        let result = left.or(&right);
        // Validity is AND'd, so if either side is null, result is null
        assert_eq!(result.validity(), &Mask::from(bitbuffer![0 1]));
    }
}
