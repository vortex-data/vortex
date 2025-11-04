// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_vector::VectorOps;
use vortex_vector::bool::{BoolVector, BoolVectorMut};

use crate::logical::LogicalNot;

impl LogicalNot for &BoolVector {
    type Output = BoolVector;

    fn not(self) -> <Self as LogicalNot>::Output {
        BoolVector::new(self.bits().not(), self.validity().clone())
    }
}

impl LogicalNot for BoolVector {
    type Output = BoolVector;

    fn not(self) -> <Self as LogicalNot>::Output {
        // Attempt to re-use the underlying buffer if possible
        let (bits, validity) = self.into_parts();
        let bits = match bits.try_into_mut() {
            Ok(bits) => bits.not().freeze(),
            Err(bits) => (&bits).not(),
        };
        BoolVector::new(bits, validity)
    }
}

impl LogicalNot for BoolVectorMut {
    type Output = BoolVectorMut;

    fn not(self) -> <Self as LogicalNot>::Output {
        let (bits, validity) = self.into_parts();
        // SAFETY: we did not change the length of capacity
        unsafe { BoolVectorMut::new_unchecked(bits.not(), validity) }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolVector;

    use super::*;

    #[test]
    fn test_not_basic() {
        let vec = BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4));

        let result = vec.not();
        assert_eq!(result.bits(), &bitbuffer![0 1 0 1]);
        assert_eq!(result.validity(), &Mask::new_true(4));
    }

    #[test]
    fn test_not_owned() {
        let vec = BoolVector::new(bitbuffer![1 1], Mask::new_true(2));

        let result = vec.not();
        assert_eq!(result.bits(), &bitbuffer![0 0]);
    }
}
