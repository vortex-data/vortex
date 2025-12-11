// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Logical NOT operation.

use std::ops::Not;

use vortex_vector::BoolDatum;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;
use vortex_vector::bool::BoolVectorMut;

use crate::logical::LogicalNot;

impl LogicalNot for &BoolScalar {
    type Output = BoolScalar;

    fn not(self) -> BoolScalar {
        BoolScalar::new(self.value().map(|v| !v))
    }
}

impl LogicalNot for &BoolVector {
    type Output = BoolVector;

    fn not(self) -> <Self as LogicalNot>::Output {
        BoolVector::new(self.bits().not(), self.validity().clone())
    }
}

impl LogicalNot for &BoolDatum {
    type Output = BoolDatum;

    fn not(self) -> BoolDatum {
        match self {
            BoolDatum::Scalar(sc) => BoolDatum::Scalar(sc.not()),
            BoolDatum::Vector(vec) => BoolDatum::Vector(vec.not()),
        }
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
        // SAFETY: we did not change the length of capacity.
        unsafe { BoolVectorMut::new_unchecked(bits.not(), validity) }
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

    #[test]
    fn test_not_scalar() {
        let sc = BoolScalar::new(Some(true));
        assert_eq!((&sc).not().value(), Some(false));

        let sc = BoolScalar::new(Some(false));
        assert_eq!((&sc).not().value(), Some(true));

        let sc = BoolScalar::new(None);
        assert_eq!((&sc).not().value(), None);
    }

    #[test]
    fn test_not_datum_scalar() {
        let datum = BoolDatum::Scalar(BoolScalar::new(Some(true)));
        let result = datum.not();
        let BoolDatum::Scalar(sc) = result else {
            panic!("Expected Scalar");
        };
        assert_eq!(sc.value(), Some(false));
    }

    #[test]
    fn test_not_datum_vector() {
        let datum = BoolDatum::Vector(BoolVector::new(bitbuffer![1 0], Mask::new_true(2)));
        let result = datum.not();
        let BoolDatum::Vector(vec) = result else {
            panic!("Expected Vector");
        };
        assert_eq!(vec.bits(), &bitbuffer![0 1]);
    }
}
