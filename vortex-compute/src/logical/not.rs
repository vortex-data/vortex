// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_vector::{BoolVector, BoolVectorMut, VectorOps};

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
