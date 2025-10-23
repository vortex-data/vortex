// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_vector::{BoolVector, VectorOps};

use crate::logical::LogicalNot;

// TODO(ngates): should we try to into_mut and reuse the existing buffer? Let's benchmark.
impl LogicalNot for &BoolVector {
    type Output = BoolVector;

    fn not(self) -> <Self as LogicalNot>::Output {
        BoolVector::new(self.bits().not(), self.validity().clone())
    }
}
