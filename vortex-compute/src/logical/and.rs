// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, BitOr};

use vortex_vector::{BoolVector, VectorOps};

use crate::logical::LogicalAnd;

// TODO(ngates): should we try to into_mut and reuse the existing buffer? Let's benchmark.
impl LogicalAnd for &BoolVector {
    type Output = BoolVector;

    fn and(self, other: &BoolVector) -> BoolVector {
        BoolVector::new(
            self.bits().bitand(other.bits()),
            self.validity().bitor(other.validity()),
        )
    }
}
