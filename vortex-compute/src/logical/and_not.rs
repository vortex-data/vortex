// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitOr;

use vortex_vector::{BoolVector, VectorOps};

use crate::logical::LogicalAndNot;

// TODO(ngates): should we try to into_mut and reuse the existing buffer? Let's benchmark.
impl LogicalAndNot for &BoolVector {
    type Output = BoolVector;

    fn and_not(self, other: &BoolVector) -> BoolVector {
        BoolVector::new(
            self.bits().bitand_not(other.bits()),
            self.validity().bitor(other.validity()),
        )
    }
}
