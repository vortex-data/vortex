// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitOr;

use vortex_vector::{BoolVector, VectorOps};

use crate::logical::LogicalOr;

// TODO(ngates): should we try to into_mut and reuse the existing buffer? Let's benchmark.
impl LogicalOr for &BoolVector {
    type Output = BoolVector;

    fn or(self, other: &BoolVector) -> BoolVector {
        BoolVector::new(
            self.bits().bitor(other.bits()),
            self.validity().bitor(other.validity()),
        )
    }
}
