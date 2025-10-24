// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::{BoolVector, VectorOps};

use crate::filter::Filter;

impl Filter for BoolVector {
    fn filter(&self, mask: &Mask) -> Self {
        Self::new(self.bits().filter(mask), self.validity().filter(mask))
    }
}
