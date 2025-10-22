// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::vectors::filter::Filter;
use vortex_mask::Mask;
use vortex_vector::{BoolVector, BoolVectorMut, VectorOps};

impl Filter for BoolVector {
    type Mutable = BoolVectorMut;

    fn filter(&self, mask: &Mask) -> Self {
        self.filter_into(mask, BoolVectorMut::with_capacity(0))
    }

    fn filter_into(&self, mask: &Mask, out: Self::Mutable) -> Self {
        let (bits_out, validity_out) = out.into_parts();
        Self::new(
            self.bits().filter_into(mask, bits_out),
            self.validity().filter_into(mask, validity_out),
        )
    }
}
