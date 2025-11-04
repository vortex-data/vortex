// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;

use crate::filter::Filter;

impl Filter for BoolVector {
    fn filter(&self, mask: &Mask) -> Self {
        let filtered_bits = self.bits().filter(mask);
        let filtered_validity = self.validity().filter(mask);

        // SAFETY: We filter the bits and validity with the same mask, and since they came from an
        // existing and valid `BoolVector`, we know that the filtered output must have the same
        // length.
        unsafe { Self::new_unchecked(filtered_bits, filtered_validity) }
    }
}
