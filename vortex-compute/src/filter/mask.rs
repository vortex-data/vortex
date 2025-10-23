// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::filter::Filter;

impl Filter for Mask {
    fn filter(&self, mask: &Mask) -> Self {
        assert_eq!(self.len(), mask.len());

        match (self, mask) {
            (Mask::AllTrue(_), _) => Mask::AllTrue(mask.true_count()),
            (Mask::AllFalse(_), _) => Mask::AllFalse(mask.true_count()),

            (Mask::Values(_), Mask::AllTrue(_)) => self.clone(),
            (Mask::Values(_), Mask::AllFalse(_)) => Self::new_true(0),
            (Mask::Values(v1), Mask::Values(_)) => Mask::from(v1.bit_buffer().filter(mask)),
        }
    }
}
