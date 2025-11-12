// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::{Mask, MaskMut};

use crate::filter::Filter;

impl Filter for &Mask {
    type Output = Mask;

    fn filter(self, selection_mask: &Mask) -> Mask {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        match (self, selection_mask) {
            (Mask::AllTrue(_), _) => Mask::AllTrue(selection_mask.true_count()),
            (Mask::AllFalse(_), _) => Mask::AllFalse(selection_mask.true_count()),

            (Mask::Values(_), Mask::AllTrue(_)) => self.clone(),
            (Mask::Values(_), Mask::AllFalse(_)) => Mask::new_true(0),
            (Mask::Values(v1), Mask::Values(_)) => {
                Mask::from(v1.bit_buffer().filter(selection_mask))
            }
        }
    }
}

impl Filter for &mut MaskMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        // TODO(connor): There is definitely a better way to do this (in place).
        let filtered = self.clone().freeze().filter(selection_mask).into_mut();
        *self = filtered;
    }
}
