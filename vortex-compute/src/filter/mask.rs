// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitView;
use vortex_error::VortexExpect;
use vortex_mask::{Mask, MaskMut};

use crate::filter::Filter;

impl Filter<Mask> for &Mask {
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

impl<const NB: usize> Filter<BitView<'_, NB>> for &Mask {
    type Output = Mask;

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        match self {
            Mask::AllTrue(_) => Mask::AllTrue(selection.true_count()),
            Mask::AllFalse(_) => Mask::AllFalse(selection.true_count()),
            Mask::Values(v) => Mask::from(v.bit_buffer().filter(selection)),
        }
    }
}

impl Filter<Mask> for &mut MaskMut {
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

impl<const NB: usize> Filter<BitView<'_, NB>> for &mut MaskMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        if self.all_true() {
            *self = MaskMut::new_true(selection.true_count());
            return;
        }
        if self.all_false() {
            *self = MaskMut::new_false(selection.true_count());
            return;
        }
        self.as_bit_buffer_mut()
            .vortex_expect("Checked all-true and all-false cases; should have bit buffer")
            .filter(selection);
    }
}
