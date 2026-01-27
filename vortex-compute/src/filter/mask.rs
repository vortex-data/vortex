// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitView;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskMut;
use vortex_mask::MaskValues;

use crate::filter::Filter;

impl Filter<Mask> for &Mask {
    type Output = Mask;

    fn filter(self, selection_mask: &Mask) -> Mask {
        match (self, selection_mask) {
            (Mask::AllTrue(_), _) => Mask::AllTrue(selection_mask.true_count()),
            (Mask::AllFalse(_), _) => Mask::AllFalse(selection_mask.true_count()),

            (Mask::Values(_), Mask::AllTrue(_)) => self.clone(),
            (Mask::Values(_), Mask::AllFalse(_)) => Mask::new_true(0),
            (Mask::Values(_), Mask::Values(v2)) => self.filter(v2.as_ref()),
        }
    }
}

impl Filter<MaskValues> for &Mask {
    type Output = Mask;

    fn filter(self, mask_values: &MaskValues) -> Mask {
        assert_eq!(
            mask_values.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        match self {
            Mask::AllTrue(_) => Mask::AllTrue(mask_values.true_count()),
            Mask::AllFalse(_) => Mask::AllFalse(mask_values.true_count()),
            Mask::Values(v) => Mask::from(v.bit_buffer().filter(mask_values)),
        }
    }
}

impl Filter<[usize]> for &Mask {
    type Output = Mask;

    fn filter(self, indices: &[usize]) -> Mask {
        match self {
            Mask::AllTrue(_) => Mask::AllTrue(indices.len()),
            Mask::AllFalse(_) => Mask::AllFalse(indices.len()),
            Mask::Values(v) => Mask::from(v.bit_buffer().filter(indices)),
        }
    }
}

impl Filter<[(usize, usize)]> for &Mask {
    type Output = Mask;

    fn filter(self, slices: &[(usize, usize)]) -> Mask {
        let output_len: usize = slices.iter().map(|(start, end)| end - start).sum();
        match self {
            Mask::AllTrue(_) => Mask::AllTrue(output_len),
            Mask::AllFalse(_) => Mask::AllFalse(output_len),
            Mask::Values(v) => Mask::from(v.bit_buffer().filter(slices)),
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
        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(v) => self.filter(v.as_ref()),
        }
    }
}

impl Filter<MaskValues> for &mut MaskMut {
    type Output = ();

    fn filter(self, mask_values: &MaskValues) {
        assert_eq!(
            mask_values.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        // TODO(connor): There is definitely a better way to do this (in place).
        let filtered = self.clone().freeze().filter(mask_values).into_mut();
        *self = filtered;
    }
}

impl Filter<[usize]> for &mut MaskMut {
    type Output = ();

    fn filter(self, indices: &[usize]) {
        // TODO(connor): There is definitely a better way to do this (in place).
        let filtered = self.clone().freeze().filter(indices).into_mut();
        *self = filtered;
    }
}

impl Filter<[(usize, usize)]> for &mut MaskMut {
    type Output = ();

    fn filter(self, slices: &[(usize, usize)]) {
        // TODO(connor): There is definitely a better way to do this (in place).
        let filtered = self.clone().freeze().filter(slices).into_mut();
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
