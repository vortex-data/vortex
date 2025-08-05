// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::selection::Selection;
use crate::experiment::view::{Canonical, ViewMut};
use std::mem::take;

impl<'a> ViewMut<'a> {
    pub fn flatten<C: Canonical>(&mut self) {
        assert_eq!(self.vtype, C::vtype(), "ViewMut::flatten: type mismatch");
        let selection = take(&mut self.selection);
        let len = match selection {
            Selection::Prefix { len } => len,
            Selection::Constant { element, len } => {
                let value = &self.as_ref::<C>()[element];
                self.as_mut::<C>()[0..len].fill(*value);
                len
            }
            Selection::Mask(mask) => {
                // FIXME(ngates): we need to deal also with iter_slices, and to use BMI where
                //  possible.
                let mut offset = 0;
                mask.as_view().iter_ones(|idx| {
                    unsafe {
                        // SAFETY: We assume that the elements are of type T and that the view is valid.
                        *self.as_mut::<C>().get_unchecked_mut(offset) =
                            *self.as_ref::<C>().get_unchecked(idx);
                        offset += 1;
                    }
                });
                offset
            }
        };
        self.selection = Selection::Prefix { len };
    }
}
