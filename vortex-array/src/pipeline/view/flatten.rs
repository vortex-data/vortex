// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::N;
use crate::pipeline::bits::BitView;
use crate::pipeline::types::Element;
use crate::pipeline::view::ViewMut;

impl<'a> ViewMut<'a> {
    /// Flatten the view using the provided mask.
    pub fn select_mask<E: Element>(&mut self, mask: &BitView) {
        assert_eq!(
            self.vtype,
            E::vtype(),
            "ViewMut::flatten_mask: type mismatch"
        );

        match mask.true_count() {
            0 => {
                // If the mask has no true bits, we set the length to 0.
                self.len = 0;
                return;
            }
            N => {
                // If the mask has N true bits, we copy all elements.
                self.len = N;
            }
            _ => {
                let mut offset = 0;
                mask.iter_ones(|idx| {
                    unsafe {
                        // SAFETY: We assume that the elements are of type E and that the view is valid.
                        *self.as_mut::<E>().get_unchecked_mut(offset) =
                            *self.as_ref::<E>().get_unchecked(idx);
                        offset += 1;
                    }
                });
                self.len = offset;
            }
        }
    }
}
