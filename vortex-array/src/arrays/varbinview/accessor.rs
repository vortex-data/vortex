// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use crate::ToCanonical;
use crate::accessor::ArrayAccessor;
use crate::arrays::VarBinViewArray;
use crate::validity::Validity;

impl ArrayAccessor<[u8]> for VarBinViewArray {
    fn with_iterator<F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R, R>(
        &self,
        f: F,
    ) -> R {
        let bytes = (0..self.data_buffers().len())
            .map(|i| self.buffer(i))
            .collect::<Vec<_>>();

        let views = self.views();

        match self.validity() {
            Validity::NonNullable | Validity::AllValid => {
                let mut iter = views.iter().map(|view| {
                    if view.is_inlined() {
                        Some(view.as_inlined().value())
                    } else {
                        Some(
                            &bytes[view.as_view().buffer_index as usize][view.as_view().as_range()],
                        )
                    }
                });
                f(&mut iter)
            }
            Validity::AllInvalid => f(&mut iter::repeat_n(None, views.len())),
            Validity::Array(v) => {
                let validity = v.to_bool().into_bit_buffer();
                let mut iter = views.iter().zip(validity.iter()).map(|(view, valid)| {
                    if valid {
                        if view.is_inlined() {
                            Some(view.as_inlined().value())
                        } else {
                            Some(
                                &bytes[view.as_view().buffer_index as usize]
                                    [view.as_view().as_range()],
                            )
                        }
                    } else {
                        None
                    }
                });
                f(&mut iter)
            }
        }
    }
}

impl ArrayAccessor<[u8]> for &VarBinViewArray {
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R,
    {
        <VarBinViewArray as ArrayAccessor<[u8]>>::with_iterator(*self, f)
    }
}
