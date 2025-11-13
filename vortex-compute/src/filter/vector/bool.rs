// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::bool::{BoolVector, BoolVectorMut};

use crate::filter::{Filter, MaskIndices};

macro_rules! delegate_filter_impl {
    ($mask_ty:ty) => {
        impl Filter<$mask_ty> for &BoolVector {
            type Output = BoolVector;

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                let filtered_bits = self.bits().filter(selection);
                let filtered_validity = self.validity().filter(selection);

                // SAFETY: We filter the bits and validity with the same mask, and since they came from an
                // existing and valid `BoolVector`, we know that the filtered output must have the same
                // length.
                unsafe { BoolVector::new_unchecked(filtered_bits, filtered_validity) }
            }
        }

        impl Filter<$mask_ty> for &mut BoolVectorMut {
            type Output = ();

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                // TODO(aduffy): how can we do this faster in-place?
                unsafe {
                    let bits = self.bits_mut();
                    *bits = (*bits).clone().freeze().filter(selection).into_mut();
                    self.validity_mut().filter(selection);
                }
            }
        }
    };
}

delegate_filter_impl!(Mask);
delegate_filter_impl!(MaskIndices<'_>);
