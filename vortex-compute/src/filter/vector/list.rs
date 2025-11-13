use std::sync::Arc;

use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::listview::{ListViewVector, ListViewVectorMut};

use crate::filter::{Filter, MaskIndices};

macro_rules! delegate_filter_impl {
    ($mask_ty:ty) => {
        impl Filter<$mask_ty> for &ListViewVector {
            type Output = ListViewVector;

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                let offsets = self.offsets().filter(selection);
                let sizes = self.sizes().filter(selection);
                let validity = self.validity().filter(selection);

                // SAFETY: all components filtered with same mask
                unsafe {
                    ListViewVector::new_unchecked(
                        Arc::clone(self.elements()),
                        offsets,
                        sizes,
                        validity,
                    )
                }
            }
        }

        impl Filter<$mask_ty> for &mut ListViewVectorMut {
            type Output = ();

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                // SAFETY: offsets, sizes, validity all being filtered with same mask
                unsafe {
                    self.offsets_mut().filter(selection);
                    self.sizes_mut().filter(selection);
                    self.validity_mut().filter(selection);
                }
            }
        }
    };
}

delegate_filter_impl!(Mask);
delegate_filter_impl!(MaskIndices<'_>);
