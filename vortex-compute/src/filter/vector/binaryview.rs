use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::binaryview::{BinaryViewType, BinaryViewVector, BinaryViewVectorMut};

use crate::filter::{Filter, MaskIndices};

macro_rules! delegate_filter_impl {
    ($mask_ty:ty) => {
        impl<T: BinaryViewType> Filter<$mask_ty> for &BinaryViewVector<T> {
            type Output = BinaryViewVector<T>;

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                let views = self.views().filter(selection);
                let validity = self.validity().filter(selection);

                // SAFETY: we filter the views and validity using the same mask
                unsafe {
                    BinaryViewVector::<T>::new_unchecked(views, self.buffers().clone(), validity)
                }
            }
        }

        impl<T: BinaryViewType> Filter<$mask_ty> for &mut BinaryViewVectorMut<T> {
            type Output = ();

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                // SAFETY: views and validity filtered by the same mask will have
                //  same resultant length.
                unsafe {
                    self.views_mut().filter(selection);
                    self.validity_mut().filter(selection);
                }
            }
        }
    };
}

delegate_filter_impl!(Mask);
delegate_filter_impl!(MaskIndices<'_>);
