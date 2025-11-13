use vortex_dtype::NativeDecimalType;
use vortex_mask::Mask;
use vortex_vector::VectorOps;
use vortex_vector::decimal::{DVector, DVectorMut};

use crate::filter::{Filter, MaskIndices};

macro_rules! delegate_filter_impl {
    ($mask_ty:ty) => {
        impl<D: NativeDecimalType> Filter<$mask_ty> for &DVector<D> {
            type Output = DVector<D>;

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                let elements = self.elements().filter(selection);
                let validity = self.validity().filter(selection);
                // SAFETY: we're filtering the elements and validity with the same mask
                unsafe { DVector::<D>::new_unchecked(self.precision_scale(), elements, validity) }
            }
        }

        impl<D: NativeDecimalType> Filter<$mask_ty> for &mut DVectorMut<D> {
            type Output = ();

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                // SAFETY: we filter elements and validity using the same mask
                unsafe {
                    self.elements_mut().filter(selection);
                    self.validity_mut().filter(selection);
                }
            }
        }
    };
}

delegate_filter_impl!(Mask);
delegate_filter_impl!(MaskIndices<'_>);
