use std::sync::Arc;

use vortex_mask::Mask;
use vortex_vector::struct_::{StructVector, StructVectorMut};
use vortex_vector::{Vector, VectorOps};

use crate::filter::{Filter, MaskIndices};

macro_rules! delegate_filter_impl {
    ($mask_ty:ty) => {
        impl Filter<$mask_ty> for &StructVector {
            type Output = StructVector;

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                let fields: Vec<Vector> = self
                    .fields()
                    .iter()
                    .map(|field| field.filter(selection))
                    .collect();

                let fields = Arc::new(fields.into_boxed_slice());
                let validity = self.validity().filter(selection);

                // SAFETY: all field vectors and validity are filtered with same mask
                unsafe { StructVector::new_unchecked(fields, validity) }
            }
        }

        impl Filter<$mask_ty> for &mut StructVectorMut {
            type Output = ();

            fn filter(self, selection: &$mask_ty) -> Self::Output {
                // SAFETY: all field vectors and selection vector are filtered with same mask
                unsafe {
                    for field in self.fields_mut() {
                        field.filter(selection);
                    }

                    self.validity_mut().filter(selection);
                }
            }
        }
    };
}

delegate_filter_impl!(Mask);
delegate_filter_impl!(MaskIndices<'_>);
