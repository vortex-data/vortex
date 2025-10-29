// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute function for masking the validity of vectors.

use std::ops::BitAnd;

use vortex_dtype::NativePType;
use vortex_mask::Mask;
use vortex_vector::{
    BoolVector, NullVector, PVector, PrimitiveVector, StructVector, VarBinType, VarBinVector,
    Vector, match_each_pvector, match_each_vector,
};

/// Trait for masking the validity of an array or vector.
pub trait MaskValidity {
    /// Masks the validity of the object using the provided mask.
    ///
    /// The output has its validity intersected with the given mask, resulting in a new validity
    /// with equal or fewer valid entries.
    fn mask_validity(self, mask: &Mask) -> Self;
}

impl MaskValidity for Vector {
    fn mask_validity(self, mask: &Mask) -> Self {
        match_each_vector!(self, |v| { MaskValidity::mask_validity(v, mask).into() })
    }
}

impl MaskValidity for NullVector {
    fn mask_validity(self, _mask: &Mask) -> Self {
        // Null vectors have no validity to mask; they are always fully null.
        self
    }
}

impl MaskValidity for BoolVector {
    fn mask_validity(self, mask: &Mask) -> Self {
        let (bits, validity) = self.into_parts();
        // SAFETY: we are preserving the original bits buffer and only modifying the validity.
        unsafe { Self::new_unchecked(bits, validity.bitand(mask)) }
    }
}

impl MaskValidity for PrimitiveVector {
    fn mask_validity(self, mask: &Mask) -> Self {
        match_each_pvector!(self, |v| { MaskValidity::mask_validity(v, mask).into() })
    }
}

impl<T: NativePType> MaskValidity for PVector<T> {
    fn mask_validity(self, mask: &Mask) -> Self {
        let (data, validity) = self.into_parts();
        // SAFETY: we are preserving the original data buffer and only modifying the validity.
        unsafe { Self::new_unchecked(data, validity.bitand(mask)) }
    }
}

impl<T: VarBinType> MaskValidity for VarBinVector<T> {
    fn mask_validity(self, mask: &Mask) -> Self {
        let (views, buffers, validity) = self.into_parts();
        // SAFETY: we are preserving the original views and buffers, only modifying the validity.
        unsafe { Self::new_unchecked(views, buffers, validity.bitand(mask)) }
    }
}

impl MaskValidity for StructVector {
    fn mask_validity(self, mask: &Mask) -> Self {
        let (fields, validity) = self.into_parts();
        // SAFETY: we are preserving the original fields and only modifying the validity.
        unsafe { StructVector::new_unchecked(fields, validity.bitand(mask)) }
    }
}
