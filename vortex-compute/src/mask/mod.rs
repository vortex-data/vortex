// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute function for masking the validity of vectors.

use std::ops::BitAnd;

use vortex_dtype::NativePType;
use vortex_mask::Mask;
use vortex_vector::{
    BoolVector, NullVector, PrimitiveVector, StructVector, Vector, match_each_pvector,
    match_each_vector,
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
        Self::new(bits, validity.bitand(mask))
    }
}

impl MaskValidity for PrimitiveVector {
    fn mask_validity(self, mask: &Mask) -> Self {
        match_each_pvector!(self, |v| { MaskValidity::mask_validity(v, mask).into() })
    }
}

impl<T: NativePType> MaskValidity for vortex_vector::PVector<T> {
    fn mask_validity(self, mask: &Mask) -> Self {
        let (data, validity) = self.into_parts();
        Self::new(data, validity.bitand(mask))
    }
}

impl MaskValidity for StructVector {
    fn mask_validity(self, mask: &Mask) -> Self {
        let (fields, validity) = self.into_parts();
        StructVector::new(fields, validity.bitand(mask))
    }
}
