// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayCommon;
use vortex_array::ArrayRef;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::stats::ArrayStats;

pub mod for_compress;
pub mod for_decompress;

/// Frame of Reference (FoR) encoded array.
///
/// This encoding stores values as offsets from a reference value, which can significantly reduce
/// storage requirements when values are clustered around a specific point.
#[derive(Clone, Debug)]
pub struct FoRArray {
    pub(super) encoded: ArrayRef,
    pub(super) reference: Scalar,
    pub(super) common: ArrayCommon,
}

impl FoRArray {
    pub(crate) unsafe fn new_unchecked(encoded: ArrayRef, reference: Scalar) -> Self {
        let len = encoded.len();
        let dtype = reference.dtype().clone();
        Self {
            encoded,
            reference,
            common: ArrayCommon::new(len, dtype),
        }
    }

    #[inline]
    pub(crate) fn stats_set(&self) -> &ArrayStats {
        self.common.stats()
    }
}

/// Extension trait for [`FoRArray`] methods.
pub trait FoRArrayExt {
    /// Returns the primitive type of this array.
    fn ptype(&self) -> PType;

    /// Returns a reference to the encoded child array.
    fn encoded(&self) -> &ArrayRef;

    /// Returns the reference scalar value.
    fn reference_scalar(&self) -> &Scalar;
}

impl FoRArrayExt for FoRArray {
    #[inline]
    fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    #[inline]
    fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }

    #[inline]
    fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }
}
