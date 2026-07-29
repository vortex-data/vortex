// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub mod for_compress;
pub mod for_decompress;

#[array_slots(crate::FoR)]
pub struct FoRSlots {
    /// The encoded array with the frame-of-reference (minimum value) subtracted.
    #[slot(0)]
    pub encoded: ArrayRef,
}

/// Frame of Reference (FoR) encoded array.
///
/// This encoding stores values as offsets from a reference value, which can significantly reduce
/// storage requirements when values are clustered around a specific point.
#[derive(Clone, Debug)]
pub struct FoRData {
    pub(super) reference: Scalar,
}

pub trait FoRArrayExt: FoRArraySlotsExt {
    fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }
}

impl<T: TypedArrayRef<crate::FoR>> FoRArrayExt for T {}

impl Display for FoRData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "reference: {}", self.reference)
    }
}

impl FoRData {
    pub(crate) fn try_new(reference: Scalar) -> VortexResult<Self> {
        vortex_ensure!(!reference.is_null(), "Reference value cannot be null");
        vortex_ensure!(
            reference.dtype().is_int(),
            "FoR requires an integer reference dtype, got {}",
            reference.dtype()
        );
        Ok(Self { reference })
    }

    #[inline]
    pub fn ptype(&self) -> PType {
        self.reference.dtype().as_ptype()
    }
}
