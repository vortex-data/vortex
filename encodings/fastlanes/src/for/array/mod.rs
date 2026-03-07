// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayCommon;
use vortex_array::ArrayRef;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::stats::ArrayStats;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

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
    pub fn try_new(encoded: ArrayRef, reference: Scalar) -> VortexResult<Self> {
        if reference.is_null() {
            vortex_bail!("Reference value cannot be null");
        }
        let reference = reference.cast(
            &reference
                .dtype()
                .with_nullability(encoded.dtype().nullability()),
        )?;

        let len = encoded.len();
        let dtype = reference.dtype().clone();
        Ok(Self {
            encoded,
            reference,
            common: ArrayCommon::new(len, dtype),
        })
    }

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
    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    #[inline]
    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }

    #[inline]
    pub fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }

    #[inline]
    pub(crate) fn stats_set(&self) -> &ArrayStats {
        self.common.stats()
    }
}
