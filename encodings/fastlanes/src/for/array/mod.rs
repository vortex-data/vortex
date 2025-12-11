// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::stats::ArrayStats;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scalar::Scalar;

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
    pub(super) stats_set: ArrayStats,
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

        Ok(Self {
            encoded,
            reference,
            stats_set: Default::default(),
        })
    }

    pub(crate) unsafe fn new_unchecked(encoded: ArrayRef, reference: Scalar) -> Self {
        Self {
            encoded,
            reference,
            stats_set: Default::default(),
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
        &self.stats_set
    }
}
