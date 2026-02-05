// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::stats::ArrayStats;
use crate::validity::Validity;

#[derive(Clone, Debug)]
pub struct MaskedArray {
    pub(super) child: ArrayRef,
    pub(super) validity: Validity,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl MaskedArray {
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        if matches!(validity, Validity::NonNullable) {
            vortex_bail!("MaskedArray must have nullable validity, got {validity:?}")
        }

        if !child.all_valid()? {
            vortex_bail!("MaskedArray children must not have nulls");
        }

        if let Some(validity_len) = validity.maybe_len()
            && validity_len != child.len()
        {
            vortex_bail!("Validity must be the same length as a MaskedArray's child");
        }

        // MaskedArray's nullability is determined solely by its validity, not the child's dtype.
        // The child can have nullable dtype but must not have any actual null values.
        let dtype = child.dtype().as_nullable();

        Ok(Self {
            child,
            validity,
            dtype,
            stats: ArrayStats::default(),
        })
    }

    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}
