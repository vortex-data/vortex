// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayCommon;
use crate::ArrayRef;
use crate::validity::Validity;

#[derive(Clone, Debug)]
pub struct MaskedArray {
    pub(super) child: ArrayRef,
    pub(super) validity: Validity,
    pub(super) common: ArrayCommon,
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
        let len = child.len();

        Ok(Self {
            child,
            validity,
            common: ArrayCommon::new(len, dtype),
        })
    }

    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}
