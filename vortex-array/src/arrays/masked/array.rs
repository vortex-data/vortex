// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Not;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Array;
use crate::ArrayRef;
use crate::compute::mask;
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

    pub fn mask(&self) -> &Validity {
        &self.validity
    }

    pub(crate) fn masked_child(&self) -> VortexResult<ArrayRef> {
        let intersected_validity = self
            .child
            .validity_mask()
            .bitand(&self.validity.to_mask(self.len()));
        // Note: the compute function takes the validity inverted!!
        mask(&self.child, &intersected_validity.not())
    }
}
