// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::arrays::Masked;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ArrayInner;

#[derive(Clone, Debug)]
pub struct MaskedData {
    pub(super) child: ArrayRef,
    pub(super) validity: Validity,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl MaskedData {
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

    /// Returns the dtype of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.child.len()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the validity of the array.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Returns the validity as a [`Mask`](vortex_mask::Mask).
    pub fn validity_mask(&self) -> vortex_mask::Mask {
        self.validity.to_mask(self.len())
    }

    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}

impl ArrayInner<Masked> {
    /// Constructs a new `MaskedArray`.
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        ArrayInner::try_from_data(MaskedData::try_new(child, validity)?)
    }
}
