// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::validity_to_child;
use crate::arrays::Masked;
use crate::dtype::DType;
use crate::validity::Validity;

/// The underlying child array being masked.
pub(super) const CHILD_SLOT: usize = 0;
/// The validity bitmap defining which elements are non-null.
pub(super) const VALIDITY_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child", "validity"];

#[derive(Clone, Debug)]
pub struct MaskedData;

pub trait MaskedArrayExt {
    fn masked_data(&self) -> &MaskedData;
    fn masked_dtype(&self) -> &DType;
    fn masked_len(&self) -> usize;

    fn child(&self) -> &ArrayRef {
        self.as_slots()[CHILD_SLOT]
            .as_ref()
            .expect("validated masked child slot")
    }

    fn validity_child(&self) -> Option<&ArrayRef> {
        self.as_slots()[VALIDITY_SLOT].as_ref()
    }

    fn masked_validity(&self) -> Validity {
        match self.validity_child() {
            Some(validity) => Validity::Array(validity.clone()),
            None => Validity::AllValid,
        }
    }

    fn masked_validity_mask(&self) -> vortex_mask::Mask {
        self.masked_validity().to_mask(self.masked_len())
    }

    fn as_slots(&self) -> &[Option<ArrayRef>];
}

impl MaskedArrayExt for Array<Masked> {
    fn masked_data(&self) -> &MaskedData {
        self.data()
    }

    fn masked_dtype(&self) -> &DType {
        self.dtype()
    }

    fn masked_len(&self) -> usize {
        self.len()
    }

    fn as_slots(&self) -> &[Option<ArrayRef>] {
        self.slots()
    }
}

impl MaskedArrayExt for ArrayView<'_, Masked> {
    fn masked_data(&self) -> &MaskedData {
        self.data()
    }

    fn masked_dtype(&self) -> &DType {
        self.dtype()
    }

    fn masked_len(&self) -> usize {
        self.len()
    }

    fn as_slots(&self) -> &[Option<ArrayRef>] {
        self.slots()
    }
}

impl MaskedData {
    pub(crate) fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
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
        Ok(Self)
    }
}

impl Array<Masked> {
    /// Constructs a new `MaskedArray`.
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        let dtype = child.dtype().as_nullable();
        let len = child.len();
        let validity_slot = validity_to_child(&validity, len);
        let data = MaskedData::try_new(child.clone(), validity)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Masked, dtype, len, data)
                    .with_slots(vec![Some(child), validity_slot]),
            )
        })
    }
}
