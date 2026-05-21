// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::array_slots;
use crate::arrays::Masked;
use crate::validity::Validity;

#[array_slots(Masked)]
pub struct MaskedSlots {
    /// The underlying child array being masked.
    pub child: ArrayRef,
    /// The validity bitmap defining which elements are non-null.
    pub validity: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct MaskedData;

impl Display for MaskedData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait MaskedArrayExt: TypedArrayRef<Masked> + MaskedArraySlotsExt {
    fn masked_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[MaskedSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}
impl<T: TypedArrayRef<Masked>> MaskedArrayExt for T {}

impl MaskedData {
    pub(crate) fn try_new(
        child_len: usize,
        child_all_valid: bool,
        validity: Validity,
    ) -> VortexResult<Self> {
        if matches!(validity, Validity::NonNullable) {
            vortex_bail!("MaskedArray must have nullable validity, got {validity:?}")
        }

        if !child_all_valid {
            vortex_bail!("MaskedArray children must not have nulls");
        }

        if let Some(validity_len) = validity.maybe_len()
            && validity_len != child_len
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
        let data = MaskedData::try_new(
            len,
            child.all_valid(&mut LEGACY_SESSION.create_execution_ctx())?,
            validity,
        )?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Masked, dtype, len, data)
                    .with_slots(smallvec![Some(child), validity_slot]),
            )
        })
    }
}
