// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
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
    /// The underlying child array being masked. The child may itself contain nulls; the array's
    /// logical validity is the child's validity ANDed with the mask.
    #[slot(0)]
    pub child: ArrayRef,
    /// The validity bitmap masking out additional elements as null.
    #[slot(1)]
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
    /// The mask stored by this array, as a [`Validity`].
    ///
    /// This is only the mask slot: the array's logical validity is this mask ANDed with the
    /// child's own validity. Use `validity()` for the merged validity.
    fn masked_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[MaskedSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }

    /// Returns `true` if the child is trivially free of nulls, without executing any compute.
    ///
    /// A `false` return is conservative: a child with [`Validity::Array`] may still be all-valid.
    fn child_is_null_free(&self) -> VortexResult<bool> {
        Ok(matches!(
            self.child().validity()?,
            Validity::NonNullable | Validity::AllValid
        ))
    }
}
impl<T: TypedArrayRef<Masked>> MaskedArrayExt for T {}

impl MaskedData {
    pub(crate) fn try_new(child_len: usize, validity: Validity) -> VortexResult<Self> {
        if matches!(validity, Validity::NonNullable) {
            vortex_bail!("MaskedArray must have nullable validity, got {validity:?}")
        }

        if let Some(validity_len) = validity.maybe_len()
            && validity_len != child_len
        {
            vortex_bail!("Validity must be the same length as a MaskedArray's child");
        }

        // MaskedArray's nullability is determined solely by its validity, not the child's dtype.
        // The child may contain nulls; they are lazily merged (ANDed) with the mask.
        Ok(Self)
    }
}

impl Array<Masked> {
    /// Constructs a new `MaskedArray`.
    ///
    /// The child may contain nulls: the array's logical validity is the child's validity ANDed
    /// with `validity`. Serialization requires a null-free child, so such arrays must be
    /// normalized (executing the mask into the child) before writing.
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        let dtype = child.dtype().as_nullable();
        let len = child.len();
        let validity_slot = validity_to_child(&validity, len);
        let data = MaskedData::try_new(len, validity)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Masked, dtype, len, data)
                    .with_slots(smallvec![Some(child), validity_slot]),
            )
        })
    }
}
