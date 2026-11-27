// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayView;
use crate::IntoArray;
use crate::array::ValidityVTable;
use crate::arrays::patched::BoolPatchStrategy;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::validity::Validity;

impl ValidityVTable<Patched> for Patched {
    fn validity(array: ArrayView<'_, Self>) -> VortexResult<Validity> {
        let (PatchStrategy::Bool(BoolPatchStrategy::Replace)
        | PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let base = array.inner().validity()?;
        let replacements = array.patch_values().validity()?;
        match (base, replacements) {
            (Validity::NonNullable, Validity::NonNullable) => Ok(Validity::NonNullable),
            (Validity::AllValid, Validity::AllValid) => Ok(Validity::AllValid),
            (Validity::AllInvalid, Validity::AllInvalid) => Ok(Validity::AllInvalid),
            (base, replacements) => Ok(Validity::Array(
                Patched::from_relative(
                    base.to_array(array.len()),
                    array.patch_indices().clone(),
                    replacements.to_array(array.patch_values().len()),
                    array.patch_chunk_offsets().clone(),
                    array.offset,
                    PatchStrategy::Bool(BoolPatchStrategy::Replace),
                )?
                .into_array(),
            )),
        }
    }
}
