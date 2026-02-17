// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPFloat;
use crate::ALPVTable;
use crate::match_each_alp_float_ptype;

impl OperationsVTable<ALPVTable> for ALPVTable {
    fn scalar_at(array: &ALPArray, index: usize) -> VortexResult<Scalar> {
        if let Some(patches) = array.patches()
            && let Some(patch) = patches.get_patched(index)?
        {
            return patch.cast(array.dtype());
        }

        let encoded_val = array.encoded().scalar_at(index)?;

        Ok(match_each_alp_float_ptype!(array.ptype(), |T| {
            let encoded_val: <T as ALPFloat>::ALPInt =
                (&encoded_val).try_into().vortex_expect("invalid ALPInt");
            Scalar::primitive(
                <T as ALPFloat>::decode_single(encoded_val, array.exponents()),
                array.dtype().nullability(),
            )
        }))
    }
}
