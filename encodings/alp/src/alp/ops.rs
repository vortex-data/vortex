// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::{ALPArray, ALPFloat, ALPVTable, match_each_alp_float_ptype};

impl OperationsVTable<ALPVTable> for ALPVTable {
    fn slice(array: &ALPArray, start: usize, stop: usize) -> ArrayRef {
        ALPArray::new(
            array.encoded().slice(start, stop),
            array.exponents(),
            array.patches().and_then(|p| p.slice(start, stop)),
        )
        .into_array()
    }

    fn scalar_at(array: &ALPArray, index: usize) -> Scalar {
        if let Some(patches) = array.patches()
            && let Some(patch) = patches.get_patched(index)
        {
            return patch.cast(array.dtype()).vortex_expect("cast failure");
        }

        let encoded_val = array.encoded().scalar_at(index);

        match_each_alp_float_ptype!(array.ptype(), |T| {
            let encoded_val: <T as ALPFloat>::ALPInt = encoded_val
                .as_ref()
                .try_into()
                .vortex_expect("invalid ALPInt");
            Scalar::primitive(
                <T as ALPFloat>::decode_single(encoded_val, array.exponents()),
                array.dtype().nullability(),
            )
        })
    }
}
