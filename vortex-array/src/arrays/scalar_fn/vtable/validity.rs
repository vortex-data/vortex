// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorOps;

use crate::Array;
use crate::LEGACY_SESSION;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::VectorExecutor;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn is_valid(array: &ScalarFnArray, index: usize) -> bool {
        array.scalar_at(index).is_valid()
    }

    fn all_valid(array: &ScalarFnArray) -> bool {
        match array.scalar_fn.signature().is_null_sensitive() {
            true => {
                // If the function is null sensitive, we cannot guarantee all valid without evaluating
                // the function
                false
            }
            false => {
                // If the function is not null sensitive, we can guarantee all valid if all children
                // are all valid
                array.children().iter().all(|child| child.all_valid())
            }
        }
    }

    fn all_invalid(array: &ScalarFnArray) -> bool {
        match array.scalar_fn.signature().is_null_sensitive() {
            true => {
                // If the function is null sensitive, we cannot guarantee all invalid without evaluating
                // the function
                false
            }
            false => {
                // If the function is not null sensitive, we can guarantee all invalid if any child
                // is all invalid
                array.children().iter().any(|child| child.all_invalid())
            }
        }
    }

    fn validity_mask(array: &ScalarFnArray) -> Mask {
        let datum = array
            .to_array()
            .execute_datum(&LEGACY_SESSION)
            .vortex_expect("Validity mask computation should be fallible");
        match datum {
            Datum::Scalar(s) => Mask::new(array.len, s.is_valid()),
            Datum::Vector(v) => v.validity().clone(),
        }
    }
}
