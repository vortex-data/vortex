// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::SCALAR_FN_SESSION;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::functions::NullHandling;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn is_valid(array: &ScalarFnArray, index: usize) -> bool {
        array.scalar_at(index).is_valid()
    }

    fn all_valid(array: &ScalarFnArray) -> bool {
        match array.scalar_fn.signature().null_handling() {
            NullHandling::Propagate | NullHandling::AbsorbsNull => {
                // Requires all children to guarantee all_valid
                array.children().iter().all(|child| child.all_valid())
            }
            NullHandling::Custom => {
                // We cannot guarantee that the array is all valid without evaluating the function
                false
            }
        }
    }

    fn all_invalid(array: &ScalarFnArray) -> bool {
        match array.scalar_fn.signature().null_handling() {
            NullHandling::Propagate => {
                // All null if any child is all null
                array.children().iter().any(|child| child.all_invalid())
            }
            NullHandling::AbsorbsNull | NullHandling::Custom => {
                // We cannot guarantee that the array is all valid without evaluating the function
                false
            }
        }
    }

    fn validity_mask(array: &ScalarFnArray) -> Mask {
        let vector = array
            .execute(&SCALAR_FN_SESSION)
            .vortex_expect("Validity mask computation should be fallible");
        Mask::from_buffer(vector.into_bool().into_parts().0)
    }
}
