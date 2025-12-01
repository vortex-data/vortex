// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::SCALAR_FN_SESSION;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::functions::NullHandling;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn is_valid(array: &ScalarFnArray, index: usize) -> VortexResult<bool> {
        Ok(array.scalar_at(index).is_valid())
    }

    fn all_valid(array: &ScalarFnArray) -> VortexResult<bool> {
        match array.scalar_fn.signature().null_handling() {
            NullHandling::Propagate | NullHandling::AbsorbsNull => {
                // Requires all children to guarantee all_valid
                for child in array.children().iter() {
                    if !child.all_valid()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            NullHandling::Custom => {
                // We cannot guarantee that the array is all valid without evaluating the function
                Ok(false)
            }
        }
    }

    fn all_invalid(array: &ScalarFnArray) -> VortexResult<bool> {
        match array.scalar_fn.signature().null_handling() {
            NullHandling::Propagate => {
                // All null if any child is all null
                for child in array.children().iter() {
                    if child.all_invalid()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            NullHandling::AbsorbsNull | NullHandling::Custom => {
                // We cannot guarantee that the array is all valid without evaluating the function
                Ok(false)
            }
        }
    }

    fn validity_mask(array: &ScalarFnArray) -> VortexResult<Mask> {
        let vector = array.execute(&SCALAR_FN_SESSION)?;
        Ok(Mask::from_buffer(vector.into_bool().into_bits()))
    }
}
