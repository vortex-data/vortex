// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<StructVTable> for StructVTable {
    fn scalar_at(array: &StructArray, index: usize) -> VortexResult<Scalar> {
        // The vtable contract guarantees index is non-null before this is called.
        let field_scalars: VortexResult<Vec<_>> = array
            .scalar_at_fields(index)?
            .vortex_expect("scalar_at precondition: index is guaranteed non-null")
            .collect();
        Ok(Scalar::struct_(array.dtype().clone(), field_scalars?))
    }
}
