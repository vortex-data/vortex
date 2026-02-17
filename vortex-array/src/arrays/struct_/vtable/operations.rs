// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<StructVTable> for StructVTable {
    fn scalar_at(array: &StructArray, index: usize) -> VortexResult<Scalar> {
        let field_scalars: VortexResult<Vec<_>> = array
            .unmasked_fields()
            .iter()
            .map(|field| field.scalar_at(index))
            .collect();
        Ok(Scalar::struct_(array.dtype().clone(), field_scalars?))
    }
}
