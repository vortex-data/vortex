// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<StructVTable> for StructVTable {
    fn scalar_at(array: &StructArray, index: usize) -> VortexResult<Scalar> {
        let field_scalars: VortexResult<Vec<Scalar>> = array
            .unmasked_fields()
            .iter()
            .map(|field| field.scalar_at(index))
            .collect();
        // SAFETY: The vtable guarantees index is in-bounds and non-null before this is called.
        // Each field's scalar_at returns a scalar with the field's own dtype.
        Ok(unsafe { Scalar::struct_unchecked(array.dtype().clone(), field_scalars?) })
    }
}
