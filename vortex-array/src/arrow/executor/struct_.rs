// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray;
use arrow_schema::Fields;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::StructVTable;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;

pub(super) fn to_arrow_struct(
    array: ArrayRef,
    fields: &Fields,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();
    let validity = array.validity_mask();

    let mut field_arrays = Vec::with_capacity(fields.len());

    match array.try_into::<StructVTable>() {
        Ok(array) => {
            // If the array is already a struct type, then we can convert each field.
            for (field, child) in fields.iter().zip_eq(array.into_fields().into_iter()) {
                let field_array = child.execute_arrow(field.data_type(), session)?;
                vortex_ensure!(
                    field.is_nullable() || field_array.null_count() == 0,
                    "Cannot convert field '{}' to non-nullable Arrow field because it contains nulls",
                    field.name()
                );
                field_arrays.push(field_array);
            }
        }
        Err(array) => {
            // Otherwise, we have some options:
            //  1. Use get_item expression to extract each field? This is a bit sad because get_item
            //     will perform the validity masking again.
            //  2. Execute a full struct vector. But this may do unnecessary work on fields that may
            //    have a more direct conversion to the desired Arrow field type.
            //  3. Something else?
            //
            // For now, we go with option 1. Although we really ought to figure out CSE for this.
            for field in fields.iter() {
                let field_array = array
                    .get_item(field.name().as_str())?
                    .execute_arrow(field.data_type(), session)?;
                vortex_ensure!(
                    field.is_nullable() || field_array.null_count() == 0,
                    "Cannot convert field '{}' to non-nullable Arrow field because it contains nulls",
                    field.name()
                );
                field_arrays.push(field_array);
            }
        }
    }

    Ok(Arc::new(unsafe {
        StructArray::new_unchecked_with_length(
            fields.clone(),
            field_arrays.into(),
            to_null_buffer(validity),
            len,
        )
    }))
}
