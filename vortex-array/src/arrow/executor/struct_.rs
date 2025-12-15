// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::Fields;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ScalarFnVTable;
use crate::arrays::StructVTable;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::expr::Pack;
use crate::vtable::ValidityHelper;

pub(super) fn to_arrow_struct(
    array: ArrayRef,
    fields: &Fields,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();

    // First, we attempt to short-circuit if the array is already a StructVTable:
    let array = match array.try_into::<StructVTable>() {
        Ok(array) => {
            let validity = to_arrow_null_buffer(array.validity(), array.len(), session)?;
            return create_from_fields(fields, array.into_fields(), validity, len, session);
        }
        Err(array) => array,
    };

    // We can also short-circuit if the array is a `pack` scalar function:
    if let Some(array) = array.as_opt::<ScalarFnVTable>()
        && let Some(_pack_options) = array.scalar_fn().as_opt::<Pack>()
    {
        return create_from_fields(
            fields,
            array.children().to_vec(),
            None, // Pack is never null,
            len,
            session,
        );
    }

    // Otherwise, we have some options:
    //  1. Use get_item expression to extract each field? This is a bit sad because get_item
    //     will perform the validity masking again.
    //  2. Execute a full struct vector. But this may do unnecessary work on fields that may
    //    have a more direct conversion to the desired Arrow field type.
    //  3. Something else?
    //
    // For now, we go with option 1. Although we really ought to figure out CSE for this.
    let field_arrays = fields
        .iter()
        .map(|f| array.get_item(f.name().as_str()))
        .try_collect()?;

    if !array.all_valid() {
        // TODO(ngates): we should grab the nullability using the is_not_null expression.
        vortex_bail!(
            "Cannot convert nullable Struct array with nulls to Arrow\n{}",
            array.display_tree()
        );
    }

    create_from_fields(fields, field_arrays, None, len, session)
}

fn create_from_fields(
    fields: &Fields,
    vortex_fields: Vec<ArrayRef>,
    null_buffer: Option<NullBuffer>,
    len: usize,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let mut arrow_fields = Vec::with_capacity(vortex_fields.len());
    for (field, vx_field) in fields.iter().zip(vortex_fields.into_iter()) {
        let arrow_field = vx_field.execute_arrow(field.data_type(), session)?;
        vortex_ensure!(
            field.is_nullable() || arrow_field.null_count() == 0,
            "Cannot convert field '{}' to non-nullable Arrow field because it contains nulls",
            field.name()
        );
        arrow_fields.push(arrow_field);
    }

    Ok(Arc::new(unsafe {
        StructArray::new_unchecked_with_length(fields.clone(), arrow_fields, null_buffer, len)
    }))
}
