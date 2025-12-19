// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use arrow_schema::Fields;
use vortex_compute::arrow::IntoArrow;
use vortex_dtype::DType;
use vortex_dtype::StructFields;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::VectorExecutor;
use crate::arrays::ChunkedVTable;
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

    // If the array is chunked, then we invert the chunk-of-struct to struct-of-chunk.
    let array = match array.try_into::<ChunkedVTable>() {
        Ok(array) => {
            // NOTE(ngates): this currently uses the old into_canonical code path, but we should
            //  just call directly into the swizzle-chunks function.
            array.to_struct().into_array()
        }
        Err(array) => array,
    };

    // Attempt to short-circuit if the array is already a StructVTable:
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

    // Otherwise, we fall back to executing the full struct vector.
    // First we apply a cast to ensure we push down casting where possible into the struct fields.
    let vx_fields = StructFields::from_arrow(fields);
    let array = array.cast(DType::Struct(
        vx_fields,
        vortex_dtype::Nullability::Nullable,
    ))?;

    let struct_array = array.execute_vector(session)?.into_struct().into_arrow()?;

    // Finally, we cast to Arrow to ensure any types not representable by Vortex (e.g. Dictionary)
    // are properly converted.
    arrow_cast::cast(&struct_array, &DataType::Struct(fields.clone())).map_err(VortexError::from)
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
