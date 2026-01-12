// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::FieldRef;
use vortex_compute::arrow::IntoArrow;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::vtable::ValidityHelper;

pub(super) fn to_arrow_fixed_list(
    array: ArrayRef,
    list_size: i32,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<arrow_array::ArrayRef> {
    // Check for Vortex FixedSizeListArray and convert directly.
    if let Some(array) = array.as_opt::<FixedSizeListVTable>() {
        return list_to_list(array, elements_field, list_size, ctx);
    }

    // Otherwise, we execute the array to become a FixedSizeListArray.
    let vector = array
        .execute(ctx)?
        .to_vector(ctx)?
        .into_fixed_size_list_opt()
        .ok_or_else(|| vortex_err!("Failed to convert array to FixedSizeListArray"))?;
    vortex_ensure!(
        Ok(list_size) == i32::try_from(vector.list_size()),
        "Cannot convert FixedSizeList with list size {} to Arrow array with list size {}",
        vector.list_size(),
        list_size
    );

    Ok(Arc::new(vector.into_arrow()?))
}

fn list_to_list(
    array: &FixedSizeListArray,
    elements_field: &FieldRef,
    list_size: i32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<arrow_array::ArrayRef> {
    vortex_ensure!(
        Ok(list_size) == i32::try_from(array.list_size()),
        "Cannot convert FixedSizeList with list size {} to Arrow array with list size {}",
        array.list_size(),
        list_size
    );

    let elements = array
        .elements()
        .clone()
        .execute_arrow(elements_field.data_type(), ctx)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert FixedSizeListArray to non-nullable Arrow array when elements are nullable"
    );

    let null_buffer = to_arrow_null_buffer(array.validity(), array.len(), ctx)?;

    Ok(Arc::new(arrow_array::FixedSizeListArray::new(
        elements_field.clone(),
        list_size,
        elements,
        null_buffer,
    )))
}
