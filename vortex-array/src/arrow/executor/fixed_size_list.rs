// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::FieldRef;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;

pub(super) fn to_arrow_fixed_list(
    array: ArrayRef,
    list_size: i32,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<arrow_array::ArrayRef> {
    // Check for Vortex FixedSizeListArray and convert directly.
    if let Some(array) = array.as_opt::<FixedSizeList>() {
        return list_to_list(&array.into_owned(), elements_field, list_size, ctx);
    }

    // Otherwise, we execute the array to become a FixedSizeListArray.
    let fixed_size_list = array.execute::<FixedSizeListArray>(ctx)?;
    list_to_list(&fixed_size_list, elements_field, list_size, ctx)
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
        .execute_arrow(Some(elements_field.data_type()), ctx)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert FixedSizeListArray to non-nullable Arrow array when elements are nullable"
    );

    let null_buffer = to_arrow_null_buffer(array.validity(), array.len(), ctx)?;

    Ok(Arc::new(
        arrow_array::FixedSizeListArray::try_new_with_length(
            elements_field.clone(),
            list_size,
            elements,
            null_buffer,
            array.len(),
        )?,
    ))
}
