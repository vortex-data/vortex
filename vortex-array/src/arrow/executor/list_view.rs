// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::GenericListViewArray;
use arrow_array::OffsetSizeTrait;
use arrow_schema::FieldRef;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::PrimitiveArray;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;

pub(super) fn to_arrow_list_view<O: OffsetSizeTrait + IntegerPType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<arrow_array::ArrayRef> {
    // Check for Vortex ListViewArray and convert directly.
    let array = match array.try_into::<ListViewVTable>() {
        Ok(array) => return list_view_to_list_view::<O>(array, elements_field, ctx),
        Err(array) => array,
    };

    // Otherwise, we execute to ListViewArray and convert.
    let list_view_array = array.execute::<ListViewArray>(ctx)?;
    list_view_to_list_view::<O>(list_view_array, elements_field, ctx)
}

fn list_view_to_list_view<O: OffsetSizeTrait + IntegerPType>(
    array: ListViewArray,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<arrow_array::ArrayRef> {
    let (elements, offsets, sizes, validity) = array.into_parts();

    let elements = elements.execute_arrow(elements_field.data_type(), ctx)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Elements field is non-nullable but elements array contains nulls"
    );

    let offsets = offsets
        .cast(DType::Primitive(O::PTYPE, NonNullable))?
        .execute::<PrimitiveArray>(ctx)?
        .to_buffer::<O>()
        .into_arrow_scalar_buffer();
    let sizes = sizes
        .cast(DType::Primitive(O::PTYPE, NonNullable))?
        .execute::<PrimitiveArray>(ctx)?
        .to_buffer::<O>()
        .into_arrow_scalar_buffer();

    let null_buffer = to_arrow_null_buffer(&validity, offsets.len(), ctx)?;

    Ok(Arc::new(GenericListViewArray::<O>::new(
        elements_field.clone(),
        offsets,
        sizes,
        elements,
        null_buffer,
    )))
}
