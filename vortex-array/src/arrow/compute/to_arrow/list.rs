// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListArray;
use arrow_array::OffsetSizeTrait;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VectorExecutor;
use crate::VortexSessionExecute;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::list_view_from_list;
use crate::arrow::IntoArrowArray;
use crate::arrow::compute::ToArrowKernel;
use crate::arrow::compute::ToArrowKernelAdapter;
use crate::arrow::null_buffer::to_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::register_kernel;

impl ToArrowKernel for ListVTable {
    fn to_arrow(
        &self,
        array: &ListArray,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        match arrow_type {
            None => {
                // Default to a `ListArray` with `i32` offsets (preferred) when no `arrow_type` is
                // specified.
                list_array_to_arrow_list::<i32>(array, None)
            }
            Some(DataType::List(field)) => list_array_to_arrow_list::<i32>(array, Some(field)),
            Some(DataType::LargeList(field)) => list_array_to_arrow_list::<i64>(array, Some(field)),
            Some(dt @ DataType::ListView(_)) | Some(dt @ DataType::LargeListView(_)) => {
                // Convert `ListArray` to `ListViewArray`, then use the canonical conversion.
                let list_view = list_view_from_list(array.clone());
                Ok(list_view.into_array().into_arrow(dt)?)
            }
            _ => vortex_bail!(
                "Cannot convert `ListArray` to non-list Arrow type: {:?}",
                arrow_type
            ),
        }
        .map(Some)
    }
}

register_kernel!(ToArrowKernelAdapter(ListVTable).lift());

/// Converts a Vortex [`ListArray`] directly into an arrow [`GenericListArray`].
fn list_array_to_arrow_list<O: IntegerPType + OffsetSizeTrait>(
    array: &ListArray,
    element: Option<&FieldRef>,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the offsets and sizes into the specified width (determined by `O::PTYPE`).
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let offsets = array
        .offsets()
        .cast(DType::Primitive(O::PTYPE, array.dtype().nullability()))?
        .execute(&mut ctx)?
        .to_vector(&mut ctx)?
        .into_primitive()
        .downcast::<O>()
        .into_nonnull_buffer();

    // Convert `offsets` and `validity` to Arrow buffers.
    let arrow_offsets = offsets.into_arrow_offset_buffer();
    let nulls = to_null_buffer(array.validity_mask());

    // Convert the child `elements` array to Arrow.
    let (elements, element_field) = {
        if let Some(element) = element {
            // Convert elements to the specific Arrow type the caller wants.
            (
                array.elements().clone().into_arrow(element.data_type())?,
                element.clone(),
            )
        } else {
            // Otherwise, convert into whatever Arrow prefers.
            let elements = array.elements().clone().into_arrow_preferred()?;
            let element_field = Arc::new(Field::new_list_field(
                elements.data_type().clone(),
                array.elements().dtype().is_nullable(),
            ));
            (elements, element_field)
        }
    };

    Ok(Arc::new(GenericListArray::new(
        element_field,
        arrow_offsets,
        elements,
        nulls,
    )))
}
