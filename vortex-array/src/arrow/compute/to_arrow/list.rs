// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef as ArrowArrayRef, GenericListArray, OffsetSizeTrait};
use arrow_schema::{DataType, Field, FieldRef};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{ListArray, ListVTable, list_view_from_list};
use crate::arrow::IntoArrowArray;
use crate::arrow::compute::{ToArrowKernel, ToArrowKernelAdapter};
use crate::compute::cast;
use crate::{IntoArray, OffsetPType, ToCanonical, register_kernel};

impl ToArrowKernel for ListVTable {
    fn to_arrow(
        &self,
        array: &ListArray,
        arrow_type: Option<&DataType>,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        match arrow_type {
            None => {
                // Default to List with i32 offsets when no type is specified.
                list_array_to_arrow_list::<i32>(array, None)
            }
            Some(DataType::List(field)) => list_array_to_arrow_list::<i32>(array, Some(field)),
            Some(DataType::LargeList(field)) => list_array_to_arrow_list::<i64>(array, Some(field)),
            Some(dt @ DataType::ListView(_)) | Some(dt @ DataType::LargeListView(_)) => {
                // Convert ListArray to ListViewArray, then use the canonical conversion.
                let list_view = list_view_from_list(array.clone());
                Ok(list_view.into_array().into_arrow(dt)?)
            }
            _ => vortex_bail!(
                "Cannot convert ListArray to non-list Arrow type: {:?}",
                arrow_type
            ),
        }
        .map(Some)
    }
}

register_kernel!(ToArrowKernelAdapter(ListVTable).lift());

/// Converts a Vortex [`ListArray`] directly into an arrow [`GenericListArray`].
fn list_array_to_arrow_list<O: OffsetPType + OffsetSizeTrait>(
    array: &ListArray,
    element: Option<&FieldRef>,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the offsets into the correct width.
    let offsets_dtype = DType::Primitive(O::PTYPE, array.dtype().nullability());
    let arrow_offsets = cast(array.offsets(), &offsets_dtype)
        .map_err(|err| err.with_context(format!("Failed to cast offsets to {offsets_dtype}")))?
        .to_primitive();

    let (values, element_field) = if let Some(element) = element {
        (
            array.elements().clone().into_arrow(element.data_type())?,
            element.clone(),
        )
    } else {
        let values = array.elements().clone().into_arrow_preferred()?;
        let element_field = Arc::new(Field::new_list_field(
            values.data_type().clone(),
            array.elements().dtype().is_nullable(),
        ));
        (values, element_field)
    };
    let nulls = array.validity_mask().to_null_buffer();

    Ok(Arc::new(GenericListArray::new(
        element_field,
        arrow_offsets.buffer::<O>().into_arrow_offset_buffer(),
        values,
        nulls,
    )))
}
