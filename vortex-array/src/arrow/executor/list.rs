// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListArray;
use arrow_array::OffsetSizeTrait;
use arrow_schema::DataType;
use arrow_schema::FieldRef;
use vortex_buffer::BufferMut;
use vortex_compute::arrow::IntoArrow;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VectorExecutor;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::vtable::ValidityHelper;

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_list<O: OffsetSizeTrait + NativePType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    // If the Vortex array is already in List format, we can directly convert it.
    if let Some(array) = array.as_opt::<ListVTable>() {
        return list_to_list::<O>(array, elements_field, session);
    }

    // If the Vortex array is a ListViewArray, we check for our magic cheap conversion flag.
    let array = match array.try_into::<ListViewVTable>() {
        Ok(array) => {
            if array.is_zero_copy_to_list() {
                return list_view_zctl::<O>(array, elements_field, session);
            }
            array.into_array()
        }
        Err(a) => a,
    };

    // TODO(ngates): we should do the slightly more expensive thing which is to verify ZCTL.
    //  In other words, check that offsets + sizes are monotonically increasing.

    // Otherwise, we execute the array to become a ListViewVector.
    let list_view = array.execute_vector(session)?.into_arrow()?;
    match O::IS_LARGE {
        true => arrow_cast::cast(&list_view, &DataType::LargeList(elements_field.clone())),
        false => arrow_cast::cast(&list_view, &DataType::List(elements_field.clone())),
    }
    .map_err(VortexError::from)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
fn list_to_list<O: OffsetSizeTrait + NativePType>(
    array: &ListArray,
    elements_field: &FieldRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    // We must cast the offsets to the required offset type.
    let offsets = array
        .offsets()
        .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
        .execute_vector(session)?
        .into_primitive()
        .downcast::<O>()
        .into_nonnull_buffer()
        .into_arrow_offset_buffer();

    let elements = array
        .elements()
        .clone()
        .execute_arrow(elements_field.data_type(), session)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert to non-nullable Arrow array with null elements"
    );

    let null_buffer = to_arrow_null_buffer(array.validity(), array.len(), session)?;

    // TODO(ngates): use new_unchecked when it is added to arrow-rs.
    Ok(Arc::new(GenericListArray::<O>::new(
        elements_field.clone(),
        offsets,
        elements,
        null_buffer,
    )))
}

fn list_view_zctl<O: OffsetSizeTrait + NativePType>(
    array: ListViewArray,
    elements_field: &FieldRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    assert!(array.is_zero_copy_to_list());

    let (elements, offsets, sizes, validity) = array.into_parts();

    // For ZCTL, we know that we only care about the final size.
    let final_size = sizes
        .scalar_at(sizes.len() - 1)
        .cast(&DType::Primitive(O::PTYPE, Nullability::NonNullable))?;
    let final_size = final_size
        .as_primitive()
        .typed_value::<O>()
        .vortex_expect("non null");

    let offsets = offsets
        .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
        .execute_vector(session)?
        .into_primitive()
        .downcast::<O>()
        .into_nonnull_buffer();

    // List arrays need one extra element in the offsets buffer to signify the end of the last list.
    // If the offsets original came from a list, chances are there is already capacity for this!
    let mut offsets = offsets.try_into_mut().unwrap_or_else(|o| {
        let mut new_offsets = BufferMut::<O>::with_capacity(o.len() + 1);
        new_offsets.extend_from_slice(&o);
        new_offsets
    });

    // We push the final offset.
    offsets.push(if offsets.is_empty() {
        final_size
    } else {
        offsets[offsets.len() - 1] + final_size
    });

    // Extract the elements array.
    let elements = elements.execute_arrow(elements_field.data_type(), session)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert to non-nullable Arrow array with null elements"
    );

    let null_buffer = to_arrow_null_buffer(&validity, sizes.len(), session)?;

    Ok(Arc::new(GenericListArray::<O>::new(
        elements_field.clone(),
        offsets.freeze().into_arrow_offset_buffer(),
        elements,
        null_buffer,
    )))
}
