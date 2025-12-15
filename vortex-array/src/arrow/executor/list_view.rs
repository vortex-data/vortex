// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::GenericListViewArray;
use arrow_array::OffsetSizeTrait;
use arrow_schema::FieldRef;
use vortex_compute::arrow::IntoArrow;
use vortex_compute::cast::Cast;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_vector::listview::ListViewVector;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;

pub(super) fn to_arrow_list_view<O: OffsetSizeTrait + IntegerPType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    session: &VortexSession,
) -> VortexResult<arrow_array::ArrayRef> {
    // Check for Vortex ListViewArray and convert directly.
    let array = match array.try_into::<ListViewVTable>() {
        Ok(array) => return list_view_to_list_view::<O>(array, elements_field, session),
        Err(array) => array,
    };

    // Otherwise, we execute as a vector and convert.
    let mut vector = array
        .execute_vector(session)?
        .into_list_opt()
        .ok_or_else(|| vortex_err!("Failed to convert array to ListVector"))?;

    // Ensure the offset type matches.
    if vector.offsets().ptype() != O::PTYPE || vector.sizes().ptype() != O::PTYPE {
        let (elements, offsets, sizes, validity) = vector.into_parts();
        let offsets = offsets
            .cast(&DType::Primitive(O::PTYPE, NonNullable))?
            .into_primitive();
        let sizes = sizes
            .cast(&DType::Primitive(O::PTYPE, NonNullable))?
            .into_primitive();
        vector = unsafe { ListViewVector::new_unchecked(elements, offsets, sizes, validity) };
    }

    Ok(Arc::new(vector.into_arrow()?))
}

fn list_view_to_list_view<O: OffsetSizeTrait + IntegerPType>(
    array: ListViewArray,
    elements_field: &FieldRef,
    session: &VortexSession,
) -> VortexResult<arrow_array::ArrayRef> {
    let (elements, offsets, sizes, validity) = array.into_parts();

    let elements = elements.execute_arrow(elements_field.data_type(), session)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Elements field is non-nullable but elements array contains nulls"
    );

    let offsets = offsets
        .cast(DType::Primitive(O::PTYPE, NonNullable))?
        .execute_vector(session)?
        .into_primitive()
        .downcast::<O>()
        .into_nonnull_buffer()
        .into_arrow_scalar_buffer();
    let sizes = sizes
        .cast(DType::Primitive(O::PTYPE, NonNullable))?
        .execute_vector(session)?
        .into_primitive()
        .downcast::<O>()
        .into_nonnull_buffer()
        .into_arrow_scalar_buffer();

    let null_buffer = to_arrow_null_buffer(&validity, offsets.len(), session)?;

    Ok(Arc::new(GenericListViewArray::<O>::new(
        elements_field.clone(),
        offsets,
        sizes,
        elements,
        null_buffer,
    )))
}
