// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListArray;
use arrow_array::OffsetSizeTrait;
use arrow_buffer::OffsetBuffer;
use arrow_schema::FieldRef;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewArrayParts;
use crate::arrays::ListViewVTable;
use crate::arrays::PrimitiveArray;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_list<O: OffsetSizeTrait + NativePType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // If the Vortex array is already in List format, we can directly convert it.
    if let Some(array) = array.as_opt::<ListVTable>() {
        return list_to_list::<O>(array, elements_field, ctx);
    }

    // If the Vortex array is a ListViewArray, we check for our magic cheap conversion flag.
    let array = match array.try_into::<ListViewVTable>() {
        Ok(array) => {
            if array.is_zero_copy_to_list() {
                return list_view_zctl::<O>(array, elements_field, ctx);
            } else {
                return list_view_to_list::<O>(array, elements_field, ctx);
            }
        }
        Err(a) => a,
    };

    // TODO(ngates): we should do the slightly more expensive thing which is to verify ZCTL.
    //  In other words, check that offsets + sizes are monotonically increasing.

    // Otherwise, we execute the array to become a ListViewArray.
    let list_view = array.execute::<ListViewArray>(ctx)?;
    list_view_to_list::<O>(list_view, elements_field, ctx)

    // FIXME(ngates): we need this PR from arrow-rs:
    //  https://github.com/apache/arrow-rs/pull/8735
    // let list_view = array.execute(session)?.into_arrow()?;
    // match O::IS_LARGE {
    //     true => arrow_cast::cast(&list_view, &DataType::LargeList(elements_field.clone())),
    //     false => arrow_cast::cast(&list_view, &DataType::List(elements_field.clone())),
    // }
    // .map_err(VortexError::from)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
fn list_to_list<O: OffsetSizeTrait + NativePType>(
    array: &ListArray,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // We must cast the offsets to the required offset type.
    let offsets = array
        .offsets()
        .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
        .execute::<Canonical>(ctx)?
        .into_primitive()
        .to_buffer::<O>()
        .into_arrow_offset_buffer();

    let elements = array
        .elements()
        .clone()
        .execute_arrow(Some(elements_field.data_type()), ctx)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert to non-nullable Arrow array with null elements"
    );

    let null_buffer = to_arrow_null_buffer(array.validity().clone(), array.len(), ctx)?;

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
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    assert!(array.is_zero_copy_to_list());

    if array.is_empty() {
        let elements = array
            .elements()
            .clone()
            .execute_arrow(Some(elements_field.data_type()), ctx)?;
        return Ok(Arc::new(GenericListArray::<O>::new(
            elements_field.clone(),
            OffsetBuffer::new_empty(),
            elements,
            None,
        )));
    }

    let ListViewArrayParts {
        elements,
        offsets,
        sizes,
        validity,
        ..
    } = array.into_parts();

    // For ZCTL, we know that we only care about the final size.
    assert!(!sizes.is_empty());
    let final_size = sizes
        .scalar_at(sizes.len() - 1)?
        .cast(&DType::Primitive(O::PTYPE, Nullability::NonNullable))?;
    let final_size = final_size
        .as_primitive()
        .typed_value::<O>()
        .vortex_expect("non null");

    let offsets = offsets
        .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
        .execute::<Canonical>(ctx)?
        .into_primitive()
        .to_buffer::<O>();

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
    let elements = elements.execute_arrow(Some(elements_field.data_type()), ctx)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert to non-nullable Arrow array with null elements"
    );

    let null_buffer = to_arrow_null_buffer(validity, sizes.len(), ctx)?;

    Ok(Arc::new(GenericListArray::<O>::new(
        elements_field.clone(),
        offsets.freeze().into_arrow_offset_buffer(),
        elements,
        null_buffer,
    )))
}

fn list_view_to_list<O: OffsetSizeTrait + NativePType>(
    array: ListViewArray,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let ListViewArrayParts {
        elements,
        offsets,
        sizes,
        validity,
        ..
    } = array.into_parts();

    let offsets = offsets
        .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
        .execute::<Canonical>(ctx)?
        .into_primitive()
        .to_buffer::<O>();
    let sizes = sizes
        .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
        .execute::<Canonical>(ctx)?
        .into_primitive()
        .to_buffer::<O>();

    // We create a new offsets buffer for the final list array.
    // And we also create an `indices` buffer for taking the elements.
    let mut new_offsets = BufferMut::<O>::with_capacity(offsets.len() + 1);
    let mut take_indices = BufferMut::<u32>::with_capacity(elements.len());

    // Add the offset for the first subarray
    new_offsets.push(O::zero());
    for (offset, size) in offsets.iter().zip(sizes.iter()) {
        let offset = offset.as_usize();
        let size = size.as_usize();
        let end = offset + size;
        for j in offset..end {
            take_indices.push(u32::try_from(j).map_err(|_| {
                vortex_err!("List array too large for {} indices", type_name::<O>())
            })?);
        }
        new_offsets.push(O::usize_as(take_indices.len()));
    }
    assert_eq!(new_offsets.len(), offsets.len() + 1);

    // Now we can "take" the elements using the computed indices.
    let elements =
        elements.take(PrimitiveArray::new(take_indices, Validity::NonNullable).into_array())?;

    let elements = elements.execute_arrow(Some(elements_field.data_type()), ctx)?;
    vortex_ensure!(
        elements_field.is_nullable() || elements.null_count() == 0,
        "Cannot convert to non-nullable Arrow array with null elements"
    );

    let null_buffer = to_arrow_null_buffer(validity, sizes.len(), ctx)?;

    Ok(Arc::new(GenericListArray::<O>::new(
        elements_field.clone(),
        new_offsets.freeze().into_arrow_offset_buffer(),
        elements,
        null_buffer,
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::GenericListArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrow::IntoArrowArray;
    use crate::validity::Validity;

    #[test]
    fn test_to_arrow_list_i32() -> VortexResult<()> {
        // Create a ListViewArray with i32 elements: [[1, 2, 3], [4, 5]]
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i32, 3], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![3i32, 2], Validity::NonNullable);

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                elements.into_array(),
                offsets.into_array(),
                sizes.into_array(),
                Validity::AllValid,
            )
            .with_zero_copy_to_list(true)
        };

        // Convert to Arrow List with i32 offsets.
        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::List(field.into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt)?;

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .unwrap();

        assert_eq!(list.len(), 2);
        assert!(!list.is_null(0));
        assert!(!list.is_null(1));

        // Verify the values in the first list.
        let first_list = list.value(0);
        assert_eq!(first_list.len(), 3);
        let first_values = first_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(first_values.value(0), 1);
        assert_eq!(first_values.value(1), 2);
        assert_eq!(first_values.value(2), 3);

        // Verify the values in the second list.
        let second_list = list.value(1);
        assert_eq!(second_list.len(), 2);
        let second_values = second_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(second_values.value(0), 4);
        assert_eq!(second_values.value(1), 5);
        Ok(())
    }

    #[test]
    fn test_to_arrow_list_i64() -> VortexResult<()> {
        // Create a ListViewArray with i64 offsets: [[10, 20], [30]]
        let elements = PrimitiveArray::new(buffer![10i64, 20, 30], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i64, 2], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![2i64, 1], Validity::NonNullable);

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                elements.into_array(),
                offsets.into_array(),
                sizes.into_array(),
                Validity::AllValid,
            )
            .with_zero_copy_to_list(true)
        };

        // Convert to Arrow LargeList with i64 offsets.
        let field = Field::new("item", DataType::Int64, false);
        let arrow_dt = DataType::LargeList(field.into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt)?;

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i64>>()
            .unwrap();

        assert_eq!(list.len(), 2);
        assert!(!list.is_null(0));
        assert!(!list.is_null(1));
        Ok(())
    }

    #[test]
    fn test_to_arrow_list_empty_zctl() -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(
                vortex_dtype::PType::I32,
                Nullability::NonNullable,
            )),
            Nullability::NonNullable,
        );
        let list_array = unsafe {
            Canonical::empty(&dtype)
                .into_listview()
                .with_zero_copy_to_list(true)
        };

        let arrow_dt = DataType::List(Field::new("item", DataType::Int32, false).into());
        let arrow_array = list_array.into_array().into_arrow(&arrow_dt)?;
        assert_eq!(arrow_array.len(), 0);
        Ok(())
    }
}
