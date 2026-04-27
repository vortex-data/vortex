// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListArray;
use arrow_array::OffsetSizeTrait;
use arrow_buffer::OffsetBuffer;
use arrow_schema::FieldRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::arrays::Chunked;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewDataParts;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;

/// Convert a Vortex VarBinArray into an Arrow [`GenericListArray`](arrow_array:array::GenericListArray).
pub(super) fn to_arrow_list<O: OffsetSizeTrait + NativePType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // If the Vortex array is already in List format, we can directly convert it.
    if let Some(array) = array.as_opt::<List>() {
        return list_to_list::<O>(&array.into_owned(), elements_field, ctx);
    }

    // Converting each chunk individually, then using the fast concat logic from arrow
    if let Some(chunked) = array.as_opt::<Chunked>() {
        let mut arrow_chunks: Vec<ArrowArrayRef> = Vec::with_capacity(chunked.nchunks());
        for chunk in chunked.chunks() {
            arrow_chunks.push(to_arrow_list::<O>(chunk.clone(), elements_field, ctx)?);
        }

        let refs = arrow_chunks.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        return Ok(arrow_select::concat::concat(&refs)?);
    }

    // If the Vortex array is a ListViewArray, rebuild to ZCTL if needed and convert.
    let array = match array.try_downcast::<ListView>() {
        Ok(array) => {
            let zctl = if array.is_zero_copy_to_list() {
                array
            } else {
                array.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
            };
            return list_view_zctl::<O>(zctl, elements_field, ctx);
        }
        Err(a) => a,
    };

    // Otherwise, we execute the array to become a ListViewArray, then rebuild to ZCTL.
    // Note: arrow_cast::cast supports ListView → List (apache/arrow-rs#8735), but it
    // unconditionally uses take. Our rebuild uses a heuristic that picks list-by-list
    // for large lists, which avoids materializing a large index buffer.
    let list_view = array.execute::<ListViewArray>(ctx)?;
    let zctl = if list_view.is_zero_copy_to_list() {
        list_view
    } else {
        list_view.rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
    };
    list_view_zctl::<O>(zctl, elements_field, ctx)
}

/// Convert a Vortex VarBinArray into an Arrow [`GenericListArray`](arrow_array:array::GenericListArray).
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

    let null_buffer = to_arrow_null_buffer(array.validity()?, array.len(), ctx)?;

    // TODO(ngates): use new_unchecked when it is added to arrow-rs.
    Ok(Arc::new(GenericListArray::<O>::new(
        Arc::clone(elements_field),
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
            Arc::clone(elements_field),
            OffsetBuffer::new_empty(),
            elements,
            None,
        )));
    }

    let ListViewDataParts {
        elements,
        offsets,
        sizes,
        validity,
        ..
    } = array.into_data_parts();

    // For ZCTL, we know that we only care about the final size.
    assert!(!sizes.is_empty());
    let final_size = sizes
        .execute_scalar(sizes.len() - 1, ctx)?
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
        Arc::clone(elements_field),
        offsets.freeze().into_arrow_offset_buffer(),
        elements,
        null_buffer,
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::GenericListArray;
    use arrow_array::Int32Array;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::executor::list::ListViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::validity::Validity;

    #[test]
    fn test_to_arrow_list_i32() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        let arrow_array = list_array
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;

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
        let first_values = first_list.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_values.value(0), 1);
        assert_eq!(first_values.value(1), 2);
        assert_eq!(first_values.value(2), 3);

        // Verify the values in the second list.
        let second_list = list.value(1);
        assert_eq!(second_list.len(), 2);
        let second_values = second_list.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_values.value(0), 4);
        assert_eq!(second_values.value(1), 5);
        Ok(())
    }

    #[test]
    fn test_to_arrow_list_i64() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        let arrow_array = list_array
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;

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
    fn test_to_arrow_list_non_zctl() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Overlapping lists are NOT zero-copy-to-list, so this exercises the rebuild path.
        // Elements: [1, 2, 3, 4], List 0: [1,2,3], List 1: [2,3,4] (overlap at indices 1-2)
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i32, 1], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![3i32, 3], Validity::NonNullable);

        let list_array = ListViewArray::new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::NonNullable,
        );
        assert!(!list_array.is_zero_copy_to_list());

        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::List(field.into());
        let arrow_array = list_array
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;

        let list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .unwrap();

        assert_eq!(list.len(), 2);

        let first = list.value(0);
        assert_eq!(first.len(), 3);
        let first_vals = first.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_vals.values(), &[1, 2, 3]);

        let second = list.value(1);
        assert_eq!(second.len(), 3);
        let second_vals = second.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_vals.values(), &[2, 3, 4]);
        Ok(())
    }

    #[test]
    fn test_to_arrow_list_empty_zctl() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::List(
            Arc::new(DType::Primitive(crate::dtype::PType::I32, NonNullable)),
            NonNullable,
        );
        let list_array = unsafe {
            Canonical::empty(&dtype)
                .into_listview()
                .with_zero_copy_to_list(true)
        };

        let arrow_dt = DataType::List(Field::new("item", DataType::Int32, false).into());
        let arrow_array = list_array
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;
        assert_eq!(arrow_array.len(), 0);
        Ok(())
    }
}
