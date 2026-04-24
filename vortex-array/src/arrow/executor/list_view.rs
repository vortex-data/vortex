// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::GenericListViewArray;
use arrow_array::OffsetSizeTrait;
use arrow_schema::FieldRef;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewDataParts;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability::NonNullable;

pub(super) fn to_arrow_list_view<O: OffsetSizeTrait + IntegerPType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<arrow_array::ArrayRef> {
    // Check for Vortex ListViewArray and convert directly.
    let array = match array.try_downcast::<ListView>() {
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
    let ListViewDataParts {
        elements,
        offsets,
        sizes,
        validity,
        ..
    } = array.into_data_parts();

    let elements = elements.execute_arrow(Some(elements_field.data_type()), ctx)?;
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

    let null_buffer = to_arrow_null_buffer(validity, offsets.len(), ctx)?;

    Ok(Arc::new(GenericListViewArray::<O>::new(
        Arc::clone(elements_field),
        offsets,
        sizes,
        elements,
        null_buffer,
    )))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::GenericListViewArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::executor::list_view::ListViewArray;
    use crate::arrow::executor::list_view::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn test_to_arrow_listview_i32() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create a ListViewArray with overlapping views: [[1, 2], [2, 3], [3, 4]]
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i32, 1, 2], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![2i32, 2, 2], Validity::NonNullable);

        let list_array = ListViewArray::new(
            elements.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::AllValid,
        );

        // Convert to Arrow ListView with i32 offsets.
        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::ListView(field.into());
        let arrow_array = list_array
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let listview = arrow_array
            .as_any()
            .downcast_ref::<GenericListViewArray<i32>>()
            .unwrap();

        assert_eq!(listview.len(), 3);

        // Verify first list view [1, 2].
        let first_list = listview.value(0);
        assert_eq!(first_list.len(), 2);
        let first_values = first_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(first_values.value(0), 1);
        assert_eq!(first_values.value(1), 2);

        // Verify second list view [2, 3].
        let second_list = listview.value(1);
        assert_eq!(second_list.len(), 2);
        let second_values = second_list
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(second_values.value(0), 2);
        assert_eq!(second_values.value(1), 3);
        Ok(())
    }

    #[test]
    fn test_to_arrow_listview_i64() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Create a ListViewArray with nullable elements: [[100], null, [200, 300]]
        let elements = PrimitiveArray::new(buffer![100i64, 200, 300], Validity::NonNullable);
        let offsets = PrimitiveArray::new(buffer![0i64, 1, 1], Validity::NonNullable);
        let sizes = PrimitiveArray::new(buffer![1i64, 0, 2], Validity::NonNullable);
        let validity = Validity::from_iter([true, false, true]);

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                elements.into_array(),
                offsets.into_array(),
                sizes.into_array(),
                validity,
            )
            .with_zero_copy_to_list(true)
        };

        // Convert to Arrow LargeListView with i64 offsets.
        let field = Field::new("item", DataType::Int64, false);
        let arrow_dt = DataType::LargeListView(field.into());
        let arrow_array = list_array
            .into_array()
            .execute_arrow(Some(&arrow_dt), &mut ctx)?;

        // Verify the type is correct.
        assert_eq!(arrow_array.data_type(), &arrow_dt);

        // Downcast and verify the structure.
        let listview = arrow_array
            .as_any()
            .downcast_ref::<GenericListViewArray<i64>>()
            .unwrap();

        assert_eq!(listview.len(), 3);
        assert!(!listview.is_null(0));
        assert!(listview.is_null(1));
        assert!(!listview.is_null(2));

        // Verify the third list [200, 300].
        let third_list = listview.value(2);
        assert_eq!(third_list.len(), 2);
        let third_values = third_list
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(third_values.value(0), 200);
        assert_eq!(third_values.value(1), 300);
        Ok(())
    }
}
