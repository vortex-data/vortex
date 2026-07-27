// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListArray;
use arrow_array::OffsetSizeTrait;
use arrow_buffer::OffsetBuffer;
use arrow_schema::FieldRef;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::ListViewDataParts;
use vortex_array::arrays::listview::ListViewRebuildMode;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::matcher::Matcher;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::executor::validity::to_arrow_null_buffer;
use crate::session::ArrowSessionExt;

/// Matches the encodings [`to_arrow_list`] requires for export.
struct ArrowListExportable;

impl Matcher for ArrowListExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<List>() || array.is::<Chunked>() || array.is::<ListView>()).then_some(array)
    }
}

#[allow(rustdoc::broken_intra_doc_links)]
/// Convert a Vortex VarBinArray into an Arrow [`GenericListArray`](arrow_array:array::GenericListArray).
pub(super) fn to_arrow_list<O: OffsetSizeTrait + NativePType>(
    array: ArrayRef,
    elements_field: &FieldRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = array.execute_until::<ArrowListExportable>(ctx)?;

    // If the Vortex array is already in List format, we can directly convert it.
    if let Some(list) = array.as_opt::<List>() {
        return list_to_list::<O>(&list.into_owned(), elements_field, ctx);
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

    // Otherwise the array is canonical: a ListViewArray, which we rebuild to ZCTL if needed.
    // Note: arrow_cast::cast supports ListView → List (apache/arrow-rs#8735), but it
    // unconditionally uses take. Our rebuild uses a heuristic that picks list-by-list
    // for large lists, which avoids materializing a large index buffer.
    let list_view = array
        .as_opt::<ListView>()
        .vortex_expect("Must be ListView from Matcher")
        .into_owned();

    let zctl = if list_view.is_zero_copy_to_list() {
        list_view
    } else {
        list_view.rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)?
    };
    list_view_zctl::<O>(zctl, elements_field, ctx)
}

#[allow(rustdoc::broken_intra_doc_links)]
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

    let elements = ctx.session().clone().arrow().execute_arrow(
        array.elements().clone(),
        Some(elements_field.as_ref()),
        ctx,
    )?;
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
        let elements = ctx.session().clone().arrow().execute_arrow(
            array.elements().clone(),
            Some(elements_field.as_ref()),
            ctx,
        )?;
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
    let elements = ctx.session().clone().arrow().execute_arrow(
        elements,
        Some(elements_field.as_ref()),
        ctx,
    )?;
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
    use std::sync::LazyLock;

    use arrow_array::Array;
    use arrow_array::GenericListArray;
    use arrow_array::Int32Array;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::fns::mask::Mask;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrowArrayExecutor;
    use crate::executor::list::ListArray;
    use crate::executor::list::ListViewArray;

    /// A shared session for these list-executor tests, used to create execution contexts.
    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    #[test]
    fn test_to_arrow_list_i32() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
        let mut ctx = SESSION.create_execution_ctx();
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
        let mut ctx = SESSION.create_execution_ctx();
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
    fn slice_wrapped_list_exports() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Lists [[1, 2], [3], [4, 5, 6]] behind a lazy Slice wrapper selecting rows 1..3.
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable);
        let elements_ptr = elements.as_slice::<i32>().as_ptr();
        let offsets = PrimitiveArray::new(buffer![0i32, 2, 3, 6], Validity::NonNullable);
        let list = ListArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )?;
        let sliced = SliceArray::new(list.into_array(), 1..3).into_array();

        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::List(field.into());
        let arrow_array = sliced.execute_arrow(Some(&arrow_dt), &mut ctx)?;

        let arrow_list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .unwrap();
        assert_eq!(arrow_list.len(), 2);
        let first = arrow_list.value(0);
        let first_vals = first.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_vals.values(), &[3]);
        let second = arrow_list.value(1);
        let second_vals = second.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(second_vals.values(), &[4, 5, 6]);

        // The conversion shares the elements buffer regardless of which path resolves the
        // Slice (revealed List or zero-copy-to-list ListView).
        let values = arrow_list
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.values().as_ptr(), elements_ptr);
        Ok(())
    }

    #[test]
    fn mask_wrapped_list_exports() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Lists [[1, 2], [3], [4, 5, 6]] behind a lazy `mask` scalar-fn nulling out row 1 —
        // the shape a scan produces when a row mask is applied to a List-encoded column.
        let elements = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable);
        let elements_ptr = elements.as_slice::<i32>().as_ptr();
        let offsets = PrimitiveArray::new(buffer![0i32, 2, 3, 6], Validity::NonNullable);
        let list = ListArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )?;
        let mask = BoolArray::from_iter([true, false, true]);
        let masked = Mask.try_new_array(3, EmptyOptions, [list.into_array(), mask.into_array()])?;

        let field = Field::new("item", DataType::Int32, false);
        let arrow_dt = DataType::List(field.into());
        let arrow_array = masked.execute_arrow(Some(&arrow_dt), &mut ctx)?;

        let arrow_list = arrow_array
            .as_any()
            .downcast_ref::<GenericListArray<i32>>()
            .unwrap();
        assert_eq!(arrow_list.len(), 3);
        assert!(!arrow_list.is_null(0));
        assert!(arrow_list.is_null(1));
        assert!(!arrow_list.is_null(2));
        let first = arrow_list.value(0);
        let first_vals = first.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_vals.values(), &[1, 2]);
        let third = arrow_list.value(2);
        let third_vals = third.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(third_vals.values(), &[4, 5, 6]);

        // Masking only touches validity, so the conversion still shares the elements buffer.
        let values = arrow_list
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.values().as_ptr(), elements_ptr);
        Ok(())
    }

    #[test]
    fn test_to_arrow_list_empty_zctl() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::List(
            Arc::new(DType::Primitive(
                vortex_array::dtype::PType::I32,
                NonNullable,
            )),
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
