// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::FieldRef;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Map;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::map::MapArrayExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::executor::list::to_arrow_list;

/// Converts a Vortex Map array into an Arrow [`MapArray`](ArrowMapArray).
///
/// The Map entries are exported through the ListView-to-Arrow List path and then repackaged with
/// Arrow's Map field metadata. When the requested Arrow field asserts sorted keys, the Vortex Map
/// dtype must already make the same assertion.
pub(super) fn to_arrow_map(
    array: ArrayRef,
    entries_field: &FieldRef,
    keys_sorted: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = match array.try_downcast::<Map>() {
        Ok(map) => map,
        Err(array) => array.execute::<MapArray>(ctx)?,
    };

    vortex_ensure!(
        !keys_sorted || array.keys_sorted(),
        "Cannot convert unsorted Vortex map to Arrow MapArray with keys_sorted=true"
    );

    let entries = array.entries().clone();
    let entries_list_type = DataType::List(Arc::clone(entries_field));
    let entries_list = to_arrow_list::<i32>(entries, entries_field, ctx)?;
    vortex_ensure!(
        entries_list.data_type() == &entries_list_type,
        "Arrow Map entries converted to {}, expected {entries_list_type}",
        entries_list.data_type()
    );

    let entries_list = entries_list.as_list::<i32>();
    let entries = entries_list.values().as_struct().clone();
    let map = ArrowMapArray::try_new(
        Arc::clone(entries_field),
        entries_list.offsets().clone(),
        entries,
        entries_list.nulls().cloned(),
        keys_sorted,
    )?;

    Ok(Arc::new(map))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array as ArrowArray;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::MapArray as ArrowMapArray;
    use arrow_array::StringArray;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::builder::Int32Builder;
    use arrow_array::builder::MapBuilder as ArrowMapBuilder;
    use arrow_array::builder::StringBuilder;
    use arrow_array::cast::AsArray;
    use arrow_buffer::NullBuffer;
    use arrow_buffer::OffsetBuffer;
    use arrow_buffer::ScalarBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ListView;
    use vortex_array::arrays::Map as VortexMap;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_array::arrays::map::MapArraySlotsExt;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::MapBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::MapDType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::FromArrowArray as _;
    use crate::session::ArrowSessionExt as _;

    fn i32_utf8_map_field(keys_sorted: bool, nullable: bool) -> Field {
        let fields = Fields::from(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]);
        Field::new(
            "maps",
            DataType::Map(
                Arc::new(Field::new_struct("entries", fields, false)),
                keys_sorted,
            ),
            nullable,
        )
    }

    fn i32_utf8_map_array(
        offsets: Vec<i32>,
        keys: Vec<i32>,
        values: Vec<Option<&str>>,
        nulls: Option<NullBuffer>,
        keys_sorted: bool,
    ) -> VortexResult<ArrowMapArray> {
        let fields = Fields::from(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]);
        let entries = ArrowStructArray::try_new(
            fields.clone(),
            vec![
                Arc::new(Int32Array::from(keys)),
                Arc::new(StringArray::from(values)),
            ],
            None,
        )?;
        ArrowMapArray::try_new(
            Arc::new(Field::new_struct("entries", fields, false)),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            entries,
            nulls,
            keys_sorted,
        )
        .map_err(Into::into)
    }

    #[test]
    fn map_roundtrip_preserves_null_empty_duplicate_and_sorted_rows() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();
        let field = i32_utf8_map_field(true, true);
        let arrow = i32_utf8_map_array(
            vec![0, 2, 2, 2, 4],
            vec![1, 2, 1, 1],
            vec![Some("one"), None, Some("dup-old"), Some("dup-new")],
            Some(NullBuffer::from_iter([true, false, true, true])),
            true,
        )?;

        let vortex = session.from_arrow_array(Arc::new(arrow), &field)?;
        assert_eq!(
            vortex.dtype(),
            &DType::map(
                DType::Primitive(PType::I32, NonNullable),
                DType::Utf8(Nullable),
                true,
                Nullable,
            )?
        );

        let exported = session.execute_arrow(vortex, Some(&field), &mut ctx)?;
        let map = exported.as_map();
        assert_eq!(map.value_offsets(), &[0, 2, 2, 2, 4]);
        assert!(map.is_valid(0));
        assert!(map.is_null(1));
        assert!(map.is_valid(2));
        assert!(map.is_valid(3));
        assert_eq!(
            map.keys()
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[1, 2, 1, 1]
        );
        let values = map.values().as_string::<i32>();
        assert_eq!(values.value(0), "one");
        assert!(values.is_null(1));
        assert_eq!(values.value(2), "dup-old");
        assert_eq!(values.value(3), "dup-new");

        Ok(())
    }

    #[test]
    fn legacy_from_arrow_map_imports_canonical_map() -> VortexResult<()> {
        let arrow = i32_utf8_map_array(
            vec![0, 1, 2],
            vec![1, 2],
            vec![Some("one"), Some("two")],
            None,
            false,
        )?;
        let vortex = ArrayRef::from_arrow(&arrow, false)?;

        assert!(vortex.is::<VortexMap>());
        assert_eq!(
            vortex.dtype(),
            &DType::map(
                DType::Primitive(PType::I32, NonNullable),
                DType::Utf8(Nullable),
                false,
                NonNullable,
            )?
        );

        Ok(())
    }

    #[test]
    fn arrow_rs_map_builder_default_field_names_import() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();
        let mut builder = ArrowMapBuilder::new(None, StringBuilder::new(), Int32Builder::new());

        builder.keys().append_value("joe");
        builder.values().append_value(1);
        builder.append(true)?;
        builder.keys().append_value("blogs");
        builder.values().append_value(2);
        builder.keys().append_value("foo");
        builder.values().append_value(4);
        builder.append(true)?;
        builder.append(true)?;
        builder.append(false)?;

        let arrow = builder.finish();
        let DataType::Map(entries_field, keys_sorted) = arrow.data_type() else {
            panic!("expected Arrow map dtype, got {}", arrow.data_type());
        };
        assert!(!*keys_sorted);
        assert_eq!(entries_field.name(), "entries");
        let DataType::Struct(fields) = entries_field.data_type() else {
            panic!("expected Arrow map entries struct, got {entries_field:?}");
        };
        assert_eq!(fields[0].name(), "keys");
        assert_eq!(fields[1].name(), "values");

        let expected_dtype = DType::map(
            DType::Utf8(NonNullable),
            DType::Primitive(PType::I32, Nullable),
            false,
            Nullable,
        )?;
        let legacy = ArrayRef::from_arrow(&arrow, true)?;
        assert_eq!(legacy.dtype(), &expected_dtype);

        let field = Field::new("maps", arrow.data_type().clone(), true);
        let vortex = session.from_arrow_array(Arc::new(arrow), &field)?;
        assert_eq!(vortex.dtype(), &expected_dtype);
        let row0 = vortex.execute_scalar(0, &mut ctx)?;
        let row0_entries = row0.as_map().entries().collect::<Vec<_>>();
        assert_eq!(row0_entries.len(), 1);
        assert_eq!(row0_entries[0].0, Scalar::utf8("joe", NonNullable));
        assert_eq!(row0_entries[0].1, Scalar::primitive(1_i32, Nullable));
        let row1 = vortex.execute_scalar(1, &mut ctx)?;
        let row1_entries = row1.as_map().entries().collect::<Vec<_>>();
        assert_eq!(row1_entries.len(), 2);
        assert_eq!(row1_entries[0].0, Scalar::utf8("blogs", NonNullable));
        assert_eq!(row1_entries[0].1, Scalar::primitive(2_i32, Nullable));
        assert!(vortex.execute_scalar(2, &mut ctx)?.as_map().is_empty());
        assert!(vortex.execute_scalar(3, &mut ctx)?.is_null());

        Ok(())
    }

    #[test]
    fn arrow_map_import_sets_zero_copy_to_list() -> VortexResult<()> {
        let vortex_session = array_session();
        let session = vortex_session.arrow();
        let field = i32_utf8_map_field(false, false);
        let arrow = i32_utf8_map_array(
            vec![0, 2, 3],
            vec![1, 2, 3],
            vec![Some("one"), Some("two"), Some("three")],
            None,
            false,
        )?;

        let vortex = session.from_arrow_array(Arc::new(arrow), &field)?;
        let map = vortex.as_::<VortexMap>();
        assert!(map.entries().as_::<ListView>().is_zero_copy_to_list());

        let sliced = i32_utf8_map_array(
            vec![0, 1, 3, 4],
            vec![9, 10, 11, 12],
            vec![Some("nine"), Some("ten"), Some("eleven"), Some("twelve")],
            None,
            false,
        )?;
        let sliced: ArrowArrayRef = Arc::new(sliced.slice(1, 2));

        let vortex = session.from_arrow_array(sliced, &field)?;
        let map = vortex.as_::<VortexMap>();
        let entries = map.entries().as_::<ListView>();
        assert!(entries.is_zero_copy_to_list());
        assert_eq!(entries.offset_at(0), 1);
        assert_eq!(entries.offset_at(1), 3);

        Ok(())
    }

    #[test]
    fn sliced_arrow_map_import_preserves_nonzero_offsets() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();
        let field = i32_utf8_map_field(true, false);
        let arrow = i32_utf8_map_array(
            vec![0, 1, 3, 4],
            vec![9, 10, 11, 12],
            vec![Some("nine"), Some("ten"), Some("eleven"), Some("twelve")],
            None,
            true,
        )?;
        let sliced: ArrowArrayRef = Arc::new(arrow.slice(1, 2));

        let vortex = session.from_arrow_array(sliced, &field)?;
        let map = vortex.as_::<VortexMap>();
        let entries = map.entries().as_::<ListView>();
        assert_eq!(entries.offset_at(0), 1);
        assert_eq!(entries.offset_at(1), 3);
        assert_eq!(entries.size_at(0), 2);
        assert_eq!(entries.size_at(1), 1);

        let exported = session.execute_arrow(vortex, Some(&field), &mut ctx)?;
        let map = exported.as_map();

        assert_eq!(map.value_offsets(), &[1, 3, 4]);
        assert_eq!(
            map.value(0)
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[10, 11]
        );
        assert_eq!(
            map.value(1)
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[12]
        );

        Ok(())
    }

    #[test]
    fn filtered_map_with_entry_gap_exports_correct_rows() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();
        let map_dtype = MapDType::try_new(
            DType::Primitive(PType::I32, NonNullable),
            DType::Utf8(Nullable),
            false,
        )?;
        let dtype = DType::Map(map_dtype.clone(), Nullable);
        let mut builder = MapBuilder::<u64, u64>::with_capacity(map_dtype, Nullable, 3);
        let rows: [&[(i32, &str)]; 3] = [
            &[(1, "a"), (2, "b")],
            &[(3, "c"), (4, "d"), (5, "e")],
            &[(6, "f")],
        ];
        for row in rows {
            let entries = row
                .iter()
                .map(|(key, value)| {
                    (
                        Scalar::primitive(*key, NonNullable),
                        Scalar::utf8(*value, Nullable),
                    )
                })
                .collect::<Vec<_>>();
            builder.append_scalar(&Scalar::try_map(dtype.clone(), entries)?)?;
        }
        let source = builder.finish_into_map().into_array();

        // Dropping the non-empty middle row leaves a gap in the entry elements, so the export
        // must rebuild the entries list instead of reusing the builder's contiguous layout.
        let filtered = source.filter(Mask::from_iter([true, false, true]))?;

        let field = i32_utf8_map_field(false, true);
        let exported = session.execute_arrow(filtered, Some(&field), &mut ctx)?;
        let map = exported.as_map();

        assert_eq!(map.value_offsets(), &[0, 2, 3]);
        assert_eq!(
            map.keys()
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[1, 2, 6]
        );
        let values = map.values().as_string::<i32>();
        assert_eq!(values.value(0), "a");
        assert_eq!(values.value(1), "b");
        assert_eq!(values.value(2), "f");

        Ok(())
    }

    #[test]
    fn unsorted_map_rejects_sorted_arrow_target() -> VortexResult<()> {
        let vortex_session = array_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();
        let source_field = i32_utf8_map_field(false, false);
        let target_field = i32_utf8_map_field(true, false);
        let arrow = i32_utf8_map_array(
            vec![0, 2],
            vec![2, 1],
            vec![Some("two"), Some("one")],
            None,
            false,
        )?;
        let vortex = session.from_arrow_array(Arc::new(arrow), &source_field)?;

        let error = session
            .execute_arrow(vortex, Some(&target_field), &mut ctx)
            .unwrap_err();
        assert!(error.to_string().contains("keys_sorted=true"));

        Ok(())
    }

    #[test]
    fn malformed_map_field_errors_without_panic() -> VortexResult<()> {
        let session = array_session();
        let arrow_session = session.arrow();
        let arrow = i32_utf8_map_array(vec![0, 1], vec![1], vec![Some("one")], None, false)?;
        let fields = Fields::from(vec![
            Field::new("key", DataType::Int32, true),
            Field::new("value", DataType::Utf8, true),
        ]);
        let bad_field = Field::new(
            "maps",
            DataType::Map(Arc::new(Field::new_struct("entries", fields, false)), false),
            false,
        );

        let error = arrow_session
            .from_arrow_array(Arc::new(arrow), &bad_field)
            .unwrap_err();
        assert!(error.to_string().contains("key field must be non-nullable"));

        Ok(())
    }
}
