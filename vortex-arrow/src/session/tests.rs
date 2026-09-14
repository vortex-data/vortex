// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use arrow_array::DictionaryArray;
use arrow_array::GenericListArray;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::StringArray;
use arrow_array::types::Int32Type;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use rstest::rstest;
use vortex_array::VortexSessionExecute as _;
use vortex_array::array_session;
use vortex_array::arrays::Dict;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;

use super::*;

#[rstest]
#[case::prefix(0, 2)]
#[case::middle(500, 2)]
#[case::empty(500, 0)]
#[case::empty_at_end(1000, 0)]
fn test_sliced_list_imports_only_its_rows(
    #[values(false, true)] large: bool,
    #[case] start: usize,
    #[case] len: usize,
) -> VortexResult<()> {
    let lists = |rows: Range<i64>| -> ArrowArrayRef {
        let values = Arc::new(Int64Array::from_iter_values(
            rows.flat_map(|row| [row, row + 1]),
        ));
        let len = values.len() / 2;
        let field = Arc::new(Field::new("item", DataType::Int64, false));
        if large {
            Arc::new(GenericListArray::<i64>::new(
                field,
                OffsetBuffer::from_repeated_length(2, len),
                values,
                None,
            ))
        } else {
            Arc::new(GenericListArray::<i32>::new(
                field,
                OffsetBuffer::from_repeated_length(2, len),
                values,
                None,
            ))
        }
    };
    let session = ArrowSession::default();
    let sliced = session.from_arrow_array(lists(0..1000).slice(start, len), false)?;
    let fresh = session.from_arrow_array(lists(start as i64..(start + len) as i64), false)?;
    assert_eq!(sliced.nbytes(), fresh.nbytes());
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(sliced, fresh, &mut ctx);
    Ok(())
}

#[test]
fn from_arrow_fields_matches_schema_conversion() -> VortexResult<()> {
    let session = ArrowSession::default();
    let fields = Fields::from(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8View, true),
    ]);
    let struct_fields = session.from_arrow_fields(&fields)?;
    let schema_dtype = session.from_arrow_schema(&Schema::new(fields))?;
    assert_eq!(
        schema_dtype,
        DType::Struct(struct_fields, Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn from_arrow_field_maps_variant_without_importer() -> VortexResult<()> {
    let session = ArrowSession::default();
    let storage = DataType::Struct(
        vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]
        .into(),
    );
    let field = Field::new("v", storage, true).with_metadata(
        [(
            "ARROW:extension:name".to_string(),
            "arrow.parquet.variant".to_string(),
        )]
        .into(),
    );
    assert_eq!(
        session.from_arrow_field(&field)?,
        DType::Variant(Nullability::Nullable)
    );
    Ok(())
}

/// A plain Arrow dictionary imports as a Vortex `Dict` array over the dictionary values,
/// matching the dtype the schema conversion reports.
#[test]
fn dictionary_imports_as_dict_encoding() -> VortexResult<()> {
    let session = ArrowSession::default();
    let dict: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1, 0, 1]),
        Arc::new(StringArray::from(vec!["a", "b"])),
    )?);
    let field = Field::new("s", dict.data_type().clone(), false);

    let dtype = session.from_arrow_field(&field)?;
    assert_eq!(dtype, DType::Utf8(Nullability::NonNullable));

    let array = session.from_arrow_array(dict, &field)?;
    assert!(array.is::<Dict>(), "expected a Dict encoding, got {array}");
    assert_eq!(array.dtype(), &dtype);
    assert_eq!(array.len(), 4);
    Ok(())
}
