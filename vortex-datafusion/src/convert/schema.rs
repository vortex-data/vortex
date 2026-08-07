// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use vortex::dtype::DType;
use vortex_arrow::ArrowSession;

/// Calculate the physical Arrow schema for a Vortex file given its DType and the expected logical schema.
///
/// Some Arrow types don't roundtrip cleanly through Vortex's DType system:
/// - Dictionary types lose their encoding (become the value type)
/// - Utf8/LargeUtf8 become Utf8View
/// - Binary/LargeBinary become BinaryView
/// - RunEndEncoded loses its encoding
/// - Lists are even more complex, with various sizes and physical layouts that are lost
///
/// For these types, we use the logical schema's type instead of the DType's natural Arrow
/// conversion, since Vortex's Arrow executor can produce these types when requested.
pub fn calculate_physical_schema(
    dtype: &DType,
    reference_logical_schema: &Schema,
    arrow_session: &ArrowSession,
) -> DFResult<Schema> {
    let DType::Struct(struct_dtype, _) = dtype else {
        return Err(exec_datafusion_err!(
            "Expected struct dtype for schema conversion"
        ));
    };

    let fields: Vec<Field> = struct_dtype
        .names()
        .iter()
        .zip(struct_dtype.fields())
        .map(|(name, field_dtype)| {
            let logical_field = reference_logical_schema.field_with_name(name.as_ref()).ok();
            match logical_field {
                Some(logical_field) => {
                    let arrow_type = calculate_physical_field_type(
                        &field_dtype,
                        logical_field.data_type(),
                        arrow_session,
                    )?;
                    Ok(
                        Field::new(name.as_ref(), arrow_type, field_dtype.is_nullable())
                            .with_metadata(logical_field.metadata().clone()),
                    )
                }
                None => arrow_session
                    .to_arrow_field(name.as_ref(), &field_dtype)
                    .map_err(|e| exec_datafusion_err!("Failed to convert dtype to arrow: {e}")),
            }
        })
        .collect::<DFResult<Vec<_>>>()?;

    Ok(Schema::new_with_metadata(
        fields,
        reference_logical_schema.metadata().clone(),
    ))
}

/// Calculate the physical Arrow type for a field, preferring the logical type when the
/// DType doesn't roundtrip cleanly.
fn calculate_physical_field_type(
    dtype: &DType,
    logical_type: &DataType,
    arrow_session: &ArrowSession,
) -> DFResult<DataType> {
    // Check if the logical type is one that doesn't roundtrip through DType
    Ok(match logical_type {
        // Dictionary types lose their encoding when converted to DType
        DataType::Dictionary(..) => logical_type.clone(),

        // Non-view string/binary types become view types after roundtrip
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
            if dtype.is_binary() || dtype.is_utf8() {
                logical_type.clone()
            } else if matches!(logical_type, DataType::Utf8 | DataType::LargeUtf8)
                && (dtype.is_int() || dtype.is_float() || dtype.is_boolean())
            {
                // Preserve the file's physical type so the expression adapter can insert the
                // cast to the unified logical string type.
                arrow_session
                    .to_arrow_field("", dtype)
                    .map_err(|e| exec_datafusion_err!("Failed to convert dtype to arrow: {e}"))?
                    .data_type()
                    .clone()
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

        // RunEndEncoded loses its encoding
        DataType::RunEndEncoded(..) => logical_type.clone(),

        // For struct types, recursively check each field.
        DataType::Struct(logical_fields) => {
            // Walk through any extension layers to reach the underlying struct fields.
            let mut inner = dtype;
            while let DType::Extension(ext) = inner {
                inner = ext.storage_dtype();
            }
            if let DType::Struct(struct_dtype, _) = inner {
                let physical_fields: Vec<Field> = struct_dtype
                    .names()
                    .iter()
                    .zip(struct_dtype.fields())
                    .map(|(name, field_dtype)| {
                        match logical_fields.iter().find(|f| f.name() == name.as_ref()) {
                            Some(logical_field) => {
                                let arrow_type = calculate_physical_field_type(
                                    &field_dtype,
                                    logical_field.data_type(),
                                    arrow_session,
                                )?;
                                Ok(
                                    Field::new(
                                        name.as_ref(),
                                        arrow_type,
                                        field_dtype.is_nullable(),
                                    )
                                    .with_metadata(logical_field.metadata().clone()),
                                )
                            }
                            None => arrow_session
                                .to_arrow_field(name.as_ref(), &field_dtype)
                                .map_err(|e| {
                                    exec_datafusion_err!("Failed to convert dtype to arrow: {e}")
                                }),
                        }
                    })
                    .collect::<DFResult<Vec<_>>>()?;

                DataType::Struct(physical_fields.into())
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

        // For list types, recursively check the element type
        DataType::List(logical_elem) | DataType::LargeList(logical_elem) => {
            if let DType::List(elem_dtype, _) = dtype {
                let physical_elem_type = calculate_physical_field_type(
                    elem_dtype,
                    logical_elem.data_type(),
                    arrow_session,
                )?;
                let physical_field = Field::new(
                    logical_elem.name(),
                    physical_elem_type,
                    logical_elem.is_nullable(),
                );
                match logical_type {
                    DataType::List(_) => DataType::List(physical_field.into()),
                    DataType::LargeList(_) => DataType::LargeList(physical_field.into()),
                    _ => unreachable!(),
                }
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

        // For fixed-size list types, recursively check the element type
        DataType::FixedSizeList(logical_elem, size) => {
            if let DType::FixedSizeList(elem_dtype, ..) = dtype {
                let physical_elem_type = calculate_physical_field_type(
                    elem_dtype,
                    logical_elem.data_type(),
                    arrow_session,
                )?;
                let physical_field = Field::new(
                    logical_elem.name(),
                    physical_elem_type,
                    logical_elem.is_nullable(),
                );
                DataType::FixedSizeList(physical_field.into(), *size)
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

        // Map field names and child metadata come from the reference schema, while child
        // types are recursively reconciled so nested extension metadata is preserved.
        DataType::Map(logical_entries, keys_sorted) => {
            let DType::Map(map_dtype, _) = dtype else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            };
            let DataType::Struct(logical_fields) = logical_entries.data_type() else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Arrow Map entries must be a Struct, got {:?}",
                    logical_entries.data_type()
                ));
            };
            if logical_fields.len() != 2 {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Arrow Map entries must contain exactly two fields"
                ));
            }

            let key = Field::new(
                logical_fields[0].name(),
                calculate_physical_field_type(
                    &map_dtype.key_dtype(),
                    logical_fields[0].data_type(),
                    arrow_session,
                )?,
                false,
            )
            .with_metadata(logical_fields[0].metadata().clone());
            let value = Field::new(
                logical_fields[1].name(),
                calculate_physical_field_type(
                    &map_dtype.value_dtype(),
                    logical_fields[1].data_type(),
                    arrow_session,
                )?,
                logical_fields[1].is_nullable(),
            )
            .with_metadata(logical_fields[1].metadata().clone());
            let entries = Field::new_struct(logical_entries.name(), vec![key, value], false)
                .with_metadata(logical_entries.metadata().clone());

            DataType::Map(Arc::new(entries), *keys_sorted)
        }

        // For list view types, recursively check the element type
        DataType::ListView(logical_elem) | DataType::LargeListView(logical_elem) => {
            if let DType::List(elem_dtype, _) = dtype {
                let physical_elem_type = calculate_physical_field_type(
                    elem_dtype,
                    logical_elem.data_type(),
                    arrow_session,
                )?;
                let physical_field = Field::new(
                    logical_elem.name(),
                    physical_elem_type,
                    logical_elem.is_nullable(),
                );
                match logical_type {
                    DataType::ListView(_) => DataType::ListView(physical_field.into()),
                    DataType::LargeListView(_) => DataType::LargeListView(physical_field.into()),
                    _ => unreachable!(),
                }
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }
        // All other types roundtrip cleanly, use the session-aware Arrow Field inference
        // (canonical for non-extension dtypes, plugin-routed for extensions like UUID).
        _ => arrow_session
            .to_arrow_field("", dtype)
            .map_err(|e| exec_datafusion_err!("Failed to convert dtype to arrow: {e}"))?
            .data_type()
            .clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::Fields;
    use rstest::rstest;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;

    use super::*;

    #[test]
    fn test_dict_conversion() {
        // Dictionary types lose their encoding when converted to DType, but we should
        // preserve the original logical type in the physical schema.
        let logical_schema = Schema::new(vec![Field::new(
            "dict_col",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )]);

        // Vortex DType for dictionary is just the value type (Utf8)
        let dtype = DType::Struct(
            StructFields::from_iter([("dict_col", DType::Utf8(Nullability::Nullable))]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();

        // Should preserve the dictionary type from the logical schema
        assert_eq!(
            physical_schema.field(0).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        );
    }

    #[test]
    fn test_schema_metadata_preserved() -> DFResult<()> {
        let logical_schema = Schema::new_with_metadata(
            vec![Field::new("col", DataType::Int32, false)],
            [("table".to_string(), "metadata".to_string())]
                .into_iter()
                .collect(),
        );
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "col",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default())?;

        assert_eq!(
            physical_schema.metadata().get("table"),
            Some(&"metadata".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_utf8_variants_preserved() {
        // Non-view string types become view types after roundtrip through DType,
        // but we should preserve the original logical type.
        let logical_schema = Schema::new(vec![
            Field::new("utf8_col", DataType::Utf8, false),
            Field::new("large_utf8_col", DataType::LargeUtf8, true),
            Field::new("binary_col", DataType::Binary, false),
            Field::new("large_binary_col", DataType::LargeBinary, true),
        ]);

        let dtype = DType::Struct(
            StructFields::from_iter([
                ("utf8_col", DType::Utf8(Nullability::NonNullable)),
                ("large_utf8_col", DType::Utf8(Nullability::Nullable)),
                ("binary_col", DType::Binary(Nullability::NonNullable)),
                ("large_binary_col", DType::Binary(Nullability::Nullable)),
            ]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();

        assert_eq!(physical_schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(physical_schema.field(1).data_type(), &DataType::LargeUtf8);
        assert_eq!(physical_schema.field(2).data_type(), &DataType::Binary);
        assert_eq!(physical_schema.field(3).data_type(), &DataType::LargeBinary);
    }

    #[rstest]
    #[case(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DataType::Utf8,
        DataType::Int32
    )]
    #[case(
        DType::Primitive(PType::F64, Nullability::Nullable),
        DataType::Utf8,
        DataType::Float64
    )]
    #[case(
        DType::Bool(Nullability::NonNullable),
        DataType::Utf8,
        DataType::Boolean
    )]
    #[case(
        DType::Primitive(PType::I64, Nullability::Nullable),
        DataType::LargeUtf8,
        DataType::Int64
    )]
    fn test_bool_and_numeric_file_column_under_utf8_logical_type(
        #[case] physical_dtype: DType,
        #[case] logical_type: DataType,
        #[case] expected_physical_type: DataType,
    ) -> DFResult<()> {
        let logical_schema = Schema::new(vec![Field::new("col", logical_type, true)]);
        let dtype = DType::Struct(
            StructFields::from_iter([("col", physical_dtype)]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default())?;

        assert_eq!(
            physical_schema.field(0).data_type(),
            &expected_physical_type
        );
        Ok(())
    }

    #[rstest]
    #[case(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DataType::Binary
    )]
    #[case(DType::Bool(Nullability::NonNullable), DataType::LargeBinary)]
    #[case(
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        ),
        DataType::Utf8
    )]
    fn test_incompatible_file_column_under_string_or_binary_logical_type(
        #[case] physical_dtype: DType,
        #[case] logical_type: DataType,
    ) {
        let logical_schema = Schema::new(vec![Field::new("col", logical_type, false)]);
        let dtype = DType::Struct(
            StructFields::from_iter([("col", physical_dtype)]),
            Nullability::NonNullable,
        );

        let result = calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default());

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not compatible with")
        );
    }

    #[test]
    fn test_failing_conversion_incompatible_types() {
        // Test struct vs non-struct mismatch
        let logical_schema = Schema::new(vec![Field::new(
            "col",
            DataType::Struct(Fields::empty()),
            false,
        )]);

        let dtype = DType::Struct(
            StructFields::from_iter([("col", DType::Utf8(Nullability::NonNullable))]),
            Nullability::NonNullable,
        );

        let result = calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not compatible with")
        );
    }

    #[test]
    fn test_nested_struct_conversion() {
        let logical_schema = Schema::new(vec![
            Field::new(
                "outer_col",
                DataType::Struct(Fields::from(vec![
                    Field::new("inner_utf8", DataType::Utf8, false),
                    Field::new("inner_int", DataType::Int64, true),
                ])),
                true,
            ),
            Field::new("simple_col", DataType::Int32, false),
        ]);

        let dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "outer_col",
                    DType::Struct(
                        StructFields::from_iter([
                            ("inner_utf8", DType::Utf8(Nullability::NonNullable)),
                            (
                                "inner_int",
                                DType::Primitive(PType::I64, Nullability::Nullable),
                            ),
                        ]),
                        Nullability::Nullable,
                    ),
                ),
                (
                    "simple_col",
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                ),
            ]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();

        // Check outer structure
        assert_eq!(physical_schema.fields().len(), 2);

        // Check nested struct preserves Utf8 (not Utf8View)
        let outer_field = physical_schema.field(0);
        if let DataType::Struct(inner_fields) = outer_field.data_type() {
            assert_eq!(inner_fields.len(), 2);
            assert_eq!(inner_fields[0].data_type(), &DataType::Utf8);
            assert_eq!(inner_fields[1].data_type(), &DataType::Int64);
        } else {
            panic!("Expected struct type for outer_col");
        }
    }

    #[test]
    fn test_list_with_dict_elements() {
        // Test that list types with dictionary elements preserve the dictionary type
        let inner_field = Field::new(
            "item",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        );
        let logical_schema = Schema::new(vec![Field::new(
            "list_col",
            DataType::List(Arc::new(inner_field)),
            true,
        )]);

        let dtype = DType::Struct(
            StructFields::from_iter([(
                "list_col",
                DType::List(
                    Arc::new(DType::Utf8(Nullability::Nullable)),
                    Nullability::Nullable,
                ),
            )]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();

        if let DataType::List(elem_field) = physical_schema.field(0).data_type() {
            assert_eq!(
                elem_field.data_type(),
                &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
            );
        } else {
            panic!("Expected list type");
        }
    }

    #[test]
    fn test_map_schema_conversion_preserves_reference_fields() {
        let key = Field::new("custom_key", DataType::Int32, false)
            .with_metadata([("key_metadata".to_owned(), "key_value".to_owned())].into());
        let value = Field::new("custom_value", DataType::Utf8, true)
            .with_metadata([("value_metadata".to_owned(), "value_value".to_owned())].into());
        let entries = Field::new_struct("custom_entries", vec![key, value], false)
            .with_metadata([("entries_metadata".to_owned(), "entries_value".to_owned())].into());
        let logical_schema = Schema::new(vec![Field::new(
            "map_col",
            DataType::Map(Arc::new(entries), true),
            true,
        )]);
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "map_col",
                DType::map(
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                    true,
                    Nullability::Nullable,
                )
                .unwrap(),
            )]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();
        let field = physical_schema.field(0);
        assert!(field.is_nullable());
        let DataType::Map(entries, keys_sorted) = field.data_type() else {
            panic!("expected Map type, got {:?}", field.data_type());
        };
        assert!(*keys_sorted);
        assert_eq!(entries.name(), "custom_entries");
        assert_eq!(
            entries.metadata().get("entries_metadata"),
            Some(&"entries_value".to_owned())
        );
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected map entries struct, got {:?}", entries.data_type());
        };
        assert_eq!(fields[0].name(), "custom_key");
        assert_eq!(fields[0].data_type(), &DataType::Int32);
        assert!(!fields[0].is_nullable());
        assert_eq!(
            fields[0].metadata().get("key_metadata"),
            Some(&"key_value".to_owned())
        );
        assert_eq!(fields[1].name(), "custom_value");
        assert_eq!(fields[1].data_type(), &DataType::Utf8);
        assert!(fields[1].is_nullable());
        assert_eq!(
            fields[1].metadata().get("value_metadata"),
            Some(&"value_value".to_owned())
        );
    }

    #[test]
    fn test_non_struct_dtype_error() {
        // Test that non-struct DType produces an error
        let logical_schema = Schema::new(vec![Field::new("col", DataType::Int32, false)]);

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let result = calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected struct dtype")
        );
    }

    /// Names carrying raw control bytes must reach Arrow byte-for-byte.
    /// `FieldName`'s `Display` escapes via `escape_debug`, so building the
    /// field with `to_string` renames `\x08` to the literal five characters
    /// `\u{8}`. The scanned batch then disagrees with the table schema the
    /// scan was planned against, and the query fails with "column types must
    /// match schema types".
    #[test]
    fn test_control_byte_column_names_are_not_escaped() {
        let column_names = ["plain", "\u{8}", "check_id\u{10}"];
        let logical_schema = Schema::new(
            column_names
                .iter()
                .map(|n| Field::new(*n, DataType::Utf8, true))
                .collect::<Fields>(),
        );

        let dtype = DType::Struct(
            StructFields::from_iter(
                column_names
                    .iter()
                    .map(|n| (*n, DType::Utf8(Nullability::Nullable))),
            ),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();

        let names: Vec<&str> = physical_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, column_names);
    }

    /// Same, one level down: struct children are built at a separate call
    /// site in `calculate_physical_field_type`. The children here are
    /// run-end-encoded dictionaries, so this also covers the branches that
    /// take the type from the reference schema instead of the `DType`.
    #[test]
    fn test_control_byte_struct_field_names_are_not_escaped() {
        let label_names = ["app", "\u{8}", "check_id\u{10}"];
        let ree = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(Field::new(
                "values",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            )),
        );
        let logical_schema = Schema::new(vec![Field::new_struct(
            "labels",
            label_names
                .iter()
                .map(|n| Field::new(*n, ree.clone(), true))
                .collect::<Fields>(),
            false,
        )]);

        let labels_dtype = DType::Struct(
            StructFields::from_iter(
                label_names
                    .iter()
                    .map(|n| (*n, DType::Utf8(Nullability::Nullable))),
            ),
            Nullability::NonNullable,
        );
        let dtype = DType::Struct(
            StructFields::from_iter([("labels", labels_dtype)]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default()).unwrap();

        let DataType::Struct(labels) = physical_schema.field(0).data_type() else {
            panic!("expected labels to be a struct");
        };
        let names: Vec<&str> = labels.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, label_names);
    }
}
