// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use vortex::array::arrow::ArrowSession;
use vortex::dtype::DType;

/// Maximum precision that fits in an Arrow `Decimal32`.
const DECIMAL32_MAX_PRECISION: u8 = 9;
/// Maximum precision that fits in an Arrow `Decimal64`.
const DECIMAL64_MAX_PRECISION: u8 = 18;

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
    use_all_decimals: bool,
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
            let field = match logical_field {
                Some(logical_field) => {
                    let arrow_type = calculate_physical_field_type(
                        &field_dtype,
                        logical_field.data_type(),
                        arrow_session,
                    )?;
                    Field::new(name.to_string(), arrow_type, field_dtype.is_nullable())
                        .with_metadata(logical_field.metadata().clone())
                }
                None => arrow_session
                    .to_arrow_field(name.as_ref(), &field_dtype)
                    .map_err(|e| exec_datafusion_err!("Failed to convert dtype to arrow: {e}"))?,
            };
            Ok(maybe_narrow_decimals_field(field, use_all_decimals))
        })
        .collect::<DFResult<Vec<_>>>()?;

    Ok(Schema::new(fields))
}

/// Narrow `Decimal128` fields in `schema` to `Decimal32`/`Decimal64` based on their precision when
/// `use_all_decimals` is set, otherwise return the schema unchanged.
///
/// Vortex always widens decimals to `Decimal128` (or `Decimal256` above precision 38) when
/// converting to Arrow, so engines that can handle the smaller Arrow decimal types opt in via this
/// narrowing pass.
pub(crate) fn maybe_narrow_decimals(schema: Schema, use_all_decimals: bool) -> Schema {
    if !use_all_decimals {
        return schema;
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| maybe_narrow_decimals_field(field.as_ref().clone(), true))
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// Apply decimal narrowing to a single [`Field`] (recursing into nested types) when
/// `use_all_decimals` is set, otherwise return the field unchanged.
fn maybe_narrow_decimals_field(field: Field, use_all_decimals: bool) -> Field {
    if !use_all_decimals {
        return field;
    }
    let narrowed = narrow_decimals_data_type(field.data_type());
    field.with_data_type(narrowed)
}

/// Recursively narrow `Decimal128` Arrow data types to the smallest decimal type that fits their
/// precision, leaving all other types untouched.
fn narrow_decimals_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Decimal128(precision, scale) if *precision <= DECIMAL32_MAX_PRECISION => {
            DataType::Decimal32(*precision, *scale)
        }
        DataType::Decimal128(precision, scale) if *precision <= DECIMAL64_MAX_PRECISION => {
            DataType::Decimal64(*precision, *scale)
        }
        DataType::Struct(fields) => DataType::Struct(narrow_decimals_fields(fields)),
        DataType::List(field) => DataType::List(narrow_decimals_field_ref(field)),
        DataType::LargeList(field) => DataType::LargeList(narrow_decimals_field_ref(field)),
        DataType::ListView(field) => DataType::ListView(narrow_decimals_field_ref(field)),
        DataType::LargeListView(field) => DataType::LargeListView(narrow_decimals_field_ref(field)),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(narrow_decimals_field_ref(field), *size)
        }
        other => other.clone(),
    }
}

fn narrow_decimals_fields(fields: &Fields) -> Fields {
    fields
        .iter()
        .map(|field| narrow_decimals_field_ref(field))
        .collect()
}

fn narrow_decimals_field_ref(field: &Field) -> std::sync::Arc<Field> {
    std::sync::Arc::new(
        field
            .clone()
            .with_data_type(narrow_decimals_data_type(field.data_type())),
    )
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
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

        // RunEndEncoded loses its encoding
        DataType::RunEndEncoded(..) => logical_type.clone(),

        // For struct types, recursively check each field
        DataType::Struct(logical_fields) => {
            if let DType::Struct(struct_dtype, _) = dtype {
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
                                Ok(Field::new(
                                    name.to_string(),
                                    arrow_type,
                                    field_dtype.is_nullable(),
                                )
                                .with_metadata(logical_field.metadata().clone()))
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
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false)
                .unwrap();

        // Should preserve the dictionary type from the logical schema
        assert_eq!(
            physical_schema.field(0).data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
        );
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
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false)
                .unwrap();

        assert_eq!(physical_schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(physical_schema.field(1).data_type(), &DataType::LargeUtf8);
        assert_eq!(physical_schema.field(2).data_type(), &DataType::Binary);
        assert_eq!(physical_schema.field(3).data_type(), &DataType::LargeBinary);
    }

    #[test]
    fn test_failing_conversion_incompatible_types() {
        let logical_schema = Schema::new(vec![Field::new("col", DataType::Utf8, false)]);

        let dtype = DType::Struct(
            StructFields::from_iter([(
                "col",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );

        let result =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not compatible with")
        );

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

        let result =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false);
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
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false)
                .unwrap();

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
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false)
                .unwrap();

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
    fn test_non_struct_dtype_error() {
        // Test that non-struct DType produces an error
        let logical_schema = Schema::new(vec![Field::new("col", DataType::Int32, false)]);

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let result =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected struct dtype")
        );
    }

    #[test]
    fn test_decimal_widening_by_default() {
        use vortex::dtype::DecimalDType;

        // Vortex always widens decimals to Decimal128 when handing them to DataFusion, regardless
        // of their precision, unless `use_all_decimals` is enabled.
        let dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "d32",
                    DType::Decimal(DecimalDType::new(5, 2), Nullability::Nullable),
                ),
                (
                    "d64",
                    DType::Decimal(DecimalDType::new(15, 4), Nullability::Nullable),
                ),
                (
                    "d128",
                    DType::Decimal(DecimalDType::new(30, 4), Nullability::Nullable),
                ),
            ]),
            Nullability::NonNullable,
        );
        // Empty reference schema forces the `to_arrow_field` path for every column.
        let logical_schema = Schema::empty();

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), false)
                .unwrap();

        assert_eq!(
            physical_schema.field(0).data_type(),
            &DataType::Decimal128(5, 2)
        );
        assert_eq!(
            physical_schema.field(1).data_type(),
            &DataType::Decimal128(15, 4)
        );
        assert_eq!(
            physical_schema.field(2).data_type(),
            &DataType::Decimal128(30, 4)
        );
    }

    #[test]
    fn test_decimal_narrowing_when_enabled() {
        use vortex::dtype::DecimalDType;

        // With `use_all_decimals` enabled, decimals are narrowed to the smallest Arrow decimal
        // type that fits their precision.
        let dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "d32",
                    DType::Decimal(DecimalDType::new(5, 2), Nullability::Nullable),
                ),
                (
                    "d64",
                    DType::Decimal(DecimalDType::new(15, 4), Nullability::Nullable),
                ),
                (
                    "d128",
                    DType::Decimal(DecimalDType::new(30, 4), Nullability::Nullable),
                ),
            ]),
            Nullability::NonNullable,
        );
        let logical_schema = Schema::empty();

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), true)
                .unwrap();

        assert_eq!(
            physical_schema.field(0).data_type(),
            &DataType::Decimal32(5, 2)
        );
        assert_eq!(
            physical_schema.field(1).data_type(),
            &DataType::Decimal64(15, 4)
        );
        // Precision above 18 stays as Decimal128.
        assert_eq!(
            physical_schema.field(2).data_type(),
            &DataType::Decimal128(30, 4)
        );
    }

    #[test]
    fn test_decimal_narrowing_nested_struct() {
        use vortex::dtype::DecimalDType;

        // Nested decimals inside a struct should be narrowed too.
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "outer",
                DType::Struct(
                    StructFields::from_iter([(
                        "inner_d32",
                        DType::Decimal(DecimalDType::new(4, 1), Nullability::Nullable),
                    )]),
                    Nullability::Nullable,
                ),
            )]),
            Nullability::NonNullable,
        );
        let logical_schema = Schema::empty();

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default(), true)
                .unwrap();

        let DataType::Struct(inner_fields) = physical_schema.field(0).data_type() else {
            panic!("expected struct");
        };
        assert_eq!(inner_fields[0].data_type(), &DataType::Decimal32(4, 1));
    }
}
