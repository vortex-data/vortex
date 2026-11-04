// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
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
///
/// A physical field takes the metadata of its logical field, including the `ARROW:extension:name`
/// and `ARROW:extension:metadata` entries that name an Arrow extension type. So every logical
/// field, including a nested one, must name the same extension type with the same metadata as the
/// field that [`ArrowSession::to_arrow_field`] exports for the file's dtype, or both must name
/// none. Otherwise the scan would label plain values with an extension type, read extension values
/// as their plain storage, or give the values the parameters of another type, so the schema fails
/// instead. A dtype that cannot be exported fails too. In particular:
///
/// - A missing `ARROW:extension:metadata` entry matches an empty one, since writers differ on
///   whether they write an empty entry for a type without parameters.
/// - When only the metadata differs, the logical field still matches if
///   [`ArrowSession::from_arrow_field`] imports it as the file's dtype, ignoring nullability, since
///   writers can serialize the same parameters differently. A field that fails to import does not
///   match.
/// - A temporal dtype such as [`Timestamp`] exports as a native Arrow type with no extension type,
///   so its logical field must name none.
/// - A dictionary field names the extension type of its values on itself, since Arrow gives the
///   values no field of their own. Its value type must match the file's dtype like the type of any
///   other field.
/// - A run-end field names the extension type of its values on its values field, since the Arrow
///   import and export read it from there (`run_end_values_field` and `export_values` in
///   `vortex-arrow`). So its values field must match the file's dtype instead, and the run-end
///   field itself must name none.
/// - The `entries` field of a map has no Vortex dtype, so no export names an extension type on it,
///   and it must name none.
/// - A [`Null`](DType::Null) dtype has no values to mislabel, so it matches a logical field that
///   names any extension type. The values of a run-end or dictionary field over it are not checked.
///
/// [`Timestamp`]: vortex::extension::datetime::Timestamp
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

    let reference_fields = reference_logical_schema.fields();
    let fields: Vec<Field> = struct_dtype
        .names()
        .iter()
        .zip(struct_dtype.fields())
        .enumerate()
        .map(|(idx, (name, field_dtype))| {
            // Files usually share the table's column order, so try the field at the same position
            // before searching by name, which would make wide schemas quadratic.
            let logical_field = reference_fields
                .get(idx)
                .filter(|field| field.name() == name.as_ref())
                .map(|field| field.as_ref())
                .or_else(|| reference_logical_schema.field_with_name(name.as_ref()).ok());
            match logical_field {
                Some(logical_field) => {
                    let arrow_type = calculate_physical_field_type(
                        &field_dtype,
                        logical_field,
                        None,
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
///
/// Fails when `logical_field` does not match `dtype`, as [`calculate_physical_schema`] describes.
/// `parent_path` is the path of the field that contains `logical_field`, or `None` for a top-level
/// column. Errors name `logical_field` by its path.
fn calculate_physical_field_type(
    dtype: &DType,
    logical_field: &Field,
    parent_path: Option<&ColumnPath<'_>>,
    arrow_session: &ArrowSession,
) -> DFResult<DataType> {
    let path = ColumnPath {
        parent: parent_path,
        name: logical_field.name(),
    };
    let logical_type = logical_field.data_type();

    // A `Null` dtype skips the run-end and dictionary checks below, since the checks of the values
    // would reject it under value types such as `Utf8`.
    let is_null_dtype = matches!(dtype, DType::Null);
    if is_null_dtype
        && matches!(
            logical_type,
            DataType::RunEndEncoded(..) | DataType::Dictionary(..)
        )
    {
        return Ok(logical_type.clone());
    }

    // Run-end and dictionary types lose their encoding, so the executor encodes the values into the
    // logical type, and the checks apply to the values.
    if let DataType::RunEndEncoded(_, logical_values) = logical_type {
        ensure_no_extension_type(logical_field, "Run-end", &path)?;
        calculate_physical_field_type(dtype, logical_values, Some(&path), arrow_session)?;
        return Ok(logical_type.clone());
    }
    if let DataType::Dictionary(_, value_type) = logical_type {
        // Arrow gives dictionary values no field of their own, so they keep the name, metadata, and
        // path of the dictionary field.
        let logical_values = logical_field
            .clone()
            .with_data_type(value_type.as_ref().clone());
        calculate_physical_field_type(dtype, &logical_values, parent_path, arrow_session)?;
        return Ok(logical_type.clone());
    }

    let exported_field = arrow_session
        .to_arrow_field(logical_field.name(), dtype)
        .map_err(|e| exec_datafusion_err!("Failed to convert dtype to arrow: {e}"))?;
    if !is_null_dtype {
        ensure_extension_type_matches(dtype, logical_field, &exported_field, &path, arrow_session)?;
    }

    // An extension column has the Arrow layout of its storage, and the check above matched its
    // extension type, so the layout checks use the storage dtype. Temporal dtypes export as native
    // Arrow types (see `ArrowSession::to_arrow_field`), which only the default arm matches.
    let mut storage_dtype = dtype;
    while let DType::Extension(ext) = storage_dtype {
        storage_dtype = ext.storage_dtype();
    }

    // Check if the logical type is one that doesn't roundtrip through DType
    Ok(match logical_type {
        // Non-view string/binary types become view types after roundtrip
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
            if storage_dtype.is_binary() || storage_dtype.is_utf8() {
                logical_type.clone()
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

        // For struct types, recursively check each field.
        DataType::Struct(logical_fields) => {
            if let DType::Struct(struct_dtype, _) = storage_dtype {
                let physical_fields: Vec<Field> = struct_dtype
                    .names()
                    .iter()
                    .zip(struct_dtype.fields())
                    .map(|(name, field_dtype)| {
                        match logical_fields.iter().find(|f| f.name() == name.as_ref()) {
                            Some(logical_child) => {
                                let arrow_type = calculate_physical_field_type(
                                    &field_dtype,
                                    logical_child,
                                    Some(&path),
                                    arrow_session,
                                )?;
                                Ok(
                                    Field::new(
                                        name.as_ref(),
                                        arrow_type,
                                        field_dtype.is_nullable(),
                                    )
                                    .with_metadata(logical_child.metadata().clone()),
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
            if let DType::List(elem_dtype, _) = storage_dtype {
                let physical_elem_type = calculate_physical_field_type(
                    elem_dtype,
                    logical_elem,
                    Some(&path),
                    arrow_session,
                )?;
                let physical_field = Field::new(
                    logical_elem.name(),
                    physical_elem_type,
                    logical_elem.is_nullable(),
                )
                .with_metadata(logical_elem.metadata().clone());
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
            if let DType::FixedSizeList(elem_dtype, ..) = storage_dtype {
                let physical_elem_type = calculate_physical_field_type(
                    elem_dtype,
                    logical_elem,
                    Some(&path),
                    arrow_session,
                )?;
                let physical_field = Field::new(
                    logical_elem.name(),
                    physical_elem_type,
                    logical_elem.is_nullable(),
                )
                .with_metadata(logical_elem.metadata().clone());
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
            let DType::Map(map_dtype, _) = storage_dtype else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            };
            let entries_path = ColumnPath {
                parent: Some(&path),
                name: logical_entries.name(),
            };
            ensure_no_extension_type(logical_entries, "Map entries", &entries_path)?;
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
                    &logical_fields[0],
                    Some(&entries_path),
                    arrow_session,
                )?,
                false,
            )
            .with_metadata(logical_fields[0].metadata().clone());
            let value = Field::new(
                logical_fields[1].name(),
                calculate_physical_field_type(
                    &map_dtype.value_dtype(),
                    &logical_fields[1],
                    Some(&entries_path),
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
            if let DType::List(elem_dtype, _) = storage_dtype {
                let physical_elem_type = calculate_physical_field_type(
                    elem_dtype,
                    logical_elem,
                    Some(&path),
                    arrow_session,
                )?;
                let physical_field = Field::new(
                    logical_elem.name(),
                    physical_elem_type,
                    logical_elem.is_nullable(),
                )
                .with_metadata(logical_elem.metadata().clone());
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
        _ => exported_field.data_type().clone(),
    })
}

/// Fails when `logical_field` does not name the Arrow extension type of `dtype`, which
/// `exported_field` names, as [`calculate_physical_schema`] describes.
fn ensure_extension_type_matches(
    dtype: &DType,
    logical_field: &Field,
    exported_field: &Field,
    path: &ColumnPath<'_>,
    arrow_session: &ArrowSession,
) -> DFResult<()> {
    let expected = FieldExtensionType::of(exported_field);
    let actual = FieldExtensionType::of(logical_field);
    if actual == expected {
        return Ok(());
    }

    let only_metadata_differs = matches!(
        (expected, actual),
        (
            FieldExtensionType::Named { name: expected_name, .. },
            FieldExtensionType::Named { name: actual_name, .. },
        ) if expected_name == actual_name
    );
    if only_metadata_differs
        && arrow_session
            .from_arrow_field(logical_field)
            .is_ok_and(|imported| imported.eq_ignore_nullability(dtype))
    {
        return Ok(());
    }

    Err(exec_datafusion_err!(
        "Field {path} must name {expected} for Vortex DType {dtype}, got {actual}"
    ))
}

/// Fails when `field` names an Arrow extension type. `kind` names the kind of field, such as
/// `Run-end`, at the start of the error message.
fn ensure_no_extension_type(field: &Field, kind: &str, path: &ColumnPath<'_>) -> DFResult<()> {
    match FieldExtensionType::of(field) {
        FieldExtensionType::Plain => Ok(()),
        actual => Err(exec_datafusion_err!(
            "{kind} field {path} must name no Arrow extension type, got {actual}"
        )),
    }
}

/// The names of the Arrow fields from a top-level column down to a nested field, which error
/// messages join with `.`, such as `col.entries.key`.
struct ColumnPath<'a> {
    /// The path of the field that contains this one, or `None` for a top-level column.
    parent: Option<&'a ColumnPath<'a>>,
    name: &'a str,
}

impl fmt::Display for ColumnPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(parent) = self.parent {
            write!(f, "{parent}.")?;
        }
        f.write_str(self.name)
    }
}

/// The Arrow extension type that a field names in its `ARROW:extension:name` and
/// `ARROW:extension:metadata` entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldExtensionType<'a> {
    /// The field names no extension type.
    Plain,
    /// The field names the extension type `name` with `metadata`, which is empty when the field has
    /// no metadata entry, as [`calculate_physical_schema`] describes.
    Named { name: &'a str, metadata: &'a str },
}

impl<'a> FieldExtensionType<'a> {
    fn of(field: &'a Field) -> Self {
        match field.extension_type_name() {
            Some(name) => Self::Named {
                name,
                metadata: field.extension_type_metadata().unwrap_or_default(),
            },
            None => Self::Plain,
        }
    }
}

impl fmt::Display for FieldExtensionType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plain => f.write_str("no Arrow extension type"),
            Self::Named { name, metadata: "" } => write!(f, "Arrow extension type {name}"),
            Self::Named { name, metadata } => {
                write!(
                    f,
                    "Arrow extension type {name} with metadata \"{metadata}\""
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::FieldRef;
    use arrow_schema::Fields;
    use arrow_schema::TimeUnit as ArrowTimeUnit;
    use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use rstest::rstest;
    use vortex::array::ArrayRef;
    use vortex::array::EmptyMetadata;
    use vortex::array::ExecutionCtx;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::dtype::extension::ExtDType;
    use vortex::dtype::extension::ExtId;
    use vortex::dtype::extension::ExtVTable;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::extension::datetime::TimeUnit;
    use vortex::extension::datetime::Timestamp;
    use vortex::scalar::ScalarValue;
    use vortex::session::registry::CachedId;
    use vortex::session::registry::Id;
    use vortex_arrow::ArrowExport;
    use vortex_arrow::ArrowExportOptions;
    use vortex_arrow::ArrowExportVTable;
    use vortex_json::Json;

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
        assert_eq!(physical_schema, logical_schema);
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
        assert_eq!(physical_schema, logical_schema);
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
        assert_eq!(physical_schema, logical_schema);
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

        let result = calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default());
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
        assert_eq!(physical_schema, logical_schema);
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
        assert_eq!(physical_schema, logical_schema);
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
        assert_eq!(physical_schema, logical_schema);
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
        assert_eq!(physical_schema, logical_schema);
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
        assert_eq!(physical_schema, logical_schema);
    }

    #[test]
    fn test_list_family_conversion_preserves_element_metadata() -> DFResult<()> {
        let elem = || {
            Field::new("item", DataType::Int32, true)
                .with_metadata([("elem_meta".to_owned(), "v".to_owned())].into())
        };
        let logical_schema = Schema::new(vec![
            Field::new("list_col", DataType::List(Arc::new(elem())), true),
            Field::new(
                "fsl_col",
                DataType::FixedSizeList(Arc::new(elem()), 3),
                true,
            ),
            Field::new("listview_col", DataType::ListView(Arc::new(elem())), true),
        ]);
        let i32n = || Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let dtype = DType::Struct(
            StructFields::from_iter([
                ("list_col", DType::List(i32n(), Nullability::Nullable)),
                (
                    "fsl_col",
                    DType::FixedSizeList(i32n(), 3, Nullability::Nullable),
                ),
                ("listview_col", DType::List(i32n(), Nullability::Nullable)),
            ]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &ArrowSession::default())?;
        assert_eq!(physical_schema, logical_schema);
        Ok(())
    }

    /// The Vortex ID of [`AnyStorage`], which its Arrow export also uses as the extension name.
    const ANY_STORAGE: &str = "test.any_storage";

    /// An extension type that accepts every storage dtype, so one type can wrap each layout.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct AnyStorage;

    impl ExtVTable for AnyStorage {
        type Metadata = EmptyMetadata;
        type NativeValue<'a> = &'a ScalarValue;

        fn id(&self) -> ExtId {
            static ID: CachedId = CachedId::new(ANY_STORAGE);
            *ID
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(EmptyMetadata)
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    // The physical schema exports the dtype of every field, so `AnyStorage` needs an exporter.
    impl ArrowExportVTable for AnyStorage {
        fn arrow_ext_id(&self) -> Id {
            self.id()
        }

        fn vortex_id(&self) -> Id {
            self.id()
        }

        fn to_arrow_field(
            &self,
            name: &str,
            dtype: &DType,
            session: &ArrowSession,
        ) -> VortexResult<Option<Field>> {
            let storage_field =
                session.to_arrow_field(name, dtype.as_extension().storage_dtype())?;
            Ok(Some(with_extension(storage_field, ANY_STORAGE, None)))
        }

        fn execute_arrow(
            &self,
            array: ArrayRef,
            _target: &Field,
            _options: &ArrowExportOptions,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrowExport> {
            // The schema tests never execute an array.
            Ok(ArrowExport::Unsupported(array))
        }
    }

    /// Calculates the physical schema of a file whose only column has `dtype`, for a table whose
    /// only field is `logical_field`. The Arrow session exports [`AnyStorage`] but cannot import
    /// it, and it exports and imports [`Json`].
    fn single_column_physical_schema(logical_field: &Field, dtype: DType) -> DFResult<Schema> {
        let arrow_session = ArrowSession::default();
        arrow_session.register_exporter(Arc::new(AnyStorage));
        arrow_session.register_exporter(Arc::new(Json));
        arrow_session.register_importer(Arc::new(Json));

        let file_dtype = DType::struct_(
            [(logical_field.name().as_str(), dtype)],
            Nullability::NonNullable,
        );

        calculate_physical_schema(
            &file_dtype,
            &Schema::new(vec![logical_field.clone()]),
            &arrow_session,
        )
    }

    /// Asserts that the physical schema rejects `logical_field` over `dtype` with an error that
    /// contains `expected`.
    #[track_caller]
    fn assert_physical_schema_error(logical_field: &Field, dtype: DType, expected: &str) {
        let error = single_column_physical_schema(logical_field, dtype)
            .expect_err("the physical schema must reject the table field");
        assert!(error.to_string().contains(expected), "{error}");
    }

    /// Wraps `storage_dtype` in [`AnyStorage`].
    fn any_storage(storage_dtype: DType) -> DType {
        DType::Extension(
            ExtDType::<AnyStorage>::try_new(EmptyMetadata, storage_dtype)
                .vortex_expect("AnyStorage accepts every storage dtype")
                .erased(),
        )
    }

    /// The [`Json`] dtype over nullable `utf8` storage. Its export has an empty metadata entry.
    fn json() -> DType {
        DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, utf8())
                .vortex_expect("Json accepts utf8 storage")
                .erased(),
        )
    }

    /// Names the Arrow extension type `name` on `field`, with `metadata` if it is given.
    fn with_extension(mut field: Field, name: &str, metadata: Option<&str>) -> Field {
        field
            .metadata_mut()
            .insert(EXTENSION_TYPE_NAME_KEY.to_owned(), name.to_owned());
        if let Some(metadata) = metadata {
            field
                .metadata_mut()
                .insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), metadata.to_owned());
        }

        field
    }

    /// A nullable field `col` of `data_type` that names [`AnyStorage`].
    fn any_storage_column(data_type: DataType) -> Field {
        with_extension(Field::new("col", data_type, true), ANY_STORAGE, None)
    }

    fn int32() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn utf8() -> DType {
        DType::Utf8(Nullability::Nullable)
    }

    fn int32_field(name: &str) -> Field {
        Field::new(name, DataType::Int32, false)
    }

    fn int32_item() -> FieldRef {
        Arc::new(int32_field("item"))
    }

    fn utf8_column() -> Field {
        Field::new("col", DataType::Utf8, true)
    }

    fn utf8_item() -> Field {
        Field::new_list_field(DataType::Utf8, true)
    }

    fn utf8_values() -> Field {
        Field::new("values", DataType::Utf8, true)
    }

    fn utf8_dictionary() -> Field {
        Field::new_dictionary("col", DataType::Int32, DataType::Utf8, true)
    }

    /// The entries field `entries` of a map, with the fields `key` and `value`.
    fn map_entries(key: Field, value: Field) -> Field {
        Field::new_struct("entries", vec![key, value], false)
    }

    /// A map field `col` with the entries field `entries`.
    fn map_column(entries: Field) -> Field {
        Field::new("col", DataType::Map(Arc::new(entries), false), true)
    }

    fn int32_map() -> DataType {
        DataType::Map(
            Arc::new(map_entries(int32_field("key"), int32_field("value"))),
            false,
        )
    }

    fn int32_map_dtype() -> DType {
        DType::map(int32(), int32(), false, Nullability::Nullable)
            .vortex_expect("Int32 is a valid map key")
    }

    /// A run-end field `col` with `Int32` run ends and the values field `values`.
    fn run_end(values: Field) -> Field {
        Field::new(
            "col",
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                Arc::new(values),
            ),
            true,
        )
    }

    fn timestamp() -> DType {
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased())
    }

    fn timestamp_column() -> Field {
        Field::new(
            "col",
            DataType::Timestamp(ArrowTimeUnit::Millisecond, None),
            true,
        )
    }

    /// A table field that matches the extension type of the file dtype keeps its type and metadata.
    #[rstest]
    // The layout checks read the storage of an extension column.
    #[case::utf8_storage(any_storage_column(DataType::Utf8), any_storage(utf8()))]
    #[case::list_storage(
        any_storage_column(DataType::List(int32_item())),
        any_storage(DType::list(int32(), Nullability::Nullable))
    )]
    // Vortex has one list dtype, which the table can read as a `ListView` instead of a `List`.
    #[case::list_view_storage(
        any_storage_column(DataType::ListView(int32_item())),
        any_storage(DType::list(int32(), Nullability::Nullable))
    )]
    #[case::fixed_size_list_storage(
        any_storage_column(DataType::FixedSizeList(int32_item(), 3)),
        any_storage(DType::FixedSizeList(Arc::new(int32()), 3, Nullability::Nullable))
    )]
    #[case::map_storage(any_storage_column(int32_map()), any_storage(int32_map_dtype()))]
    // A run-end field names the extension type of its values on its values field.
    #[case::extension_values_over_extension(
        run_end(with_extension(utf8_values(), ANY_STORAGE, None)),
        any_storage(utf8())
    )]
    #[case::extension_values_in_struct(
        Field::new_struct(
            "col",
            vec![run_end(with_extension(utf8_values(), ANY_STORAGE, None)).with_name("a")],
            true
        ),
        DType::struct_([("a", any_storage(utf8()))], Nullability::Nullable)
    )]
    #[case::extension_values_in_list(
        Field::new_list(
            "col",
            run_end(with_extension(utf8_values(), ANY_STORAGE, None)).with_name("item"),
            true
        ),
        DType::list(any_storage(utf8()), Nullability::Nullable)
    )]
    // A dictionary field names the extension type of its values on itself.
    #[case::extension_dictionary_over_extension(
        with_extension(utf8_dictionary(), ANY_STORAGE, None),
        any_storage(utf8())
    )]
    // A field names the outer extension type of an extension stored in another one.
    #[case::extension_over_nested_extension(
        with_extension(utf8_column(), ANY_STORAGE, None),
        any_storage(json())
    )]
    // The export of `AnyStorage` has no metadata entry, which matches an empty one.
    #[case::empty_metadata_over_no_metadata(
        with_extension(utf8_column(), ANY_STORAGE, Some("")),
        any_storage(utf8())
    )]
    // The export of `Json` has an empty metadata entry, which matches a missing one.
    #[case::no_metadata_over_empty_metadata(
        with_extension(utf8_column(), "arrow.json", None),
        json()
    )]
    // `Json` imports the metadata `{}` as the same dtype as its own empty metadata.
    #[case::equivalent_metadata(with_extension(utf8_column(), "arrow.json", Some("{}")), json())]
    // A temporal dtype exports as a native Arrow type with no extension type.
    #[case::plain_column_over_temporal(timestamp_column(), timestamp())]
    fn test_extension_type_match_accepted(
        #[case] logical_field: Field,
        #[case] dtype: DType,
    ) -> DFResult<()> {
        assert_eq!(
            single_column_physical_schema(&logical_field, dtype)?,
            Schema::new(vec![logical_field])
        );

        Ok(())
    }

    /// The physical schema rejects a table field that does not match the extension type of the file
    /// dtype, as [`calculate_physical_schema`] describes, and the error names the field's path.
    #[rstest]
    #[case::plain_column_over_extension(
        utf8_column(),
        any_storage(utf8()),
        "Field col must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got no Arrow extension type"
    )]
    #[case::extension_column_over_plain(
        with_extension(utf8_column(), ANY_STORAGE, None),
        utf8(),
        "Field col must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_column_over_temporal(
        with_extension(timestamp_column(), "test.other", None),
        timestamp(),
        "Field col must name no Arrow extension type for Vortex DType vortex.timestamp[ms](i64?), got Arrow extension type test.other"
    )]
    #[case::other_extension_name(
        with_extension(utf8_column(), "test.other", None),
        any_storage(utf8()),
        "Field col must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got Arrow extension type test.other"
    )]
    // The session cannot import `AnyStorage`, so other metadata cannot import as its dtype.
    #[case::other_extension_metadata(
        with_extension(utf8_column(), ANY_STORAGE, Some("other")),
        any_storage(utf8()),
        "Field col must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got Arrow extension type test.any_storage with metadata \"other\""
    )]
    // `Json` declines to import invalid metadata.
    #[case::invalid_extension_metadata(
        with_extension(utf8_column(), "arrow.json", Some("x")),
        json(),
        "Field col must name Arrow extension type arrow.json for Vortex DType vortex.json(utf8?), got Arrow extension type arrow.json with metadata \"x\""
    )]
    #[case::inner_extension_over_nested_extension(
        with_extension(utf8_column(), "arrow.json", None),
        any_storage(json()),
        "Field col must name Arrow extension type test.any_storage for Vortex DType test.any_storage(vortex.json(utf8?)), got Arrow extension type arrow.json"
    )]
    #[case::plain_dictionary_over_extension(
        utf8_dictionary(),
        any_storage(utf8()),
        "Field col must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got no Arrow extension type"
    )]
    #[case::extension_dictionary_over_plain(
        with_extension(utf8_dictionary(), ANY_STORAGE, None),
        utf8(),
        "Field col must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_element_in_dictionary_over_plain(
        Field::new_dictionary(
            "col",
            DataType::Int32,
            DataType::List(Arc::new(with_extension(utf8_item(), ANY_STORAGE, None))),
            true
        ),
        DType::list(utf8(), Nullability::Nullable),
        "Field col.item must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_values_over_plain(
        run_end(with_extension(utf8_values(), ANY_STORAGE, None)),
        utf8(),
        "Field col.values must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::plain_values_over_extension(
        run_end(utf8_values()),
        any_storage(utf8()),
        "Field col.values must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got no Arrow extension type"
    )]
    #[case::extension_run_end_over_extension(
        with_extension(
            run_end(with_extension(utf8_values(), ANY_STORAGE, None)),
            ANY_STORAGE,
            None
        ),
        any_storage(utf8()),
        "Run-end field col must name no Arrow extension type, got Arrow extension type test.any_storage"
    )]
    #[case::plain_values_in_struct_over_extension(
        Field::new_struct("col", vec![run_end(utf8_values()).with_name("a")], true),
        DType::struct_([("a", any_storage(utf8()))], Nullability::Nullable),
        "Field col.a.values must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got no Arrow extension type"
    )]
    #[case::extension_values_in_list_over_plain(
        Field::new_list(
            "col",
            run_end(with_extension(utf8_values(), ANY_STORAGE, None)).with_name("item"),
            true
        ),
        DType::list(utf8(), Nullability::Nullable),
        "Field col.item.values must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::plain_element_over_extension(
        Field::new_list("col", utf8_item(), true),
        DType::list(any_storage(utf8()), Nullability::Nullable),
        "Field col.item must name Arrow extension type test.any_storage for Vortex DType test.any_storage(utf8?), got no Arrow extension type"
    )]
    #[case::extension_element_over_plain(
        Field::new_list("col", with_extension(utf8_item(), ANY_STORAGE, None), true),
        DType::list(utf8(), Nullability::Nullable),
        "Field col.item must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_fixed_size_list_element_over_plain(
        Field::new_fixed_size_list("col", with_extension(utf8_item(), ANY_STORAGE, None), 3, true),
        DType::FixedSizeList(Arc::new(utf8()), 3, Nullability::Nullable),
        "Field col.item must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_list_view_element_over_plain(
        Field::new(
            "col",
            DataType::ListView(Arc::new(with_extension(utf8_item(), ANY_STORAGE, None))),
            true
        ),
        DType::list(utf8(), Nullability::Nullable),
        "Field col.item must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_struct_child_over_plain(
        Field::new_struct(
            "col",
            vec![with_extension(Field::new("a", DataType::Utf8, true), ANY_STORAGE, None)],
            true
        ),
        DType::struct_([("a", utf8())], Nullability::Nullable),
        "Field col.a must name no Arrow extension type for Vortex DType utf8?, got Arrow extension type test.any_storage"
    )]
    #[case::extension_map_key_over_plain(
        map_column(map_entries(
            with_extension(int32_field("key"), ANY_STORAGE, None),
            int32_field("value")
        )),
        int32_map_dtype(),
        "Field col.entries.key must name no Arrow extension type for Vortex DType i32, got Arrow extension type test.any_storage"
    )]
    #[case::extension_map_value_over_plain(
        map_column(map_entries(
            int32_field("key"),
            with_extension(int32_field("value"), ANY_STORAGE, None)
        )),
        int32_map_dtype(),
        "Field col.entries.value must name no Arrow extension type for Vortex DType i32, got Arrow extension type test.any_storage"
    )]
    #[case::extension_map_entries(
        map_column(with_extension(
            map_entries(int32_field("key"), int32_field("value")),
            ANY_STORAGE,
            None
        )),
        int32_map_dtype(),
        "Map entries field col.entries must name no Arrow extension type, got Arrow extension type test.any_storage"
    )]
    fn test_extension_type_mismatch_rejected(
        #[case] logical_field: Field,
        #[case] dtype: DType,
        #[case] expected: &str,
    ) {
        assert_physical_schema_error(&logical_field, dtype, expected);
    }

    /// The values of a run-end or dictionary field get the layout checks of any field, since the
    /// executor encodes the values into the logical type.
    #[rstest]
    #[case::run_end(run_end(utf8_values()))]
    #[case::dictionary(utf8_dictionary())]
    fn test_encoded_values_layout_mismatch_rejected(#[case] logical_field: Field) {
        assert_physical_schema_error(
            &logical_field,
            int32(),
            "Failed to convert dtype to arrow: Vortex DType is i32 which is not compatible with Utf8",
        );
    }

    /// The physical schema exports the dtype of every field, so a dtype that the session cannot
    /// export fails, even under a table field whose layout matches its storage.
    #[test]
    fn test_unexportable_dtype_rejected() {
        let file_dtype = DType::struct_([("col", any_storage(utf8()))], Nullability::NonNullable);
        let logical_schema = Schema::new(vec![utf8_column()]);

        // The default session has no exporter for `AnyStorage`.
        let error =
            calculate_physical_schema(&file_dtype, &logical_schema, &ArrowSession::default())
                .expect_err("a dtype that the session cannot export must fail");
        assert!(
            error.to_string().contains(
                "Failed to convert dtype to arrow: extension type cannot be converted to Arrow without a plugin: test.any_storage(utf8?)"
            ),
            "{error}"
        );
    }

    /// A `Null` file column has no values to mislabel, so it matches a table field that names any
    /// extension type, with the physical type `Null` and the table field's metadata. A run-end or
    /// dictionary field over it keeps its type, which the executor fills with nulls.
    #[rstest]
    #[case::extension_column(
        with_extension(Field::new("col", DataType::Utf8View, true), ANY_STORAGE, None),
        DType::Null,
        with_extension(Field::new("col", DataType::Null, true), ANY_STORAGE, None)
    )]
    #[case::extension_element(
        Field::new_list(
            "col",
            with_extension(Field::new_list_field(DataType::Utf8View, true), ANY_STORAGE, None),
            true
        ),
        DType::list(DType::Null, Nullability::Nullable),
        Field::new_list(
            "col",
            with_extension(Field::new_list_field(DataType::Null, true), ANY_STORAGE, None),
            true
        )
    )]
    #[case::run_end(run_end(utf8_values()), DType::Null, run_end(utf8_values()))]
    #[case::run_end_in_struct(
        Field::new_struct("col", vec![run_end(utf8_values()).with_name("a")], true),
        DType::struct_([("a", DType::Null)], Nullability::Nullable),
        Field::new_struct("col", vec![run_end(utf8_values()).with_name("a")], true)
    )]
    #[case::dictionary(utf8_dictionary(), DType::Null, utf8_dictionary())]
    fn test_null_dtype_accepted(
        #[case] logical_field: Field,
        #[case] dtype: DType,
        #[case] physical_field: Field,
    ) -> DFResult<()> {
        assert_eq!(
            single_column_physical_schema(&logical_field, dtype)?,
            Schema::new(vec![physical_field])
        );

        Ok(())
    }
}
