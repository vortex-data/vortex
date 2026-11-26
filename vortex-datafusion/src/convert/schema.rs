// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow schema reconciliation for DataFusion scans.
//!
//! A Vortex dtype omits some Arrow layout choices, so a scan also needs a reference Arrow schema
//! to choose its output fields. Reusing the reference metadata **must** preserve extension identity.
//! This module checks that boundary, leaving schema evolution casts and array conversion to the
//! scan.

use std::fmt;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::Result as DFResult;
use datafusion_common::exec_datafusion_err;
use vortex::dtype::DType;
use vortex_arrow::ArrowSession;

/// Preserve requested Arrow layouts while retaining the data's extension identity.
///
/// `dtype` **must** be a struct. The result contains its columns in order, with their top-level
/// nullability. Fields found by name in `reference_logical_schema` retain its metadata and supported
/// representations, such as string offset widths, list layouts, and dictionary or run-end encoding.
/// Other fields use [`ArrowSession::to_arrow_field`].
///
/// Reference fields **must** preserve the extension identity of the data, including nested fields
/// and encoded values. [`ensure_extension_type_matches`] defines the accepted matches. Run-end
/// wrapper fields and map entries fields must name no extension, since their values carry the
/// identity. Dtypes that the session cannot export also fail.
///
/// A [`Null`](DType::Null) dtype skips extension and encoded-value checks because it has no non-null
/// values. Dictionary and run-end fields over it retain the reference type.
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

/// Reconcile one field under the contract of [`calculate_physical_schema`].
///
/// `parent_path` locates the containing field for error messages, or is `None` at the top level.
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

    // A Null dtype can use any encoded value type because every row is null. Recursing into those
    // values would reject valid targets such as Utf8.
    let is_null_dtype = matches!(dtype, DType::Null);
    if is_null_dtype
        && matches!(
            logical_type,
            DataType::RunEndEncoded(..) | DataType::Dictionary(..)
        )
    {
        return Ok(logical_type.clone());
    }

    // Vortex dtypes omit run-end and dictionary encoding. Check the values before retaining the
    // requested encoding, so an encoded target cannot bypass type checks.
    if let DataType::RunEndEncoded(_, logical_values) = logical_type {
        ensure_no_extension_type(logical_field, "Run-end", &path)?;
        calculate_physical_field_type(dtype, logical_values, Some(&path), arrow_session)?;

        return Ok(logical_type.clone());
    }
    if let DataType::Dictionary(_, value_type) = logical_type {
        // Dictionary values have no Field, so their extension metadata belongs to the dictionary
        // field. Reuse its path too, rather than inventing a child field for errors.
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

    // Inspect storage only for layout compatibility. The session's export above still uses dtype
    // to preserve extension identity and map temporal dtypes to native Arrow types.
    let mut storage_dtype = dtype;
    while let DType::Extension(ext) = storage_dtype {
        storage_dtype = ext.storage_dtype();
    }

    // Variant's exported struct has no corresponding Vortex struct dtype to reconcile.
    if storage_dtype.is_variant() {
        return Ok(exported_field.data_type().clone());
    }

    Ok(match logical_type {
        // Preserve offset-based layouts instead of the session's default view layout.
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
            if storage_dtype.is_binary() || storage_dtype.is_utf8() {
                logical_type.clone()
            } else {
                return Err(exec_datafusion_err!(
                    "Failed to convert dtype to arrow: Vortex DType is {dtype} which is not compatible with {logical_type}"
                ));
            }
        }

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

        // Map child names participate in Arrow type equality, so retain the reference names.
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
        _ => exported_field.data_type().clone(),
    })
}

/// Accept matching extension identities, including alternate serializations of the same dtype.
///
/// `exported_field` **must** be the [`ArrowSession::to_arrow_field`] export of `dtype`. Fields match
/// when both name no extension, or their extension names and metadata are equal. Missing metadata
/// equals empty metadata because Arrow writers use both forms.
///
/// If both fields name extensions, [`ArrowSession::from_arrow_field`] can also establish equality
/// by importing `logical_field` as `dtype`, ignoring nullability. This accepts equivalent metadata
/// spellings and Vortex IDs in place of Arrow names. Temporal dtypes export as native Arrow types,
/// so their fields must name no extension.
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

    // An unknown extension name imports as plain storage. Require names on both fields so that
    // this fallback cannot treat an unknown extension as equivalent to a plain dtype.
    let both_named = matches!(
        (expected, actual),
        (
            FieldExtensionType::Named { .. },
            FieldExtensionType::Named { .. }
        )
    );
    if both_named
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

/// Reject extension names on fields that only group values, such as run-end wrappers.
///
/// `kind` identifies the wrapper in the error message.
fn ensure_no_extension_type(field: &Field, kind: &str, path: &ColumnPath<'_>) -> DFResult<()> {
    match FieldExtensionType::of(field) {
        FieldExtensionType::Plain => Ok(()),
        actual => Err(exec_datafusion_err!(
            "{kind} field {path} must name no Arrow extension type, got {actual}"
        )),
    }
}

/// A field's ancestry for errors such as `col.entries.key`, without allocating a path per field.
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

/// A [`Field`]'s extension identity, borrowed from its `ARROW:extension:*` metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldExtensionType<'a> {
    /// The field names no extension type.
    Plain,
    /// The extension name and metadata, with missing metadata normalized to an empty string.
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
    use arrow_schema::Metadata;
    use arrow_schema::TimeUnit as ArrowTimeUnit;
    use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use rstest::rstest;
    use vortex::array::EmptyMetadata;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::dtype::extension::ExtDType;
    use vortex::dtype::extension::ExtId;
    use vortex::dtype::extension::ExtVTable;
    use vortex::dtype::session::DTypeSessionExt;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::extension::datetime::TimeUnit;
    use vortex::extension::datetime::Timestamp;
    use vortex::scalar::ScalarValue;
    use vortex::session::VortexSession;
    use vortex::session::registry::CachedId;
    use vortex_arrow::ArrowSessionExt;
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
        assert_eq!(physical_schema, logical_schema);
    }

    #[test]
    fn test_schema_metadata_preserved() -> DFResult<()> {
        let logical_schema = Schema::new_with_metadata(
            vec![Field::new("col", DataType::Int32, false)],
            [("table".to_string(), "metadata".to_string())]
                .into_iter()
                .collect::<Metadata>(),
        );
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "col",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );

        let physical_schema =
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())?;
        assert_eq!(physical_schema, logical_schema);
        Ok(())
    }

    #[test]
    fn test_variant_uses_arrow_storage_type() -> anyhow::Result<()> {
        // A Variant has no struct dtype, although its Arrow storage is a struct.
        let session = VortexSession::empty();
        let arrow_session = session.arrow();
        let variant = DType::Variant(Nullability::Nullable);
        let logical_schema = Schema::new(vec![arrow_session.to_arrow_field("v", &variant)?]);
        assert!(matches!(
            logical_schema.field(0).data_type(),
            DataType::Struct(_)
        ));
        let dtype = DType::Struct(
            StructFields::from_iter([("v", variant)]),
            Nullability::NonNullable,
        );

        let physical_schema = calculate_physical_schema(&dtype, &logical_schema, &arrow_session)?;
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
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

        let result =
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow());
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow());
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
        assert_eq!(physical_schema, logical_schema);
    }

    #[test]
    fn test_map_schema_conversion_preserves_reference_fields() {
        let key = Field::new("custom_key", DataType::Int32, false)
            .with_metadata([("key_metadata".to_owned(), "key_value".to_owned())]);
        let value = Field::new("custom_value", DataType::Utf8, true)
            .with_metadata([("value_metadata".to_owned(), "value_value".to_owned())]);
        let entries = Field::new_struct("custom_entries", vec![key, value], false)
            .with_metadata([("entries_metadata".to_owned(), "entries_value".to_owned())]);
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
        assert_eq!(physical_schema, logical_schema);
    }

    #[test]
    fn test_non_struct_dtype_error() {
        // Test that non-struct DType produces an error
        let logical_schema = Schema::new(vec![Field::new("col", DataType::Int32, false)]);

        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let result =
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow());
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())
                .unwrap();
        assert_eq!(physical_schema, logical_schema);
    }

    #[test]
    fn test_list_family_conversion_preserves_element_metadata() -> DFResult<()> {
        let elem = || {
            Field::new("item", DataType::Int32, true)
                .with_metadata([("elem_meta".to_owned(), "v".to_owned())])
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
            calculate_physical_schema(&dtype, &logical_schema, &VortexSession::empty().arrow())?;
        assert_eq!(physical_schema, logical_schema);
        Ok(())
    }

    const ANY_STORAGE: &str = "test.any_storage";

    /// A test extension for exercising each storage layout through the generic Vortex-ID export.
    ///
    /// Its metadata holds the nullable storage dtype, so [`ExtDType`] can set top-level nullability.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct AnyStorage;

    impl ExtVTable for AnyStorage {
        type Metadata = DType;
        type StoredMetadata = ();
        type NativeValue<'a> = &'a ScalarValue;

        fn id(&self) -> ExtId {
            static ID: CachedId = CachedId::new(ANY_STORAGE);
            *ID
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
            Ok(())
        }

        fn metadata_from_storage(
            &self,
            _stored_metadata: Self::StoredMetadata,
            storage_dtype: &DType,
        ) -> VortexResult<Self::Metadata> {
            Ok(storage_dtype.clone())
        }

        fn storage_dtype(metadata: &Self::Metadata) -> DType {
            metadata.clone()
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    /// Reconcile one column with both plugin-based JSON export and generic Vortex-ID export enabled.
    fn single_column_physical_schema(logical_field: &Field, dtype: DType) -> DFResult<Schema> {
        let session = VortexSession::empty();
        // Register the dtype as well as the Arrow plugin so aliases such as vortex.json import.
        session.dtypes().register(Json);
        let arrow_session = session.arrow();
        arrow_session.register_extension(Json);

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

    #[track_caller]
    fn assert_physical_schema_error(logical_field: &Field, dtype: DType, expected: &str) {
        let error = single_column_physical_schema(logical_field, dtype)
            .expect_err("the physical schema must reject the table field");
        assert!(error.to_string().contains(expected), "{error}");
    }

    fn any_storage(storage_dtype: DType) -> DType {
        DType::Extension(
            ExtDType::<AnyStorage>::try_from_storage((), storage_dtype)
                .vortex_expect("AnyStorage accepts every storage dtype")
                .erased(),
        )
    }

    fn json() -> DType {
        DType::Extension(
            ExtDType::<Json>::try_from_storage(EmptyMetadata, utf8())
                .vortex_expect("Json accepts utf8 storage")
                .erased(),
        )
    }

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

    fn with_any_storage(field: Field) -> Field {
        with_extension(field, ANY_STORAGE, None)
    }

    fn any_storage_column(data_type: DataType) -> Field {
        with_any_storage(Field::new("col", data_type, true))
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

    fn map_entries(key: Field, value: Field) -> Field {
        Field::new_struct("entries", vec![key, value], false)
    }

    fn map_column(entries: Field) -> Field {
        Field::new("col", DataType::Map(Arc::new(entries), false), true)
    }

    fn int32_map_dtype() -> DType {
        DType::map(int32(), int32(), false, Nullability::Nullable)
            .vortex_expect("Int32 is a valid map key")
    }

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

    /// Variant's exported storage layout, deliberately missing its extension name.
    fn variant_storage_column() -> Field {
        Field::new_struct(
            "col",
            vec![
                Field::new("metadata", DataType::BinaryView, true),
                Field::new("value", DataType::BinaryView, true),
            ],
            true,
        )
    }

    #[rstest]
    #[case::utf8_storage(any_storage_column(DataType::Utf8), any_storage(utf8()))]
    #[case::list_storage(
        any_storage_column(DataType::List(int32_item())),
        any_storage(DType::list(int32(), Nullability::Nullable))
    )]
    #[case::list_view_storage(
        any_storage_column(DataType::ListView(int32_item())),
        any_storage(DType::list(int32(), Nullability::Nullable))
    )]
    #[case::fixed_size_list_storage(
        any_storage_column(DataType::FixedSizeList(int32_item(), 3)),
        any_storage(DType::FixedSizeList(Arc::new(int32()), 3, Nullability::Nullable))
    )]
    #[case::map_storage(
        with_any_storage(map_column(map_entries(int32_field("key"), int32_field("value")))),
        any_storage(int32_map_dtype())
    )]
    #[case::extension_values_over_extension(
        run_end(with_any_storage(utf8_values())),
        any_storage(utf8())
    )]
    #[case::extension_dictionary_over_extension(
        with_any_storage(utf8_dictionary()),
        any_storage(utf8())
    )]
    // Native Arrow timestamps leave the extension name available for an outer Vortex extension.
    #[case::plain_column_over_temporal(timestamp_column(), timestamp())]
    #[case::extension_over_temporal(with_any_storage(timestamp_column()), any_storage(timestamp()))]
    #[case::no_metadata_over_empty_metadata(
        with_extension(utf8_column(), "arrow.json", None),
        json()
    )]
    #[case::empty_metadata_over_no_metadata(
        with_extension(variant_storage_column(), "arrow.parquet.variant", Some("")),
        DType::Variant(Nullability::Nullable)
    )]
    // The JSON importer accepts both an empty string and {}, although their bytes differ.
    #[case::equivalent_metadata(with_extension(utf8_column(), "arrow.json", Some("{}")), json())]
    #[case::vortex_id_name(with_extension(utf8_column(), "vortex.json", None), json())]
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

    #[rstest]
    #[case::plain_column_over_extension(
        utf8_column(),
        any_storage(utf8()),
        "Field col must name Arrow extension type test.any_storage for Vortex DType test.any_storage[utf8?](utf8?), got no Arrow extension type"
    )]
    #[case::extension_column_over_plain(
        with_any_storage(utf8_column()),
        utf8(),
        "Field col must name no Arrow extension type"
    )]
    #[case::other_extension_name(
        with_extension(utf8_column(), "test.other", None),
        any_storage(utf8()),
        "got Arrow extension type test.other"
    )]
    #[case::invalid_extension_metadata(
        with_extension(utf8_column(), "arrow.json", Some("x")),
        json(),
        r#"got Arrow extension type arrow.json with metadata "x""#
    )]
    #[case::extension_column_over_temporal(
        with_extension(timestamp_column(), "test.other", None),
        timestamp(),
        "Field col must name no Arrow extension type"
    )]
    #[case::plain_struct_over_variant(
        variant_storage_column(),
        DType::Variant(Nullability::Nullable),
        "Field col must name Arrow extension type arrow.parquet.variant"
    )]
    #[case::extension_dictionary_over_plain(
        with_any_storage(utf8_dictionary()),
        utf8(),
        "Field col must name no Arrow extension type"
    )]
    #[case::extension_values_over_plain(
        run_end(with_any_storage(utf8_values())),
        utf8(),
        "Field col.values must name no Arrow extension type"
    )]
    #[case::extension_run_end_over_extension(
        with_any_storage(run_end(with_any_storage(utf8_values()))),
        any_storage(utf8()),
        "Run-end field col must name no Arrow extension type, got Arrow extension type test.any_storage"
    )]
    #[case::extension_element_over_plain(
        Field::new_list("col", with_any_storage(utf8_item()), true),
        DType::list(utf8(), Nullability::Nullable),
        "Field col.item must name no Arrow extension type"
    )]
    #[case::extension_fixed_size_list_element_over_plain(
        Field::new_fixed_size_list("col", with_any_storage(utf8_item()), 3, true),
        DType::FixedSizeList(Arc::new(utf8()), 3, Nullability::Nullable),
        "Field col.item must name no Arrow extension type"
    )]
    #[case::extension_list_view_element_over_plain(
        Field::new(
            "col",
            DataType::ListView(Arc::new(with_any_storage(utf8_item()))),
            true
        ),
        DType::list(utf8(), Nullability::Nullable),
        "Field col.item must name no Arrow extension type"
    )]
    #[case::extension_struct_child_over_plain(
        Field::new_struct(
            "col",
            vec![with_any_storage(Field::new("a", DataType::Utf8, true))],
            true
        ),
        DType::struct_([("a", utf8())], Nullability::Nullable),
        "Field col.a must name no Arrow extension type"
    )]
    #[case::extension_map_key_over_plain(
        map_column(map_entries(with_any_storage(int32_field("key")), int32_field("value"))),
        int32_map_dtype(),
        "Field col.entries.key must name no Arrow extension type"
    )]
    #[case::extension_map_value_over_plain(
        map_column(map_entries(int32_field("key"), with_any_storage(int32_field("value")))),
        int32_map_dtype(),
        "Field col.entries.value must name no Arrow extension type"
    )]
    #[case::extension_map_entries(
        map_column(with_any_storage(map_entries(int32_field("key"), int32_field("value")))),
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

    /// One Arrow field cannot carry both the outer extension name and JSON's extension name.
    #[test]
    fn test_unexportable_dtype_rejected() {
        assert_physical_schema_error(
            &with_any_storage(utf8_column()),
            any_storage(json()),
            "must not be an Arrow extension type, got arrow.json",
        );
    }

    #[rstest]
    #[case::run_end(run_end(utf8_values()))]
    #[case::dictionary(utf8_dictionary())]
    fn test_encoded_values_layout_mismatch_rejected(#[case] logical_field: Field) {
        assert_physical_schema_error(
            &logical_field,
            int32(),
            "Vortex DType is i32 which is not compatible with Utf8",
        );
    }

    #[rstest]
    #[case::extension_column(
        with_any_storage(Field::new("col", DataType::Utf8View, true)),
        DType::Null,
        with_any_storage(Field::new("col", DataType::Null, true))
    )]
    #[case::extension_element(
        Field::new_list(
            "col",
            with_any_storage(Field::new_list_field(DataType::Utf8View, true)),
            true
        ),
        DType::list(DType::Null, Nullability::Nullable),
        Field::new_list(
            "col",
            with_any_storage(Field::new_list_field(DataType::Null, true)),
            true
        )
    )]
    #[case::run_end(run_end(utf8_values()), DType::Null, run_end(utf8_values()))]
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
