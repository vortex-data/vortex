// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BinaryViewArray;
use arrow_array::Int64Array;
use arrow_array::LargeStringArray;
use arrow_array::StringArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Metadata;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypePlugin;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::PValue;
use vortex_array::scalar::ScalarValue;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_edition::UNSTABLE;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ARROW_FILE_EXTENSION_NAME;
use crate::FileRef;
use crate::FileRefField;
use crate::FileRefMetadata;
use crate::FileRefValue;
use crate::initialize;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    initialize(&session);
    session
});

/// A file reference dtype over `fields`.
fn file_ref_dtype(fields: &[FileRefField], nullability: Nullability) -> VortexResult<DType> {
    Ok(DType::Extension(
        ExtDType::<FileRef>::try_new(FileRefMetadata::try_new(fields.iter().copied())?, nullability)?
            .erased(),
    ))
}

/// A struct storage dtype with the given field names and dtypes.
fn struct_dtype(fields: &[(&str, DType)]) -> DType {
    DType::Struct(
        StructFields::new(
            fields.iter().map(|(name, _)| *name).collect::<FieldNames>(),
            fields.iter().map(|(_, dtype)| dtype.clone()).collect(),
        ),
        Nullability::NonNullable,
    )
}

#[rstest]
#[case::empty(&[], "at least one field")]
#[case::duplicate(&[FileRefField::Uri, FileRefField::Size, FileRefField::Uri], "uri more than once")]
fn metadata_rejects_invalid_field_sets(#[case] fields: &[FileRefField], #[case] expected: &str) {
    let error = FileRefMetadata::try_new(fields.iter().copied())
        .err()
        .map(|error| error.to_string());
    assert!(
        error.as_deref().is_some_and(|error| error.contains(expected)),
        "{error:?}"
    );
}

/// The storage records the defined fields and their order, so reading a dtype back from its
/// storage and empty metadata recovers it.
#[rstest]
#[case::all(&FileRefField::ALL)]
#[case::inline_only(&[FileRefField::Inline, FileRefField::ContentType])]
#[case::reordered(&[FileRefField::Checksum, FileRefField::Uri])]
fn dtype_roundtrips_through_storage(#[case] fields: &[FileRefField]) -> VortexResult<()> {
    let dtype = file_ref_dtype(fields, Nullability::Nullable)?;
    let ext = dtype.as_extension();
    let metadata = ext.serialize_metadata()?;
    assert!(metadata.is_empty());
    let read = FileRef.deserialize(&metadata, ext.storage_dtype().clone())?;
    assert_eq!(DType::Extension(read), dtype);
    Ok(())
}

#[rstest]
#[case::not_a_struct(DType::Utf8(Nullability::Nullable), "must be a struct")]
#[case::unknown_field(
    struct_dtype(&[("url", DType::Utf8(Nullability::Nullable))]),
    "unknown file reference field \"url\""
)]
#[case::wrong_case(
    struct_dtype(&[("URI", DType::Utf8(Nullability::Nullable))]),
    "unknown file reference field \"URI\""
)]
#[case::no_fields(struct_dtype(&[]), "at least one field")]
#[case::wrong_type(
    struct_dtype(&[("size", DType::Primitive(PType::I32, Nullability::Nullable))]),
    "vortex.file_ref storage must be"
)]
#[case::non_nullable_field(
    struct_dtype(&[("uri", DType::Utf8(Nullability::NonNullable))]),
    "vortex.file_ref storage must be"
)]
fn read_rejects_other_storage(#[case] storage: DType, #[case] expected: &str) {
    let error = FileRef
        .deserialize(&[], storage)
        .err()
        .map(|error| error.to_string());
    assert!(
        error.as_deref().is_some_and(|error| error.contains(expected)),
        "{error:?}"
    );
}

#[rstest]
#[case::uri(FileRefValue { uri: Some("s3://b/k"), ..Default::default() }, None)]
#[case::uri_size(FileRefValue { uri: Some("a.png"), size: Some(0), ..Default::default() }, None)]
#[case::uri_range(
    FileRefValue { uri: Some("a"), offset: Some(4), size: Some(8), ..Default::default() },
    None
)]
#[case::inline(FileRefValue { inline: Some(b""), ..Default::default() }, None)]
#[case::inline_with_locator(
    FileRefValue { inline: Some(b"x"), uri: Some("a"), size: Some(1), ..Default::default() },
    None
)]
#[case::nothing(FileRefValue::default(), Some("must set uri or inline"))]
#[case::empty_uri(
    FileRefValue { uri: Some(""), ..Default::default() },
    Some("must set uri or inline")
)]
#[case::size_only(
    FileRefValue { size: Some(1), ..Default::default() },
    Some("must set uri or inline")
)]
#[case::offset_without_uri(
    FileRefValue { inline: Some(b"x"), offset: Some(0), size: Some(1), ..Default::default() },
    Some("offset requires a uri")
)]
#[case::offset_without_size(
    FileRefValue { uri: Some("a"), offset: Some(0), ..Default::default() },
    Some("offset requires a size")
)]
#[case::negative_offset(
    FileRefValue { uri: Some("a"), offset: Some(-1), size: Some(1), ..Default::default() },
    Some("offset must not be negative")
)]
#[case::negative_size(
    FileRefValue { uri: Some("a"), size: Some(-1), ..Default::default() },
    Some("size must not be negative")
)]
fn values_follow_the_parquet_rules(
    #[case] value: FileRefValue<'_>,
    #[case] expected_error: Option<&str>,
) {
    let error = value.validate().err().map(|error| error.to_string());
    match expected_error {
        None => assert_eq!(error, None),
        Some(expected) => assert!(
            error.as_deref().is_some_and(|error| error.contains(expected)),
            "{error:?}"
        ),
    }
}

/// Storage values unpack by the column's field order and must follow the value rules.
#[test]
fn unpacks_storage_values() -> VortexResult<()> {
    let metadata = FileRefMetadata::try_new([
        FileRefField::Size,
        FileRefField::Uri,
        FileRefField::Offset,
        FileRefField::Inline,
    ])?;
    let ext_dtype = ExtDType::<FileRef>::try_new(metadata, Nullability::NonNullable)?;
    let storage = |uri: Option<&str>, offset: Option<i64>| {
        ScalarValue::Tuple(vec![
            Some(ScalarValue::Primitive(PValue::I64(8))),
            uri.map(|uri| ScalarValue::Utf8(uri.into())),
            offset.map(|offset| ScalarValue::Primitive(PValue::I64(offset))),
            None,
        ])
    };

    let value = storage(Some("s3://bucket/key"), Some(4));
    assert_eq!(
        FileRef::unpack_native(&ext_dtype, &value)?,
        FileRefValue {
            uri: Some("s3://bucket/key"),
            offset: Some(4),
            size: Some(8),
            ..Default::default()
        }
    );
    assert!(
        ext_dtype
            .validate_scalar_value(&storage(None, Some(4)))
            .is_err()
    );
    Ok(())
}

/// A field that names `arrow.parquet.file` over a struct of `children`, with `metadata` under the
/// metadata key when it is given.
fn arrow_file_field(children: Vec<Field>, metadata: Option<&str>) -> Field {
    let name = (
        EXTENSION_TYPE_NAME_KEY.to_string(),
        ARROW_FILE_EXTENSION_NAME.to_string(),
    );
    let metadata = metadata.map(|metadata| {
        (
            EXTENSION_TYPE_METADATA_KEY.to_string(),
            metadata.to_string(),
        )
    });
    Field::new("file", DataType::Struct(Fields::from(children)), true)
        .with_metadata([name].into_iter().chain(metadata).collect::<Metadata>())
}

/// A field that names `arrow.parquet.file` imports when it follows the canonical spec, which
/// allows any subset of the fields in any order, each string type and each binary type.
#[rstest]
#[case::all_views(
    vec![
        Field::new("uri", DataType::Utf8View, true),
        Field::new("offset", DataType::Int64, true),
        Field::new("size", DataType::Int64, true),
        Field::new("content_type", DataType::Utf8View, true),
        Field::new("checksum", DataType::Utf8View, true),
        Field::new("inline", DataType::BinaryView, true),
    ],
    None,
    &FileRefField::ALL,
)]
#[case::reordered_large(
    vec![
        Field::new("checksum", DataType::LargeUtf8, true),
        Field::new("uri", DataType::Utf8, true),
    ],
    Some(""),
    &[FileRefField::Checksum, FileRefField::Uri],
)]
#[case::inline_only(
    vec![
        Field::new("inline", DataType::LargeBinary, true),
        Field::new("content_type", DataType::Utf8, true),
    ],
    Some(""),
    &[FileRefField::Inline, FileRefField::ContentType],
)]
fn imports_fields_in_the_spec(
    #[case] children: Vec<Field>,
    #[case] metadata: Option<&str>,
    #[case] expected: &[FileRefField],
) -> VortexResult<()> {
    assert_eq!(
        SESSION
            .arrow()
            .from_arrow_field(&arrow_file_field(children, metadata))?,
        file_ref_dtype(expected, Nullability::Nullable)?
    );
    Ok(())
}

/// A field that names `arrow.parquet.file` but breaks the spec fails to import instead of
/// importing as plain storage.
#[rstest]
#[case::metadata(vec![Field::new("uri", DataType::Utf8, true)], Some("{}"), "must be empty")]
#[case::no_fields(vec![], None, "at least one field")]
#[case::unknown_field(vec![Field::new("url", DataType::Utf8, true)], None, "unknown field")]
#[case::duplicate(
    vec![Field::new("uri", DataType::Utf8, true), Field::new("uri", DataType::Utf8, true)],
    None,
    "more than once"
)]
#[case::wrong_type(vec![Field::new("size", DataType::Int32, true)], None, "cannot have type")]
#[case::non_nullable(vec![Field::new("uri", DataType::Utf8, false)], None, "must be nullable")]
#[case::nested_extension(
    vec![Field::new("uri", DataType::Utf8, true).with_metadata(
        [(EXTENSION_TYPE_NAME_KEY.to_string(), "arrow.json".to_string())].into_iter().collect::<Metadata>()
    )],
    None,
    "must not be an extension type"
)]
fn rejects_fields_outside_the_spec(
    #[case] children: Vec<Field>,
    #[case] metadata: Option<&str>,
    #[case] expected: &str,
) {
    let error = SESSION
        .arrow()
        .from_arrow_field(&arrow_file_field(children, metadata))
        .err()
        .map(|error| error.to_string());
    assert!(
        error.as_deref().is_some_and(|error| error.contains(expected)),
        "{error:?}"
    );
}

#[test]
fn rejects_non_struct_storage() {
    let field = Field::new("file", DataType::Utf8, true).with_metadata(
        [(
            EXTENSION_TYPE_NAME_KEY.to_string(),
            ARROW_FILE_EXTENSION_NAME.to_string(),
        )]
        .into_iter()
        .collect::<Metadata>(),
    );
    let error = SESSION.arrow().from_arrow_field(&field).err();
    assert!(
        error.is_some_and(|error| error.to_string().contains("must be a struct")),
        "a non-struct arrow.parquet.file field imported"
    );
}

/// The default export names the canonical extension with empty metadata, and imports back to the
/// same dtype.
#[test]
fn exports_the_canonical_extension() -> VortexResult<()> {
    let dtype = file_ref_dtype(
        &[FileRefField::Uri, FileRefField::Size],
        Nullability::Nullable,
    )?;
    let field = SESSION.arrow().to_arrow_field("file", &dtype)?;
    assert_eq!(field.extension_type_name(), Some(ARROW_FILE_EXTENSION_NAME));
    assert_eq!(field.extension_type_metadata(), Some(""));
    assert_eq!(SESSION.arrow().from_arrow_field(&field)?, dtype);
    Ok(())
}

/// A column exports back to the Arrow field and values that it imports from, including string
/// widths, field order and null rows.
#[test]
fn arrays_roundtrip_through_arrow() -> VortexResult<()> {
    let children = vec![
        Field::new("inline", DataType::BinaryView, true),
        Field::new("uri", DataType::LargeUtf8, true),
        Field::new("size", DataType::Int64, true),
        Field::new("checksum", DataType::Utf8, true),
    ];
    let field = arrow_file_field(children.clone(), Some(""));
    let array: ArrowArrayRef = Arc::new(ArrowStructArray::try_new(
        Fields::from(children),
        vec![
            Arc::new(BinaryViewArray::from(vec![Some(b"png".as_slice()), None, None])),
            Arc::new(LargeStringArray::from(vec![None, Some("s3://b/k"), None])),
            Arc::new(Int64Array::from(vec![None, Some(42), None])),
            Arc::new(StringArray::from(vec![None, Some("md5:00"), None])),
        ],
        Some(NullBuffer::from(vec![true, true, false])),
    )?);

    let imported = SESSION
        .arrow()
        .from_arrow_array(Arc::clone(&array), &field)?;
    assert_eq!(
        imported.dtype(),
        &file_ref_dtype(
            &[
                FileRefField::Inline,
                FileRefField::Uri,
                FileRefField::Size,
                FileRefField::Checksum,
            ],
            Nullability::Nullable,
        )?
    );

    let mut ctx = SESSION.create_execution_ctx();
    let exported = imported.execute_arrow(Some(&field), &mut ctx)?;
    assert_eq!(exported.to_data(), array.to_data());
    Ok(())
}

/// Export refuses an Arrow field whose fields differ from the column's.
#[test]
fn export_rejects_other_fields() -> VortexResult<()> {
    let children = vec![Field::new("uri", DataType::Utf8, true)];
    let array: ArrowArrayRef = Arc::new(ArrowStructArray::try_new(
        Fields::from(children.clone()),
        vec![Arc::new(StringArray::from(vec![Some("a")]))],
        None,
    )?);
    let imported = SESSION
        .arrow()
        .from_arrow_array(array, &arrow_file_field(children, None))?;

    let target = arrow_file_field(
        vec![
            Field::new("uri", DataType::Utf8, true),
            Field::new("size", DataType::Int64, true),
        ],
        None,
    );
    let mut ctx = SESSION.create_execution_ctx();
    assert!(imported.execute_arrow(Some(&target), &mut ctx).is_err());
    Ok(())
}

#[test]
fn initialize_joins_the_unstable_edition_once() {
    let session = VortexSession::empty();
    initialize(&session);
    initialize(&session);
    let members = session
        .editions()
        .components_in(&UNSTABLE, ComponentKind::DType)
        .into_iter()
        .map(|inclusion| inclusion.component_id)
        .collect::<Vec<_>>();
    assert_eq!(members, [ExtVTable::id(&FileRef)]);
}
