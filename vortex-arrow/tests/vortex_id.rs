// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

//! Tests for the Arrow export and import of an extension type without an Arrow plugin. Such a type
//! exports its storage, plus Arrow extension metadata that holds its Vortex ID.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::DictionaryArray;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::Datum;
use vortex_arrow::ext_metadata::EXTENSION_METADATA_ENCODING_KEY;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// A session that can look up [`Tagged`] by its Vortex ID.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    session.dtypes().register(Tagged);
    vortex_arrow::initialize(&session);
    session
});

static TAGGED_ID: CachedId = CachedId::new("test.tagged");

/// An extension type without an Arrow plugin. It accepts any storage, and its metadata is one tag
/// byte.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Tagged;

impl ExtVTable for Tagged {
    type Metadata = u8;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        *TAGGED_ID
    }

    fn serialize_metadata(&self, tag: &u8) -> VortexResult<Vec<u8>> {
        Ok(vec![*tag])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<u8> {
        match metadata {
            [tag] => Ok(*tag),
            _ => vortex_bail!("Tagged metadata must be one byte, got {metadata:?}"),
        }
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

fn tagged(tag: u8, storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
    Ok(ExtDType::try_with_vtable(Tagged, tag, storage_dtype)?.erased())
}

fn tagged_array(tag: u8, storage: ArrayRef) -> VortexResult<ArrayRef> {
    let ext_dtype = tagged(tag, storage.dtype().clone())?;
    Ok(ExtensionArray::try_new(ext_dtype, storage)?.into_array())
}

/// A [`Tagged`] array over the nullable `i64` storage `[1, null, 3]`.
fn tagged_i64s(tag: u8) -> VortexResult<ArrayRef> {
    let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array();
    tagged_array(tag, storage)
}

/// A struct whose only field holds [`tagged_i64s`].
fn struct_of_tagged() -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[("t", tagged_i64s(0xff)?)])?.into_array())
}

/// The lists `[[1, null], [3]]` over [`tagged_i64s`].
fn list_of_tagged() -> VortexResult<ArrayRef> {
    let elements = tagged_i64s(0xff)?;
    let offsets = buffer![0i32, 2, 3].into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let lists = ListArray::try_new(elements, offsets, Validity::NonNullable, &mut ctx)?;

    Ok(lists.into_array())
}

/// The millisecond timestamps `[1, 2, 3]`, whose type has a native Arrow type.
fn timestamps() -> VortexResult<ArrayRef> {
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    Ok(ExtensionArray::try_new(ext_dtype, buffer![1i64, 2, 3].into_array())?.into_array())
}

/// A [`Tagged`] array over [`timestamps`]. The storage field of a timestamp is a native Arrow type
/// without an extension name, so [`Tagged`] can add its own.
fn tagged_timestamps() -> VortexResult<ArrayRef> {
    tagged_array(0xff, timestamps()?)
}

/// The field from [`ArrowSession::to_arrow_storage_field`] for `ext_dtype`, with the Arrow type
/// `storage`.
fn field_with_ext_metadata(
    name: &str,
    storage: DataType,
    ext_dtype: &ExtDTypeRef,
) -> VortexResult<Field> {
    let field = SESSION.arrow().to_arrow_storage_field(name, ext_dtype)?;

    Ok(field.with_data_type(storage))
}

/// An Arrow field of type `Int64` that names [`Tagged`], plus the given field metadata.
fn tagged_int64_field(metadata: &[(&str, &str)]) -> Field {
    let mut field = Field::new("t", DataType::Int64, true);
    field
        .metadata_mut()
        .insert(EXTENSION_TYPE_NAME_KEY.to_owned(), "test.tagged".to_owned());
    field.metadata_mut().extend(
        metadata
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned())),
    );

    field
}

#[test]
fn export_field_holds_vortex_id_and_marked_base64_metadata() -> VortexResult<()> {
    let field = SESSION
        .arrow()
        .to_arrow_field("t", tagged_i64s(0xff)?.dtype())?;

    // The metadata `[0xff]` is not UTF-8, so the field carries it in padded base64.
    let expected_field = Field::new("t", DataType::Int64, true).with_metadata(
        [
            (EXTENSION_TYPE_NAME_KEY.to_owned(), "test.tagged".to_owned()),
            (EXTENSION_TYPE_METADATA_KEY.to_owned(), "/w==".to_owned()),
            (
                EXTENSION_METADATA_ENCODING_KEY.to_owned(),
                "base64".to_owned(),
            ),
        ]
        .into(),
    );
    assert_eq!(field, expected_field);

    Ok(())
}

/// An export to a target with the right Vortex ID but different metadata would give the values a
/// different meaning, so it must fail. A session that knows the Vortex ID compares dtypes, and any
/// other session compares the metadata bytes.
#[rstest]
#[case::registered(SESSION.arrow().clone(), "must name test.tagged[255]")]
#[case::unregistered(ArrowSession::default(), "must carry the metadata of test.tagged[255]")]
fn export_rejects_target_with_other_metadata(
    #[case] session: ArrowSession,
    #[case] message: &str,
) -> VortexResult<()> {
    let other_dtype = tagged(7, DType::Primitive(PType::I64, Nullability::Nullable))?;
    let target = session.to_arrow_field("t", &DType::Extension(other_dtype))?;

    let err = session
        .execute_arrow(
            tagged_i64s(0xff)?,
            Some(&target),
            &mut SESSION.create_execution_ctx(),
        )
        .expect_err("a target with other metadata must be rejected");
    assert!(err.to_string().contains(message), "unexpected error: {err}");

    Ok(())
}

/// An Arrow field has only one extension name, so the storage of an extension type without a plugin
/// must not have its own.
#[test]
fn export_rejects_storage_with_extension_name() -> VortexResult<()> {
    let inner = tagged(1, DType::Primitive(PType::I64, Nullability::NonNullable))?;
    let dtype = DType::Extension(tagged(2, DType::Extension(inner))?);

    let err = SESSION
        .arrow()
        .to_arrow_field("t", &dtype)
        .expect_err("storage with its own extension name must be rejected");
    assert!(
        err.to_string()
            .contains("must not be an Arrow extension type, got test.tagged"),
        "unexpected error: {err}"
    );

    Ok(())
}

/// An Arrow data type, an export without a target, and a [`Datum`] have no field to hold the
/// extension metadata, so they fail for a top-level type without a plugin.
#[test]
fn exports_without_a_field_reject_types_without_a_plugin() -> VortexResult<()> {
    let array = tagged_i64s(0xff)?;
    let message = "needs an extension type with an Arrow plugin";
    let mut ctx = SESSION.create_execution_ctx();

    let errors = [
        SESSION.arrow().to_arrow_datatype(array.dtype()).err(),
        SESSION
            .arrow()
            .execute_arrow(array.clone(), None, &mut ctx)
            .err(),
        Datum::try_new(&array, &mut ctx).err(),
    ];
    for err in errors {
        let err = err.expect("an export without a field must fail instead of losing the type");
        assert!(err.to_string().contains(message), "unexpected error: {err}");
    }

    Ok(())
}

/// A nested extension type keeps its extension metadata on its child field, so an export without a
/// target succeeds.
#[test]
fn nested_export_without_a_target_keeps_ext_metadata() -> VortexResult<()> {
    let array = struct_of_tagged()?;
    let field = SESSION.arrow().to_arrow_field("", array.dtype())?;
    let mut ctx = SESSION.create_execution_ctx();
    let exported = SESSION.arrow().execute_arrow(array, None, &mut ctx)?;

    assert_eq!(exported.data_type(), field.data_type());

    Ok(())
}

/// Importing with the field that the export used restores the array, wherever the extension type is
/// nested.
#[rstest]
#[case::top_level(tagged_i64s(0xff))]
#[case::struct_field(struct_of_tagged())]
#[case::list_element(list_of_tagged())]
#[case::over_timestamp(tagged_timestamps())]
fn round_trip_keeps_the_extension_type(#[case] array: VortexResult<ArrayRef>) -> VortexResult<()> {
    let array = array?;
    let session = SESSION.arrow();
    let mut ctx = SESSION.create_execution_ctx();
    let field = session.to_arrow_field("t", array.dtype())?;
    let exported = session.execute_arrow(array.clone(), Some(&field), &mut ctx)?;

    assert_eq!(session.from_arrow_field(&field)?, *array.dtype());
    assert_arrays_eq!(session.from_arrow_array(exported, &field)?, array, &mut ctx);

    Ok(())
}

/// A timestamp exported to an `Int64` field with its extension metadata keeps its type.
#[test]
fn timestamp_round_trips_through_its_storage() -> VortexResult<()> {
    let timestamps = timestamps()?;
    let field = field_with_ext_metadata("ts", DataType::Int64, timestamps.dtype().as_extension())?;
    let mut ctx = SESSION.create_execution_ctx();

    let exported = SESSION
        .arrow()
        .execute_arrow(timestamps.clone(), Some(&field), &mut ctx)?;
    assert_eq!(exported.data_type(), &DataType::Int64);

    let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
    assert_arrays_eq!(imported, timestamps, &mut ctx);

    Ok(())
}

/// A session that does not know the Vortex ID imports the field as plain storage.
#[test]
fn unknown_vortex_id_imports_as_storage() -> VortexResult<()> {
    let field = SESSION
        .arrow()
        .to_arrow_field("t", tagged_i64s(0xff)?.dtype())?;
    let arrow: ArrowArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
    let session = ArrowSession::default();

    let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]);
    assert_eq!(session.from_arrow_field(&field)?, *storage.dtype());
    assert_arrays_eq!(
        session.from_arrow_array(arrow, &field)?,
        storage,
        &mut SESSION.create_execution_ctx()
    );

    Ok(())
}

/// Metadata without the encoding key is its own UTF-8 bytes, as other producers of these fields
/// write it.
#[test]
fn unmarked_metadata_imports_as_its_text() -> VortexResult<()> {
    let field = tagged_int64_field(&[(EXTENSION_TYPE_METADATA_KEY, "x")]);
    let expected = tagged(b'x', DType::Primitive(PType::I64, Nullability::Nullable))?;

    assert_eq!(
        SESSION.arrow().from_arrow_field(&field)?,
        DType::Extension(expected)
    );

    Ok(())
}

/// A dictionary imports as a [`Dict`] whose values have the extension type, which matches what the
/// importer plugins produce.
#[test]
fn dictionary_values_keep_the_extension_type() -> VortexResult<()> {
    let ext_dtype = tagged(0xff, DType::Primitive(PType::I64, Nullability::NonNullable))?;
    let dictionary: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1, 0]),
        Arc::new(Int64Array::from(vec![10, 20])),
    )?);
    let field = field_with_ext_metadata("t", dictionary.data_type().clone(), &ext_dtype)?;

    let imported = SESSION.arrow().from_arrow_array(dictionary, &field)?;
    assert!(
        imported.is::<Dict>(),
        "expected a Dict encoding, got {imported}"
    );
    assert_eq!(
        SESSION.arrow().from_arrow_field(&field)?,
        DType::Extension(ext_dtype.clone())
    );

    let expected = ExtensionArray::try_new(ext_dtype, buffer![10i64, 20, 10].into_array())?;
    assert_arrays_eq!(imported, expected, &mut SESSION.create_execution_ctx());

    Ok(())
}

/// If the extension name of a field is a known Vortex ID, its metadata must be valid for that type.
/// A missing `ARROW:extension:metadata` means empty metadata, and unmarked metadata is its own
/// text. [`Tagged`] rejects both here, because neither is one byte.
#[rstest]
#[case::not_base64(
    &[(EXTENSION_TYPE_METADATA_KEY, "%"), (EXTENSION_METADATA_ENCODING_KEY, "base64")],
    "must carry base64 extension metadata"
)]
#[case::unknown_encoding(
    &[(EXTENSION_TYPE_METADATA_KEY, "ff"), (EXTENSION_METADATA_ENCODING_KEY, "hex")],
    r#"must have the extension metadata encoding "base64", got "hex""#
)]
#[case::missing(&[], "Tagged metadata must be one byte, got []")]
#[case::unmarked_base64(
    &[(EXTENSION_TYPE_METADATA_KEY, "/w==")],
    "Tagged metadata must be one byte, got [47, 119, 61, 61]"
)]
fn import_rejects_bad_metadata(
    #[case] metadata: &[(&str, &str)],
    #[case] message: &str,
) -> VortexResult<()> {
    let field = tagged_int64_field(metadata);

    let err = SESSION
        .arrow()
        .from_arrow_field(&field)
        .expect_err("invalid metadata must be rejected");
    assert!(err.to_string().contains(message), "unexpected error: {err}");

    Ok(())
}
