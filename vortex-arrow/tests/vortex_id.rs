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
use arrow_array::MapArray;
use arrow_array::RunArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::types::Int32Type;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
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
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::Datum;
use vortex_arrow::ExecuteArrowExt;
use vortex_arrow::ext_metadata::EXTENSION_METADATA_ENCODING_KEY;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// A session that can look up [`Tagged`] by its Vortex ID and by [`TAGGED_ALIAS`].
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    session.dtypes().register(Tagged);
    session
        .dtypes()
        .registry()
        .insert(*TAGGED_ALIAS_ID, Arc::new(Tagged));
    session
});

static TAGGED_ID: CachedId = CachedId::new("test.tagged");

/// Another name that the dtype registry of [`SESSION`] resolves to [`Tagged`].
const TAGGED_ALIAS: &str = "test.tagged.alias";
static TAGGED_ALIAS_ID: CachedId = CachedId::new(TAGGED_ALIAS);

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
    Ok(ExtensionArray::try_new_from_vtable(Tagged, tag, storage)?.into_array())
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

/// The field from [`ArrowSession::to_arrow_storage_field`] for the type of `array`, with the Arrow
/// type `data_type`. It takes a [`VortexResult`] so that an rstest case can pass a fixture call.
///
/// [`ArrowSession::to_arrow_storage_field`]: vortex_arrow::ArrowSession::to_arrow_storage_field
fn storage_field_as(array: VortexResult<ArrayRef>, data_type: DataType) -> VortexResult<Field> {
    let field = SESSION
        .arrow()
        .to_arrow_storage_field("t", array?.dtype().as_extension())?;

    Ok(field.with_data_type(data_type))
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

/// An export to a target that names a Vortex ID fails if importing the target would give the
/// values another type or another meaning.
#[rstest]
#[case::other_metadata(
    tagged_i64s(0xff),
    storage_field_as(tagged_i64s(7), DataType::Int64),
    "must carry the metadata of test.tagged[255](i64?), got [7]"
)]
#[case::other_storage_type(
    tagged_i64s(0xff),
    storage_field_as(tagged_i64s(0xff), DataType::Int32),
    "must have an Arrow type that imports as the storage of test.tagged[255](i64?), got Int32"
)]
#[case::other_time_unit(
    timestamps(),
    storage_field_as(timestamps(), DataType::Timestamp(ArrowTimeUnit::Second, None)),
    "must have an Arrow type that imports as the storage of vortex.timestamp"
)]
#[case::other_vortex_id(
    timestamps(),
    storage_field_as(tagged_i64s(0xff), DataType::Int64),
    "names the extension type test.tagged, so the array must have that type"
)]
fn export_rejects_target_that_changes_the_type(
    #[case] array: VortexResult<ArrayRef>,
    #[case] target: VortexResult<Field>,
    #[case] message: &str,
) -> VortexResult<()> {
    let err = array?
        .execute_arrow(Some(&target?), &mut SESSION.create_execution_ctx())
        .expect_err("a target that changes the type must be rejected");
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
#[rstest]
#[case::to_arrow_datatype(|array: &ArrayRef, _: &mut ExecutionCtx| {
    SESSION.arrow().to_arrow_datatype(array.dtype()).map(drop)
})]
#[case::execute_arrow(|array: &ArrayRef, ctx: &mut ExecutionCtx| {
    array.clone().execute_arrow(None, ctx).map(drop)
})]
#[case::datum(|array: &ArrayRef, ctx: &mut ExecutionCtx| Datum::try_new(array, ctx).map(drop))]
fn exports_without_a_field_reject_types_without_a_plugin(
    #[case] export: fn(&ArrayRef, &mut ExecutionCtx) -> VortexResult<()>,
) -> VortexResult<()> {
    let err = export(&tagged_i64s(0xff)?, &mut SESSION.create_execution_ctx())
        .expect_err("an export without a field must fail instead of losing the type");
    assert!(
        err.to_string()
            .contains("needs an extension type with an Arrow plugin"),
        "unexpected error: {err}"
    );

    Ok(())
}

/// A nested extension type keeps its extension metadata on its child field, so an export without a
/// target succeeds.
#[test]
fn nested_export_without_a_target_keeps_ext_metadata() -> VortexResult<()> {
    let array = struct_of_tagged()?;
    let field = SESSION.arrow().to_arrow_field("", array.dtype())?;
    let mut ctx = SESSION.create_execution_ctx();
    let exported = array.execute_arrow(None, &mut ctx)?;

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
    let exported = array.clone().execute_arrow(Some(&field), &mut ctx)?;

    assert_eq!(session.from_arrow_field(&field)?, *array.dtype());
    assert_arrays_eq!(session.from_arrow_array(exported, &field)?, array, &mut ctx);

    Ok(())
}

/// A timestamp exported to a field that names `vortex.timestamp` keeps its type, whether the field
/// has the storage type or the native Arrow timestamp type.
#[rstest]
#[case::storage(DataType::Int64)]
#[case::arrow_timestamp(DataType::Timestamp(ArrowTimeUnit::Millisecond, None))]
fn timestamp_round_trips_through_a_vortex_id_field(
    #[case] data_type: DataType,
) -> VortexResult<()> {
    let timestamps = timestamps()?;
    let field = storage_field_as(Ok(timestamps.clone()), data_type.clone())?;
    let mut ctx = SESSION.create_execution_ctx();

    let exported = timestamps.clone().execute_arrow(Some(&field), &mut ctx)?;
    assert_eq!(exported.data_type(), &data_type);

    assert_eq!(
        SESSION.arrow().from_arrow_field(&field)?,
        *timestamps.dtype()
    );
    let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
    assert_arrays_eq!(imported, timestamps, &mut ctx);

    Ok(())
}

/// A field that names an alias of a Vortex ID exports and imports like a field that names the ID.
#[test]
fn alias_of_vortex_id_round_trips() -> VortexResult<()> {
    let array = tagged_i64s(0xff)?;
    let mut field = storage_field_as(Ok(array.clone()), DataType::Int64)?;
    field
        .metadata_mut()
        .insert(EXTENSION_TYPE_NAME_KEY.to_owned(), TAGGED_ALIAS.to_owned());
    let mut ctx = SESSION.create_execution_ctx();

    let exported = array.clone().execute_arrow(Some(&field), &mut ctx)?;
    let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
    assert_arrays_eq!(imported, array, &mut ctx);

    Ok(())
}

/// A plain array exports to a field that names a Vortex ID, because a plugin such as `Vector`
/// exports its storage to the same field that it received.
#[test]
fn plain_array_exports_to_a_vortex_id_field() -> VortexResult<()> {
    let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array();
    let field = storage_field_as(tagged_i64s(0xff), DataType::Int64)?;
    let mut ctx = SESSION.create_execution_ctx();

    let exported = storage.clone().execute_arrow(Some(&field), &mut ctx)?;
    let plain_field = Field::new("t", DataType::Int64, true);
    let imported = SESSION.arrow().from_arrow_array(exported, &plain_field)?;
    assert_arrays_eq!(imported, storage, &mut ctx);

    Ok(())
}

/// The export to a field that names a Vortex ID compares nested extension types by their storage,
/// because the import of the field may not restore them. Here the session does not know the nested
/// Vortex ID, and an import plugin that loses metadata, such as the version of a UUID, has the same
/// effect.
#[test]
fn export_compares_nested_extension_types_by_storage() -> VortexResult<()> {
    let array = tagged_array(0x01, struct_of_tagged()?)?;
    let field = SESSION
        .arrow()
        .to_arrow_storage_field("t", array.dtype().as_extension())?;
    let session = VortexSession::empty();
    let mut ctx = session.create_execution_ctx();

    let exported = array.clone().execute_arrow(Some(&field), &mut ctx)?;
    let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
    assert_arrays_eq!(imported, array, &mut ctx);

    Ok(())
}

/// A session that does not know the Vortex ID imports the field as plain storage.
#[test]
fn unknown_vortex_id_imports_as_storage() -> VortexResult<()> {
    let field = SESSION
        .arrow()
        .to_arrow_field("t", tagged_i64s(0xff)?.dtype())?;
    let arrow: ArrowArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
    let unaware_session = VortexSession::empty();
    let session = unaware_session.arrow();

    let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]);
    assert_eq!(session.from_arrow_field(&field)?, *storage.dtype());
    assert_arrays_eq!(
        session.from_arrow_array(arrow, &field)?,
        storage,
        &mut SESSION.create_execution_ctx()
    );

    Ok(())
}

/// The Arrow session of a Vortex session knows the extension types of its dtype session, including
/// the ones that are registered after the Arrow session is created.
#[test]
fn vortex_id_imports_types_registered_after_the_arrow_session() -> VortexResult<()> {
    let array = tagged_i64s(0xff)?;
    let session = VortexSession::empty();
    let field = session.arrow().to_arrow_field("t", array.dtype())?;

    session.dtypes().register(Tagged);

    assert_eq!(session.arrow().from_arrow_field(&field)?, *array.dtype());

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
    let field = SESSION
        .arrow()
        .to_arrow_storage_field("t", &ext_dtype)?
        .with_data_type(dictionary.data_type().clone());

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

/// A run-end field whose values name the same Vortex ID imports as that type once, because its
/// values already have the type.
#[test]
fn run_end_field_that_names_its_type_twice_wraps_it_once() -> VortexResult<()> {
    let ext_dtype = tagged(0xff, DType::Primitive(PType::I64, Nullability::Nullable))?;
    let values_field = SESSION
        .arrow()
        .to_arrow_storage_field("values", &ext_dtype)?;
    let metadata = values_field.metadata().clone();
    let run_ends_field = Field::new("run_ends", DataType::Int32, false);
    let data_type = DataType::RunEndEncoded(Arc::new(run_ends_field), Arc::new(values_field));
    let field = Field::new("t", data_type, true).with_metadata(metadata);

    let run_ends = Int32Array::from(vec![2, 3]);
    let values = Int64Array::from(vec![Some(1), None]);
    let arrow: ArrowArrayRef = Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values)?);

    let expected_dtype = DType::Extension(ext_dtype);
    assert_eq!(SESSION.arrow().from_arrow_field(&field)?, expected_dtype);
    assert_eq!(
        *SESSION.arrow().from_arrow_array(arrow, &field)?.dtype(),
        expected_dtype
    );

    Ok(())
}

/// The entries field of a map is part of the map type, so the dtype import and the array import
/// both ignore a Vortex ID on it.
#[test]
fn map_entries_ignore_a_vortex_id() -> VortexResult<()> {
    let key_field = Arc::new(Field::new("key", DataType::Int64, false));
    let value_field = Arc::new(Field::new("value", DataType::Int64, true));
    let entries = ArrowStructArray::new(
        Fields::from(vec![key_field, value_field]),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![Some(3), None])),
        ],
        None,
    );
    let entries_field = tagged_int64_field(&[])
        .with_name("entries")
        .with_data_type(entries.data_type().clone())
        .with_nullable(false);
    let map = MapArray::try_new(
        Arc::new(entries_field),
        OffsetBuffer::new(vec![0, 2].into()),
        entries,
        None,
        false,
    )?;
    let field = Field::new("m", map.data_type().clone(), false);

    let dtype = SESSION.arrow().from_arrow_field(&field)?;
    let imported = SESSION.arrow().from_arrow_array(Arc::new(map), &field)?;
    assert_eq!(*imported.dtype(), dtype);

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
    let err = SESSION
        .arrow()
        .from_arrow_field(&tagged_int64_field(metadata))
        .expect_err("invalid metadata must be rejected");
    assert!(err.to_string().contains(message), "unexpected error: {err}");

    Ok(())
}
